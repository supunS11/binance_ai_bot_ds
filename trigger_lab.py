"""Detector-level replay harness - step 0.1 of the trigger-restructuring
plan (2026-09-21 architecture audit).

WHAT THIS IS, AND WHY IT IS NOT backtest.py
-------------------------------------------
backtest.py replays the FULL strategy (signal_engine.evaluate ->
risk_manager.build_trade_plan -> main._evaluate_symbol), which is exactly
right for integration checking but carries two properties that make it
unusable for per-trigger validation:

  1. It is capped at ~2 days of real range, because CVD is a REQUIRED,
     gating input and exchange.get_historical_agg_trades hard-rejects any
     startTime older than ~2 days (_AGG_TRADES_MAX_LOOKBACK_MS). Anything
     older evaluates as ORDER_FLOW_DATA_UNAVAILABLE.
  2. Every candidate it produces has already passed the whole gate stack,
     so it measures "trigger AND 16-24 gates", never the trigger alone.

This module answers the different question the audit needs: *when this
detector fires, does the setup have an edge* - independent of gates, at a
sample size live trading would need months to reach. It calls the REAL
detector functions (market_structure / liquidity_sweep), never a
reimplementation, so a result here transfers to production behaviour.

Klines are the one historical source with no depth limit
(exchange.get_historical_klines paginates arbitrarily far back and caches
to disk itself), so every kline-only trigger reaches n>1000 in a single
run. The three feed-dependent sources - CVD, open interest, liquidations -
are deliberately NOT reconstructed here; see PLAN STEP 0.3 for why they
need forward collection instead.

SAFETY
------
Imports only config / exchange / market_structure / liquidity_sweep.
Deliberately does NOT import main, execution, position_manager or
risk_manager - nothing here can place, modify or cancel an order, and no
code path touches live state. Only public REST kline endpoints are called.

FIDELITY BOUNDARIES (state these with any result)
-------------------------------------------------
  - Evaluates once per CLOSED candle. Live evaluates every
    SIGNAL_EVAL_INTERVAL_SECONDS against a forming candle, but
    REQUIRE_CLOSE_CONFIRMED_BREAK=True means the detectors themselves read
    the last CLOSED candle either way, so detection is faithful; only the
    entry price is approximated (the detection candle's close, versus
    whatever the live price was mid-candle).
  - Stop uses the _apply_min_stop_distance FLOOR model
    (max(entry*MIN_STOP_DISTANCE_PCT/100, atr*MIN_STOP_DISTANCE_ATR_MULTIPLE)),
    not the full structure-based stop. This matches the methodology the
    prior reject-journal audits used, so results stay comparable to them.
  - First touch wins; a bar touching BOTH stop and target is scored as a
    LOSS. Never relax this - the project's established convention.
  - No gates are applied. That is the point.

Usage
-----
    python trigger_lab.py --triggers ORDER_BLOCK_RETEST --days 90 --symbols-top 60
    python trigger_lab.py --triggers ALL --days 90 --symbols-top 120 --out data/lab
"""
import argparse
import csv
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

import config
import exchange
import liquidity_sweep
import market_structure

# Matches WS_KLINE_HISTORY_LIMIT: live hands the detectors this many
# candles, so the replay window must be the same size or structure
# lookbacks see a different amount of history than production does.
WINDOW = int(getattr(config, "WS_KLINE_HISTORY_LIMIT", 200))

# How far forward a detection is allowed to resolve before it is called a
# timeout. 48 x 1h matches the prior audits' 48h cap.
FORWARD_CAP = 48

OUT_DIR_DEFAULT = "data/lab"


# --------------------------------------------------------------------------
# candles
# --------------------------------------------------------------------------
def _build_candle(row):
    """Identical shape to backtest.py's own _build_candle - deliberately
    duplicated rather than imported, because importing backtest.py pulls
    in main.py and execution.py with it (see SAFETY above)."""
    return {
        "open_time": int(row["time"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "closed": True,
    }


def load_candles(symbol, interval, start_ms, end_ms):
    df = exchange.get_historical_klines(symbol, interval, start_ms, end_ms)

    if df is None or getattr(df, "empty", True):
        return []

    return [_build_candle(row) for _, row in df.iterrows()]


def top_symbols(n):
    """Top-n by 24h quote volume - the same selection main.py's watchlist
    refresh uses, so the replay universe matches what the bot actually
    scans rather than an arbitrary list."""
    try:
        volumes = exchange.get_24h_quote_volumes()
    except Exception as exc:
        print(f"  ! could not fetch volumes: {exc}")
        return []

    if not volumes:
        return []

    if isinstance(volumes, dict):
        pairs = sorted(volumes.items(), key=lambda kv: kv[1], reverse=True)
        syms = [s for s, _ in pairs]
    else:
        syms = list(volumes)

    return [s for s in syms if s.endswith("USDT")][:n]


# --------------------------------------------------------------------------
# detectors - every one calls the REAL production function
# --------------------------------------------------------------------------
def _d_order_block_retest(window):
    return market_structure.find_order_block_retest(window)


def _d_ob_fvg_retest(window):
    return market_structure.find_fvg_retest(window)


def _d_liquidity_sweep(window):
    pools = market_structure.find_liquidity_pools(
        market_structure.find_swing_points(window)
    )
    return liquidity_sweep.detect_sweep(window, pools)


def _d_structure_break(window):
    structure = market_structure.structure_state(window)
    brk = market_structure.live_break_check(window, structure)

    if not brk or not brk.get("broken"):
        return None

    return {
        "direction": brk["direction"],
        "level": brk.get("level"),
        "open_time": brk.get("open_time") or window[-1]["open_time"],
    }


def _d_ema_pullback(window):
    ema_value = market_structure.exponential_moving_average(window)

    if ema_value is None:
        return None

    return market_structure.detect_ema_pullback(window, ema_value)


def _d_choch_retest(window):
    analysis = market_structure.analyze(window)
    last_event = analysis.get("last_event")

    if not last_event or last_event.get("type") != "CHoCH":
        return None

    age = len(window) - 1 - last_event["index"]
    direction = last_event["direction"]
    level = (
        analysis.get("last_swing_low") if direction == "BULLISH"
        else analysis.get("last_swing_high")
    )

    if level is None:
        return None

    retest = market_structure.detect_level_pullback(window, level)

    if retest is None or retest.get("direction") != direction:
        return None

    # Deliberately NOT applying CHOCH_TRIGGER_MIN/MAX_AGE_CANDLES here -
    # measuring the full age curve is the entire point of plan step 3.2
    # (the live 9-10 band came from n=13 with zero observations at age 9).
    return {
        "direction": direction,
        "level": level,
        "open_time": retest.get("open_time") or window[-1]["open_time"],
        "choch_age": age,
    }


def _d_random(window):
    """CONTROL. Fires on a random candle in a random direction, with no
    reference to structure at all, and rides the identical stop model,
    forward walk and scoring as every real detector.

    This is the measurement that makes the others interpretable. If a real
    trigger cannot beat this, it is not selecting anything - and if this
    control came out materially different from the random-walk expectation
    (33.3% wins at a -1R/+2R barrier), the harness itself would be suspect.
    Seeded off the candle's own open_time so a re-run reproduces exactly.
    """
    seed = (window[-1]["open_time"] // 3600000) * 2654435761 % 2147483647

    if seed % 23 != 0:          # ~4.3% of candles, comparable to real rates
        return None

    return {
        "direction": "BULLISH" if (seed >> 8) % 2 else "BEARISH",
        "level": None,          # no structure -> floor stop, by construction
        "open_time": window[-1]["open_time"],
    }


DETECTORS = {
    "RANDOM_CONTROL": _d_random,
    "ORDER_BLOCK_RETEST": _d_order_block_retest,
    "OB_FVG_RETEST": _d_ob_fvg_retest,
    "LIQUIDITY_SWEEP": _d_liquidity_sweep,
    "STRUCTURE_BREAK": _d_structure_break,
    "EMA_PULLBACK": _d_ema_pullback,
    "CHOCH_RETEST": _d_choch_retest,
}


# --------------------------------------------------------------------------
# trigger-specific features (plan phase 5 - "what should this trigger own")
# --------------------------------------------------------------------------
def _features(trigger, window, det, atr):
    latest = window[-1]
    entry = latest["close"]
    rng = max(latest["high"] - latest["low"], 1e-12)
    feats = {
        "body_atr": abs(latest["close"] - latest["open"]) / atr if atr else "",
        "close_pos": (latest["close"] - latest["low"]) / rng,
        "vol_ratio": "",
    }

    vols = [c["volume"] for c in window[-21:-1] if c["volume"] > 0]

    if vols:
        med = statistics.median(vols)
        feats["vol_ratio"] = latest["volume"] / med if med else ""

    if trigger == "ORDER_BLOCK_RETEST":
        block = det.get("block") or {}
        hi, lo = block.get("high"), block.get("low")

        if hi is not None and lo is not None:
            feats["size_atr"] = (hi - lo) / atr if atr else ""
            feats["block_age"] = (len(window) - 1) - block.get("index", 0)
            # Does the close actually RECLAIM past the block edge, or does
            # it merely sit inside it? The live detector accepts the
            # latter (market_structure.py:274-335) - plan step 3.1. This
            # records which case each detection is, so the fix can be
            # measured before it ships.
            if det["direction"] == "BULLISH":
                feats["reclaimed"] = int(entry > hi)
                feats["pierce_atr"] = (hi - latest["low"]) / atr if atr else ""
            else:
                feats["reclaimed"] = int(entry < lo)
                feats["pierce_atr"] = (latest["high"] - lo) / atr if atr else ""

    elif trigger == "LIQUIDITY_SWEEP":
        level = det.get("level")

        if level is not None and atr:
            feats["wick_atr"] = (
                (level - latest["low"]) / atr if det["direction"] == "BULLISH"
                else (latest["high"] - level) / atr
            )

    elif trigger == "CHOCH_RETEST":
        feats["choch_age"] = det.get("choch_age", "")

    return feats


# --------------------------------------------------------------------------
# forward walk
# --------------------------------------------------------------------------
def _min_stop_distance(entry, atr):
    """_apply_min_stop_distance's floor, reproduced exactly
    (risk_manager.py:520-522)."""
    min_pct = max(float(config.MIN_STOP_DISTANCE_PCT), 0) / 100
    min_atr_multiple = max(float(config.MIN_STOP_DISTANCE_ATR_MULTIPLE), 0)
    return max(entry * min_pct, float(atr or 0) * min_atr_multiple)


def _apply_min_stop_distance(sl_price, entry, side, atr=0):
    """risk_manager._apply_min_stop_distance, reproduced (risk_manager.py:497)."""
    if entry <= 0:
        return sl_price

    min_distance = _min_stop_distance(entry, atr)

    if min_distance <= 0 or abs(entry - sl_price) >= min_distance:
        return sl_price

    return entry - min_distance if side == "BUY" else entry + min_distance


def stop_price(level, entry, side, atr, model="structure"):
    """Mirrors risk_manager.compute_stop_loss under the LIVE configuration
    (SL_TP_USE_HTF_STRUCTURE=False, confirmed on the VPS): the stop sits
    just beyond the trigger's OWN structure_level, offset by
    STRUCTURE_STOP_ATR_BUFFER, then floored.

    This distinction turned out to decide the whole measurement. A stop at
    a FIXED ATR distance makes every trade a symmetric -1R/+2R barrier
    around an essentially arbitrary point, and a symmetric barrier on a
    near-random walk returns the random-walk answer - 33.3% win rate - no
    matter which detector chose the entry. That is exactly what the first
    run produced for all six triggers (32.1-33.8%).

    A structure stop is different in kind: it sits beyond the level the
    setup is predicated on, so `risk` varies with how far price is from
    its own invalidation point, and being wrong means the thesis actually
    broke rather than that price wandered one ATR.

    `model="floor"` keeps the old behaviour for comparison.
    """
    if entry <= 0:
        return None

    if model == "floor" or level is None:
        d = _min_stop_distance(entry, atr)
        return (entry - d) if side == "BUY" else (entry + d)

    buffer = float(atr or 0) * max(float(config.STRUCTURE_STOP_ATR_BUFFER), 0)
    sl = level - buffer if side == "BUY" else level + buffer

    # The level must sit on the ADVERSE side of entry or the stop means
    # nothing. Live never sees this case (the trigger fires ON the level,
    # so entry is at or just past it); when the buffer pushes it through,
    # skip rather than invent a stop on the wrong side of entry.
    if (side == "BUY" and sl >= entry) or (side == "SELL" and sl <= entry):
        return None

    return _apply_min_stop_distance(sl, entry, side, atr)


def forward_walk(candles, start_index, entry, side, risk, target_r, cap=FORWARD_CAP):
    """First touch wins. A bar touching BOTH stop and target is scored a
    LOSS - the project's standing convention, never relaxed."""
    if risk <= 0:
        return None

    if side == "BUY":
        sl, tp = entry - risk, entry + target_r * risk
    else:
        sl, tp = entry + risk, entry - target_r * risk

    mfe = mae = 0.0
    last_close = entry
    bars = 0

    for c in candles[start_index + 1: start_index + 1 + cap]:
        last_close = c["close"]
        bars += 1

        if side == "BUY":
            mfe = max(mfe, (c["high"] - entry) / risk)
            mae = min(mae, (c["low"] - entry) / risk)
            hit_sl, hit_tp = c["low"] <= sl, c["high"] >= tp
        else:
            mfe = max(mfe, (entry - c["low"]) / risk)
            mae = min(mae, (entry - c["high"]) / risk)
            hit_sl, hit_tp = c["high"] >= sl, c["low"] <= tp

        if hit_sl:
            return {"outcome": "SL", "r": -1.0, "mfe_r": mfe, "mae_r": mae, "bars": bars}

        if hit_tp:
            return {"outcome": "TP", "r": float(target_r), "mfe_r": mfe, "mae_r": mae, "bars": bars}

    realized = (
        (last_close - entry) / risk if side == "BUY" else (entry - last_close) / risk
    )
    return {"outcome": "TIMEOUT", "r": realized, "mfe_r": mfe, "mae_r": mae, "bars": bars}


# --------------------------------------------------------------------------
# per-symbol replay
# --------------------------------------------------------------------------
def replay_symbol(symbol, trigger, candles, target_r, overlapping=False,
                  stop_model="structure"):
    """Returns (rows, raw_detection_count).

    NON-OVERLAPPING is the default and matters more than it looks. A setup
    keeps re-detecting on every candle while it remains valid (an EMA
    pullback re-qualifies for as long as price hugs the EMA), so counting
    each one as a trade both inflates n and makes the sample
    non-independent - the "trades" are the same move measured repeatedly,
    which quietly destroys every significance claim built on it.

    Live cannot take those either: one position per symbol, MAX_TOTAL_
    POSITIONS=4. So the faithful model is to take a detection, walk it to
    its resolution, and only then become eligible again - which is exactly
    what a person trading this would do.

    `raw` is still reported separately, because "number of raw detections"
    is its own plan metric (0.4) and describes how often the pattern
    occurs at all.
    """
    detector = DETECTORS[trigger]
    rows = []
    raw = 0
    next_eligible = WINDOW

    for i in range(WINDOW, len(candles)):
        window = candles[i - WINDOW: i + 1]

        try:
            det = detector(window)
        except Exception:
            continue

        if not det:
            continue

        raw += 1

        if not overlapping and i < next_eligible:
            continue

        latest = candles[i]
        entry = latest["close"]
        atr = market_structure.average_true_range(window) or 0
        side = "BUY" if det["direction"] == "BULLISH" else "SELL"
        level = det.get("level")
        sl = stop_price(level, entry, side, atr, stop_model)

        if sl is None:
            continue

        risk = abs(entry - sl)

        if risk <= 0 or entry <= 0:
            continue

        res = forward_walk(candles, i, entry, side, risk, target_r)

        if res is None:
            continue

        row = {
            "symbol": symbol,
            "trigger": trigger,
            "open_time": latest["open_time"],
            "utc": datetime.fromtimestamp(
                latest["open_time"] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "side": side,
            "entry": entry,
            "atr": atr,
            "atr_pct": (atr / entry * 100) if entry else "",
            "risk": risk,
            "risk_pct": (risk / entry * 100) if entry else "",
            # The trigger fires ON its level, so entry sits very close to
            # it and the structure stop is often TIGHTER than the floor -
            # in which case _apply_min_stop_distance widens it and the
            # structural relationship is discarded anyway. Measuring how
            # often that happens decides whether the stop model is really
            # structural in practice or only in intent.
            "floored": int(abs(risk - _min_stop_distance(entry, atr)) < 1e-9),
            "sl": sl,
            "level": level if level is not None else "",
            # How far entry sits past the level it fired against, in R.
            # Live rejects >MAX_ENTRY_EXTENSION_R (0.5) inside
            # build_trade_plan; recorded rather than filtered here so the
            # population can be sliced both ways.
            "extension_r": (abs(entry - level) / risk) if (level is not None and risk) else "",
            "target_r": target_r,
        }
        row.update(_features(trigger, window, det, atr))
        row.update(res)
        rows.append(row)
        next_eligible = i + res["bars"] + 1

    return rows, raw


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summarize(rows, label, raw=None):
    if not rows:
        print(f"  {label}: no detections")
        return

    n = len(rows)
    if raw:
        print(f"  {label}  [raw detections {raw}, non-overlapping trades {n} "
              f"= {n/raw*100:.1f}%]")
    wins = sum(1 for r in rows if r["outcome"] == "TP")
    losses = sum(1 for r in rows if r["outcome"] == "SL")
    touts = n - wins - losses
    exp = sum(r["r"] for r in rows) / n
    target_r = rows[0]["target_r"]
    breakeven = 100.0 / (1.0 + target_r)
    mfes = sorted(r["mfe_r"] for r in rows)

    def pct(p):
        return mfes[min(int(len(mfes) * p), len(mfes) - 1)]

    print(f"  {label}")
    print(f"    n={n}  TP={wins} ({wins/n*100:.1f}%)  SL={losses} ({losses/n*100:.1f}%)  timeout={touts}")
    print(f"    expectancy   {exp:+.3f}R at {target_r}R   (break-even win rate {breakeven:.1f}%)")
    print(f"    MFE median   {statistics.median(mfes):.2f}R   p25 {pct(.25):.2f}  p75 {pct(.75):.2f}  p90 {pct(.90):.2f}")

    for r_level in (1.0, 1.5, 2.0, 2.5):
        reach = sum(1 for m in mfes if m >= r_level) / n * 100
        print(f"    reach {r_level}R      {reach:5.1f}%   -> expectancy if targeted there: "
              f"{(reach/100*r_level - (1-reach/100)):+.3f}R")

    for side in ("BUY", "SELL"):
        sub = [r for r in rows if r["side"] == side]
        if sub:
            print(f"    {side:<4} n={len(sub):<5} exp {sum(r['r'] for r in sub)/len(sub):+.3f}R  "
                  f"win {sum(1 for r in sub if r['outcome']=='TP')/len(sub)*100:.1f}%")

    floored = [r for r in rows if r.get("floored")]
    structural = [r for r in rows if not r.get("floored")]
    print(f"    stop floored {len(floored)/n*100:.1f}%  "
          f"(structural {len(structural)/n*100:.1f}%)")

    for lbl, sub in (("floored", floored), ("structural", structural)):
        if len(sub) >= 50:
            print(f"      {lbl:<11} n={len(sub):<6} exp {sum(r['r'] for r in sub)/len(sub):+.3f}R  "
                  f"win {sum(1 for r in sub if r['outcome']=='TP')/len(sub)*100:.1f}%")


def write_csv(rows, path):
    if not rows:
        return

    keys = []

    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()

        for r in rows:
            w.writerow(r)

    print(f"  wrote {len(rows)} rows -> {path}")


# --------------------------------------------------------------------------
def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggers", default="ORDER_BLOCK_RETEST",
                    help="comma list, or ALL")
    ap.add_argument("--symbols", default="", help="comma list; overrides --symbols-top")
    ap.add_argument("--symbols-top", type=int, default=60)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--interval", default=None, help="default: WS_KLINE_INTERVAL")
    ap.add_argument("--target", type=float, default=None,
                    help="default: TP1_R_MULTIPLE")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    # The stop FLOOR is not a neutral parameter: fee cost in R is
    # (2 x taker) / (risk/entry), so the floor sets the fee hurdle every
    # trade must clear. At MIN_STOP_DISTANCE_PCT=0.6 that hurdle is ~0.167R;
    # at 1.2 it halves. risk_manager's own comment already records that 0.6%
    # "gets hit on ~40% of trades, with every floor-clamped trade resolving
    # as a loss or scratch". These let that be swept rather than assumed.
    ap.add_argument("--min-stop-pct", type=float, default=None,
                    dest="min_stop_pct",
                    help="override config.MIN_STOP_DISTANCE_PCT for this run")
    ap.add_argument("--min-stop-atr", type=float, default=None,
                    dest="min_stop_atr",
                    help="override config.MIN_STOP_DISTANCE_ATR_MULTIPLE for this run")
    ap.add_argument("--stop-model", default="structure", choices=("structure", "floor"),
                    dest="stop_model",
                    help="structure = live model (level +/- ATR buffer, then floor); "
                         "floor = fixed ATR/pct distance (diagnostic only)")
    ap.add_argument("--overlapping", action="store_true",
                    help="count every detection as its own trade (inflates n, "
                         "breaks independence - diagnostic use only)")
    args = ap.parse_args()

    if args.min_stop_pct is not None:
        config.MIN_STOP_DISTANCE_PCT = args.min_stop_pct

    if args.min_stop_atr is not None:
        config.MIN_STOP_DISTANCE_ATR_MULTIPLE = args.min_stop_atr

    interval = args.interval or config.WS_KLINE_INTERVAL
    target_r = args.target if args.target is not None else float(config.TP1_R_MULTIPLE)
    triggers = (
        list(DETECTORS) if args.triggers.strip().upper() == "ALL"
        else [t.strip().upper() for t in args.triggers.split(",") if t.strip()]
    )

    for t in triggers:
        if t not in DETECTORS:
            print(f"unknown trigger {t}; known: {', '.join(DETECTORS)}")
            return 1

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = top_symbols(args.symbols_top)

    if not symbols:
        print("no symbols resolved")
        return 1

    end_ms = int(time.time() * 1000)
    start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000
    )

    print(f"trigger_lab  interval={interval}  target={target_r}R  window={WINDOW}")
    print(f"  stop_model={args.stop_model}  min_stop_pct={config.MIN_STOP_DISTANCE_PCT}"
          f"  min_stop_atr={config.MIN_STOP_DISTANCE_ATR_MULTIPLE}")
    print(f"  symbols={len(symbols)}  days={args.days}  triggers={','.join(triggers)}")
    print(f"  range {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc):%Y-%m-%d} "
          f"-> {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc):%Y-%m-%d}\n")

    all_rows = {t: [] for t in triggers}
    raw_counts = {}
    t0 = time.time()

    for idx, symbol in enumerate(symbols, 1):
        try:
            candles = load_candles(symbol, interval, start_ms, end_ms)
        except Exception as exc:
            print(f"  [{idx}/{len(symbols)}] {symbol}: fetch failed ({exc})")
            continue

        if len(candles) < WINDOW + 10:
            print(f"  [{idx}/{len(symbols)}] {symbol}: only {len(candles)} candles, skipped")
            continue

        counts = []

        for t in triggers:
            rows, raw = replay_symbol(symbol, t, candles, target_r,
                                      args.overlapping, args.stop_model)
            all_rows[t].extend(rows)
            raw_counts[t] = raw_counts.get(t, 0) + raw
            counts.append(f"{t.split('_')[0][:6]}={len(rows)}/{raw}")

        print(f"  [{idx}/{len(symbols)}] {symbol:<14} {len(candles):>5} candles  "
              f"{'  '.join(counts)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*70}\nRESULTS  ({time.time()-t0:.0f}s total)\n{'='*70}")

    for t in triggers:
        summarize(all_rows[t], t, raw_counts.get(t))
        write_csv(all_rows[t], os.path.join(args.out, f"{t}_{interval}_{args.days}d.csv"))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
