"""Per-gate contribution measurement - which gates carry the edge?

WHY THIS EXISTS. The live bot is profitable (+79 USDT over the current
config era, profit factor 1.215, commission only 7.7% of gross profit),
but a detector-level replay of the same triggers WITHOUT gates measured
roughly break-even gross. The difference is the gate stack. That inverts
the 2026-09-21 audit's premise: the gates are not suppressing good
triggers, they appear to be where the edge lives. So the useful question
is no longer "which gates can be removed" but "which gates are carrying
the edge, and must not be touched".

METHOD. data/signal_rejects.csv records candidates each gate turned away,
with enough to replay them: entry_price and atr reconstruct the stop via
the same floor risk_manager applies, and candle_open_time anchors the
forward path. A gate whose BLOCKED population would have lost money was
protective; one whose blocked population would have won is costing money.

THREE CORRECTIONS the earlier gate audit did not make:

  1. DRIFT. Scored per side then averaged. Over a rising window every
     population looks long-biased; the prior audit's `size_atr` finding
     did not survive this correction and neither may some gate readings.
  2. FEES, correctly. Entry is a LIMIT order (LIMIT_ENTRY_MODE_ENABLED),
     so the bot pays maker on entry and taker on exit - ~0.07% round trip,
     not the 0.10% taker-both-sides a naive model assumes. Expressed in R
     that is (round_trip) / (risk/entry), which varies ~5x with stop width.
  3. SINGLE-TRIGGER ROWS ONLY. signal_trigger is comma-joined; splitting
     it flipped a prior conclusion from +0.665R to +0.113R.

Read-only. Imports nothing that can trade.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import config
import exchange

MAKER = 0.0002
TAKER = 0.0005
ROUND_TRIP = MAKER + TAKER          # limit entry, market/stop exit

FORWARD_CAP = 48
ERA_START_MS = int(time.mktime(time.strptime("2026-09-05", "%Y-%m-%d")) * 1000)
REJECTS = "/root/binance_ai_bot_ds/data/signal_rejects.csv"


def fnum(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def load_rejects(path, since_ms):
    with open(path, newline="", errors="backslashreplace") as fh:
        rows = list(csv.DictReader(fh))

    out = []
    for r in rows:
        trig = r.get("signal_trigger") or ""
        if "," in trig:                      # multi-trigger candle
            continue
        ts = fnum(r.get("candle_open_time"))
        entry = fnum(r.get("entry_price"))
        atr = fnum(r.get("atr"))
        side = (r.get("side") or "").strip().upper()
        reason = (r.get("reject_reason") or "").strip()
        if not (ts and entry and atr is not None and side in ("BUY", "SELL") and reason):
            continue
        if ts < since_ms:
            continue
        out.append({
            "symbol": (r.get("symbol") or "").strip().upper(),
            "gate": reason.split()[0],
            "trigger": trig or "<none>",
            "ts": int(ts), "entry": entry, "atr": atr, "side": side,
        })
    return out


def min_stop_distance(entry, atr):
    """risk_manager._apply_min_stop_distance's floor, verbatim."""
    return max(entry * max(float(config.MIN_STOP_DISTANCE_PCT), 0) / 100,
               float(atr or 0) * max(float(config.MIN_STOP_DISTANCE_ATR_MULTIPLE), 0))


def forward_walk(candles, i, entry, side, risk, target_r, cap=FORWARD_CAP):
    """First touch wins; a bar touching BOTH stop and target scores a LOSS.
    Identical convention to trigger_lab.py - never relaxed."""
    if risk <= 0:
        return None
    sl = entry - risk if side == "BUY" else entry + risk
    tp = entry + target_r * risk if side == "BUY" else entry - target_r * risk
    mfe = 0.0

    for c in candles[i + 1: i + 1 + cap]:
        if side == "BUY":
            mfe = max(mfe, (c["high"] - entry) / risk)
            hit_sl, hit_tp = c["low"] <= sl, c["high"] >= tp
        else:
            mfe = max(mfe, (entry - c["low"]) / risk)
            hit_sl, hit_tp = c["high"] >= sl, c["low"] <= tp
        if hit_sl:
            return {"r": -1.0, "win": 0, "mfe": mfe}
        if hit_tp:
            return {"r": float(target_r), "win": 1, "mfe": mfe}

    if not candles[i + 1: i + 1 + cap]:
        return None
    last = candles[i + 1: i + 1 + cap][-1]["close"]
    r = (last - entry) / risk if side == "BUY" else (entry - last) / risk
    return {"r": r, "win": 0, "mfe": mfe}


def summarise(rows, label, min_n=150):
    """Drift-adjusted (per side, averaged) and net of each row's own fee."""
    if len(rows) < min_n:
        return None
    per_side = {}
    for side in ("BUY", "SELL"):
        sub = [r for r in rows if r["side"] == side]
        if len(sub) < 40:
            return None
        per_side[side] = (
            sum(r["r"] - r["fee_r"] for r in sub) / len(sub),
            len(sub),
            sum(r["win"] for r in sub) / len(sub) * 100,
        )
    drift_adj = (per_side["BUY"][0] + per_side["SELL"][0]) / 2
    gross = sum(r["r"] for r in rows) / len(rows)
    return {
        "label": label, "n": len(rows), "gross": gross, "net_adj": drift_adj,
        "buy": per_side["BUY"], "sell": per_side["SELL"],
        "win": sum(r["win"] for r in rows) / len(rows) * 100,
        "fee": sorted(r["fee_r"] for r in rows)[len(rows) // 2],
    }


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=None,
                    help="default: live TP1_R_MULTIPLE")
    ap.add_argument("--max-symbols", type=int, default=0)
    args = ap.parse_args()

    target_r = args.target if args.target is not None else float(config.TP1_R_MULTIPLE)
    rows = load_rejects(REJECTS, ERA_START_MS)
    print(f"replayable reject rows (post 2026-09-05, single-trigger): {len(rows)}")
    print(f"target {target_r}R   round trip {ROUND_TRIP*100:.3f}% "
          f"(maker entry + taker exit)\n")

    by_symbol = defaultdict(list)
    for r in rows:
        by_symbol[r["symbol"]].append(r)

    symbols = sorted(by_symbol, key=lambda s: -len(by_symbol[s]))
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    end_ms = int(time.time() * 1000)
    start_ms = ERA_START_MS - 5 * 86_400_000       # priming margin
    done = []
    t0 = time.time()

    for n, symbol in enumerate(symbols, 1):
        try:
            df = exchange.get_historical_klines(symbol, config.WS_KLINE_INTERVAL,
                                                start_ms, end_ms)
        except Exception:
            df = None
        if df is None or getattr(df, "empty", True):
            continue

        candles = [{"time": int(r["time"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"])}
                   for _, r in df.iterrows()]
        index = {c["time"]: i for i, c in enumerate(candles)}

        for r in by_symbol[symbol]:
            i = index.get(r["ts"])
            if i is None or i + 1 >= len(candles):
                continue
            risk = min_stop_distance(r["entry"], r["atr"])
            if risk <= 0 or r["entry"] <= 0:
                continue
            res = forward_walk(candles, i, r["entry"], r["side"], risk, target_r)
            if res is None:
                continue
            r.update(res)
            r["fee_r"] = ROUND_TRIP / (risk / r["entry"])
            done.append(r)

        if n % 40 == 0 or n == len(symbols):
            print(f"  [{n}/{len(symbols)}] {len(done)} replayed "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nreplayed {len(done)} of {len(rows)} rows\n")

    by_gate = defaultdict(list)
    for r in done:
        by_gate[r["gate"]].append(r)

    results = [s for s in (summarise(v, k) for k, v in by_gate.items()) if s]
    results.sort(key=lambda s: s["net_adj"])

    print("=" * 96)
    print("PER-GATE CONTRIBUTION - expectancy of the population each gate BLOCKED")
    print("  negative = the gate SAVED money (protective, do not touch)")
    print("  positive = the gate COST money (candidate to relax)")
    print("=" * 96)
    print(f"  {'gate':<30}{'n':>7}{'win%':>7}{'fee':>8}{'gross':>10}"
          f"{'DRIFT-ADJ NET':>15}{'BUY':>9}{'SELL':>9}")
    print("  " + "-" * 93)
    for s in results:
        print(f"  {s['label']:<30}{s['n']:>7}{s['win']:>6.1f}%{s['fee']:>7.3f}R"
              f"{s['gross']:>+9.3f}R{s['net_adj']:>+14.3f}R"
              f"{s['buy'][0]:>+9.3f}{s['sell'][0]:>+9.3f}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    for s in results:
        both_neg = s["buy"][0] < 0 and s["sell"][0] < 0
        both_pos = s["buy"][0] > 0 and s["sell"][0] > 0
        if both_neg and s["net_adj"] < -0.05:
            v = "PROTECTIVE - keep, both sides agree"
        elif both_pos and s["net_adj"] > 0.05:
            v = "COSTING - candidate to relax, both sides agree"
        elif not (both_neg or both_pos):
            v = "sides DISAGREE - drift artifact, no conclusion"
        else:
            v = "marginal - within noise"
        print(f"  {s['label']:<30}{s['net_adj']:>+8.3f}R   {v}")

    print("\n" + "=" * 96)
    print("TOP GATES, split by trigger (n>=150 per cell)")
    print("=" * 96)
    for s in results[:5]:
        gate = s["label"]
        sub = defaultdict(list)
        for r in by_gate[gate]:
            sub[r["trigger"]].append(r)
        cells = [c for c in (summarise(v, t) for t, v in sub.items()) if c]
        if not cells:
            continue
        print(f"\n  {gate}")
        for c in sorted(cells, key=lambda c: c["net_adj"]):
            print(f"    {c['label']:<28}n={c['n']:<6}{c['net_adj']:>+8.3f}R   "
                  f"BUY {c['buy'][0]:+.3f}  SELL {c['sell'][0]:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
