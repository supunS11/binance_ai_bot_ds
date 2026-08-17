"""Backtesting CLI - replays real Binance history through the actual
production code (signal_engine.evaluate, risk_manager.build_trade_plan,
position_manager.poll_shadow, main._evaluate_symbol/_poll_positions), not a
reimplementation of the strategy. See backtest_feed.py for what is and
isn't faithfully reconstructable from historical data, and the plan this
was built from for the full reasoning.

Usage:
    python backtest.py --symbols LDOUSDT,SPXUSDT --start 2026-07-01 --end 2026-08-15

Known, accepted fidelity gaps (see backtest_feed.py's module docstring and
config sections below for why):
  - Depth imbalance, open interest, and liquidations are unavailable
    (informational-only in evaluate(), so this makes the backtest strictly
    more permissive on those specific checks, never silently wrong).
  - CVD (a REQUIRED, gating input - unlike the three above) is only
    reconstructable for the most recent ~2 days of any range: Binance's
    aggTrades endpoint hard-rejects any startTime older than that
    (confirmed empirically, see exchange.get_historical_agg_trades). For
    the older portion of a multi-week/month backtest, every candidate that
    would otherwise reach the CVD check hard-rejects instead
    (ORDER_FLOW_DATA_UNAVAILABLE) - a real, currently-unresolved fidelity
    ceiling on how far back a *full-strategy* backtest can go, not just an
    informational-field gap like the three above. A backtest restricted to
    the last ~2 days gets full fidelity; anything older is CVD-blind.
  - Replays one evaluation per CLOSED candle, not one every
    SIGNAL_EVAL_INTERVAL_SECONDS against a live forming candle -
    SIGNAL_CONFIRM_TICKS is therefore approximated as "N consecutive
    closed candles" rather than "N consecutive ~3s ticks."
  - Each symbol is backtested independently against its own fixed starting
    balance (config.SHADOW_ACCOUNT_BALANCE_USDT) - MAX_TOTAL_POSITIONS and
    shared-balance compounding across symbols are not simulated.

SAFETY: this script forces config.EXECUTION_MODE to "SHADOW" and
config.LONG_SHORT_RATIO_ENABLED to False for its entire run, regardless of
what the live .env says - replaying historical bars through main.py's real
_evaluate_symbol must never be able to place a real order or make a live
"current" REST call while evaluating a past bar. Do not remove this.
"""
import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import config
import exchange
import journal_analysis
import main
import signal_journal
from backtest_feed import BacktestFeed
from position_manager import PositionManager


def _parse_date(value):
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_candle(row):
    return {
        "open_time": int(row["time"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "closed": True,
    }


def _split_priming(df, start_ms):
    """(priming, replay) - rows strictly before start_ms get bulk-seeded
    (matches live's own REST-seed-at-startup shape); rows at/after start_ms
    are what actually gets evaluated, pushed one at a time."""
    if df is None or df.empty:
        return df, df

    before = df[df["time"] < start_ms]
    from_start = df[df["time"] >= start_ms]
    return before, from_start


def _agg_trades_to_tuples(raw_trades):
    """Binance's own aggTrade shape (p/q/T/m) -> the (timestamp_seconds,
    price, quantity, is_buyer_maker) tuples order_flow.CVDEngine.
    record_trade()/BacktestFeed.replay_trades() expect. CVDEngine's
    windows are measured in seconds (see order_flow.py), not the
    milliseconds Binance returns "T" in."""
    return [
        (int(t["T"]) / 1000.0, float(t["p"]), float(t["q"]), bool(t["m"]))
        for t in raw_trades
    ]


def _run_symbol_backtest(symbol, start_ms, end_ms, interval, htf_interval, balance):
    interval_ms = exchange.KLINE_INTERVAL_MS[interval]
    htf_interval_ms = exchange.KLINE_INTERVAL_MS[htf_interval]

    # Lookback buffers mirror exactly what live already relies on being
    # warm before it starts evaluating - WS_KLINE_HISTORY_LIMIT/
    # HTF_KLINE_HISTORY_LIMIT candles of REST-seeded history, same as
    # ws_client._seed_history does at real startup.
    ltf_priming_start = start_ms - config.WS_KLINE_HISTORY_LIMIT * interval_ms
    htf_priming_start = start_ms - config.HTF_KLINE_HISTORY_LIMIT * htf_interval_ms
    # CVD has no historical-seed equivalent live either (it only ever
    # builds from the live aggTrade stream going forward) - a short buffer
    # here just avoids every backtest's first few minutes looking
    # artificially colder than a random moment in a long-running live
    # process would be, not an attempt to fully "warm up" 60/300/900s
    # windows from nothing.
    cvd_priming_start = start_ms - config.ORDER_FLOW_MAX_WINDOW_SECONDS * 1000

    ltf_df = exchange.get_historical_klines(symbol, interval, ltf_priming_start, end_ms)
    htf_df = exchange.get_historical_klines(symbol, htf_interval, htf_priming_start, end_ms)
    raw_trades = exchange.get_historical_agg_trades(symbol, cvd_priming_start, end_ms)
    funding_rows = exchange.get_historical_funding_rates(symbol, ltf_priming_start, end_ms)

    if ltf_df is None or ltf_df.empty:
        print(f"{symbol}: no candle data for this range, skipping")
        return None

    if htf_df is None:
        htf_df = ltf_df.iloc[0:0]

    trades = _agg_trades_to_tuples(raw_trades or [])
    funding_rows = funding_rows or []

    feed = BacktestFeed()
    ltf_priming, ltf_replay = _split_priming(ltf_df, start_ms)
    htf_priming, htf_replay = _split_priming(htf_df, start_ms)

    feed.seed_ltf(symbol, ltf_priming)
    feed.seed_htf(symbol, htf_priming)

    trade_pointer = 0
    volume_window = deque()  # (close_time_seconds, quote_volume)
    funding_pointer = -1

    def _advance_due_trades(up_to_seconds):
        nonlocal trade_pointer
        due = []

        while trade_pointer < len(trades) and trades[trade_pointer][0] <= up_to_seconds:
            due.append(trades[trade_pointer])
            trade_pointer += 1

        if due:
            feed.replay_trades(symbol, due)

    def _advance_funding(up_to_ms):
        nonlocal funding_pointer

        while (
            funding_pointer + 1 < len(funding_rows)
            and funding_rows[funding_pointer + 1]["fundingTime"] <= up_to_ms
        ):
            funding_pointer += 1

        if funding_pointer >= 0:
            feed.set_funding_rate(symbol, funding_rows[funding_pointer]["fundingRate"])

    # Prime trade/funding pointers through the priming window too, so the
    # very first replay bar isn't artificially starved relative to bars
    # that follow it.
    if not ltf_priming.empty:
        priming_close_s = int(ltf_priming.iloc[-1]["close_time"]) / 1000.0
        _advance_due_trades(priming_close_s)
        _advance_funding(int(ltf_priming.iloc[-1]["close_time"]))

        for _, row in ltf_priming.iterrows():
            volume_window.append((int(row["close_time"]) / 1000.0, float(row["qav"])))

    htf_replay_rows = list(htf_replay.iterrows()) if not htf_replay.empty else []
    htf_pointer = 0

    positions = PositionManager()
    reject_counts = Counter()
    reject_symbols = {}
    stability = main.SignalStabilityTracker()
    resolved_count = 0

    for _, row in ltf_replay.iterrows():
        close_time_ms = int(row["close_time"])
        close_time_s = close_time_ms / 1000.0

        while (
            htf_pointer < len(htf_replay_rows)
            and int(htf_replay_rows[htf_pointer][1]["close_time"]) <= close_time_ms
        ):
            feed.push_htf_candle(symbol, _build_candle(htf_replay_rows[htf_pointer][1]))
            htf_pointer += 1

        feed.push_ltf_candle(symbol, _build_candle(row))
        _advance_due_trades(close_time_s)
        _advance_funding(close_time_ms)

        volume_window.append((close_time_s, float(row["qav"])))

        while volume_window and close_time_s - volume_window[0][0] > 86400:
            volume_window.popleft()

        feed.set_quote_volume(symbol, sum(v for _, v in volume_window))

        with patch("time.time", return_value=close_time_s):
            outcome_before = positions.has_open_position(symbol)
            main._evaluate_symbol(
                feed, symbol, positions, balance, reject_counts, reject_symbols, stability
            )
            main._poll_positions(feed, positions)
            positions.resolve_break_confirmations(feed.candles)

            if outcome_before and not positions.has_open_position(symbol):
                resolved_count += 1

    print(f"{symbol}: {len(ltf_replay)} candles replayed, {resolved_count} trade(s) resolved")
    return reject_counts


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list, e.g. LDOUSDT,SPXUSDT")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--interval", default=config.WS_KLINE_INTERVAL)
    parser.add_argument("--htf-interval", default=config.HTF_KLINE_INTERVAL)
    parser.add_argument("--balance", type=float, default=config.SHADOW_ACCOUNT_BALANCE_USDT)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start_ms = _parse_date(args.start)
    end_ms = _parse_date(args.end)

    if args.interval not in exchange.KLINE_INTERVAL_MS or args.htf_interval not in exchange.KLINE_INTERVAL_MS:
        print(f"Unsupported interval - choose from {sorted(exchange.KLINE_INTERVAL_MS)}", file=sys.stderr)
        return 1

    journal_path = (
        Path(__file__).resolve().parent / "data" / f"backtest_signal_journal_{int(time.time())}.csv"
    )

    print(
        f"Backtesting {symbols} from {args.start} to {args.end} "
        f"({args.interval}/{args.htf_interval}) | journal -> {journal_path.name}"
    )

    total_rejects = Counter()

    # See the module docstring's SAFETY note - both patches are load-
    # bearing for the entire run, not just the replay loop.
    with patch.object(config, "EXECUTION_MODE", "SHADOW"), \
         patch.object(config, "LONG_SHORT_RATIO_ENABLED", False), \
         patch.object(signal_journal, "JOURNAL_PATH", journal_path):
        for symbol in symbols:
            reject_counts = _run_symbol_backtest(
                symbol, start_ms, end_ms, args.interval, args.htf_interval, args.balance
            )

            if reject_counts:
                total_rejects.update(reject_counts)

    if total_rejects:
        print("\nRejected candidates by reason (why no signal fired where it didn't):")

        for reason, count in total_rejects.most_common(15):
            print(f"  {reason}={count}")

    print("\n" + journal_analysis.summarize(journal_path=journal_path))
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
