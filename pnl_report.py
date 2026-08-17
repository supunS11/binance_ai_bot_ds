"""Ground-truth realized PNL since a given lookback window, pulled
directly from Binance's own account income ledger (/fapi/v1/income) -
signal_journal.csv/journal_analysis.py tell you win/breakeven/loss counts
against *planned* prices, but not what was actually made or lost once
real fills, slippage, and fees are accounted for. This is the dollar-
denominated counterpart to that win-rate analysis.

Run directly on the machine with real API keys:
    python pnl_report.py [hours_back]

Defaults to 24 hours back if no argument is given.
"""
import sys
import time
from collections import defaultdict

import exchange


def _fetch_all(symbol=None, income_type=None, start_time=None, end_time=None):
    """Pages through /fapi/v1/income - a single call is capped at 1000
    records, so a busy account over a long enough window needs more than
    one request to see everything."""
    results = []
    limit = 1000
    cursor = start_time

    while True:
        batch = exchange.get_income_history(
            symbol=symbol, income_type=income_type, start_time=cursor,
            end_time=end_time, limit=limit,
        )

        if not batch:
            break

        results.extend(batch)

        if len(batch) < limit:
            break

        cursor = int(batch[-1]["time"]) + 1

    return results


def summarize(hours_back=24, now=None):
    now = time.time() if now is None else now
    start_time = int((now - hours_back * 3600) * 1000)
    records = _fetch_all(start_time=start_time)

    if not records:
        return f"No income records in the last {hours_back}h."

    totals = defaultdict(float)
    by_symbol = defaultdict(lambda: defaultdict(float))

    for record in records:
        income_type = record.get("incomeType", "UNKNOWN")

        try:
            amount = float(record.get("income", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0

        symbol = record.get("symbol") or "ACCOUNT"
        totals[income_type] += amount
        by_symbol[symbol][income_type] += amount

    realized_pnl = totals.get("REALIZED_PNL", 0.0)
    commission = totals.get("COMMISSION", 0.0)
    funding = totals.get("FUNDING_FEE", 0.0)
    net = sum(totals.values())

    lines = [f"PNL report - last {hours_back}h ({len(records)} income records)"]
    lines.append(f"  Realized PNL: {realized_pnl:+.4f} USDT")
    lines.append(f"  Commission:   {commission:+.4f} USDT")
    lines.append(f"  Funding:      {funding:+.4f} USDT")

    other = net - realized_pnl - commission - funding

    if abs(other) > 1e-9:
        lines.append(f"  Other:        {other:+.4f} USDT")

    lines.append(f"  Net:          {net:+.4f} USDT")

    lines.append("\nBy symbol (sorted worst to best realized PNL):")

    for symbol, breakdown in sorted(
        by_symbol.items(), key=lambda kv: kv[1].get("REALIZED_PNL", 0.0)
    ):
        symbol_realized = breakdown.get("REALIZED_PNL", 0.0)
        symbol_net = sum(breakdown.values())
        lines.append(f"  {symbol}: realized={symbol_realized:+.4f} net={symbol_net:+.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(summarize(hours))
