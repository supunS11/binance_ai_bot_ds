"""Durable, append-only CSV journal of every real forced-liquidation event
this bot sees, across all three venues (Binance/Bybit/OKX) - the raw data
liquidation_heatmap.py clusters into real historical liquidity levels.

liquidation_tracker.LiquidationEngine stays a pure in-memory, short-lived
state tracker (its own ring buffer exists only for the real-time
LIQUIDATION_SWEEP_CONFIRMED confirmation signal, bounded to
LIQUIDATION_MAX_EVENTS_PER_SYMBOL/LIQUIDATION_WINDOW_SECONDS - far too
small and short to ever build a real historical density map from). This
module is the separate, durable record - same "always write real
evidence, never trust memory" convention signal_journal.py already
established, and the same separation PositionManager/signal_journal.py
already keep (state-tracking classes never journal themselves).
"""
import csv
from pathlib import Path
import time

import config
from logger import log_warning


EVENTS_PATH = Path(__file__).resolve().parent / "data" / "liquidation_events.csv"

EVENT_FIELDNAMES = ["timestamp", "exchange", "symbol", "side", "price", "notional"]


def _existing_header(path):
    try:
        with open(path, newline="") as handle:
            return next(csv.reader(handle), None)
    except (OSError, StopIteration):
        return None


def _ensure_event_header():
    """Same backup-on-mismatch discipline as signal_journal._ensure_header -
    a stale header over new-shaped rows silently mis-keys every
    csv.DictReader that reads this file back."""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not EVENTS_PATH.exists():
        with open(EVENTS_PATH, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES).writeheader()
        return

    existing = _existing_header(EVENTS_PATH)

    if existing is not None and existing != EVENT_FIELDNAMES:
        backup_path = EVENTS_PATH.with_name(f"liquidation_events.bak_{int(time.time())}.csv")
        EVENTS_PATH.rename(backup_path)
        log_warning(
            f"liquidation_events.csv header didn't match the current schema - "
            f"backed up to {backup_path.name} and started a fresh file"
        )

        with open(EVENTS_PATH, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES).writeheader()


def append_event(exchange, symbol, side, notional, timestamp, price):
    """config.LIQUIDATION_EVENT_JOURNAL_ENABLED - one row per real forced-
    liquidation event, called from ws_client.py at the actual point each
    event arrives (the raw stream is decoupled from main.py's eval loop,
    which never sees it). Never raises into the websocket read loop - a
    journalling failure must not cost the connection, same convention
    signal_journal.py's own append functions already follow."""
    if not config.LIQUIDATION_EVENT_JOURNAL_ENABLED:
        return

    row = {
        "timestamp": timestamp,
        "exchange": str(exchange or "").upper(),
        "symbol": str(symbol or "").upper(),
        "side": str(side or "").upper(),
        "price": price if price is not None else "",
        "notional": notional,
    }

    try:
        _ensure_event_header()

        with open(EVENTS_PATH, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES).writerow(row)
    except OSError as exc:
        log_warning(f"could not append to liquidation_events.csv (continuing): {exc}")


def load_events(symbol=None, since=None):
    """Reads liquidation_events.csv back - the durable source liquidation_
    heatmap.recompute_all() reads from. Never raises; a missing or
    unreadable file returns []. Rows with no real price (price="" - the
    event was accepted by LiquidationEngine but arrived with no usable
    price) are skipped, since they carry nothing a heatmap can cluster on.

    Returns a list of dicts: {"timestamp", "exchange", "symbol", "side",
    "price", "notional"} with timestamp/price/notional as floats."""
    try:
        with open(EVENTS_PATH, newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []

    events = []

    for row in rows:
        price = row.get("price")

        if price in (None, ""):
            continue

        try:
            timestamp = float(row["timestamp"])
            price = float(price)
            notional = float(row["notional"])
        except (KeyError, TypeError, ValueError):
            continue

        if since is not None and timestamp < since:
            continue

        row_symbol = (row.get("symbol") or "").upper()

        if symbol is not None and row_symbol != symbol.upper():
            continue

        events.append({
            "timestamp": timestamp,
            "exchange": row.get("exchange") or "",
            "symbol": row_symbol,
            "side": (row.get("side") or "").upper(),
            "price": price,
            "notional": notional,
        })

    return events
