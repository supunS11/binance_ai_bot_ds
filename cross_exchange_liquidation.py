"""Cross-exchange (Bybit, OKX) real-time forced-liquidation feeds - the
same corroboration role cross_exchange_oi.py plays for open interest, but
for liquidations. WS push here instead of REST poll: liquidations are
discrete events, not a periodically-sampled value, and both venues expose
this for free over public/unauthenticated WebSocket - no vendor needed.

liquidation_tracker.LiquidationEngine is reused verbatim for both venues
(it already only ever sees normalized (symbol, side, notional, timestamp)
tuples, never a raw venue payload) - this module owns only the venue-
specific transport/parsing plumbing: building the right subscribe frame,
and turning each venue's raw message shape into the tuple LiquidationEngine.
record_liquidation expects.

Real facts confirmed live (2026-08-29) before writing this, not guessed:
- Bybit (wss://stream.bybit.com/v5/public/linear, topic `allLiquidation.
  {symbol}`) has no meaningful per-connection topic-count limit - all
  ~400 of this bot's watchlist symbols subscribed fine in one connection.
  The real constraint is per-symbol: roughly 5% of symbols get "handler
  not found" (not a supported linear-perpetual liquidation feed on Bybit
  at all, e.g. BTCDOMUSDT/1000SHIBUSDT/MANTRAUSDT) and Bybit rejects the
  ENTIRE subscribe batch if even one bad symbol is included - not just
  that one topic. See bybit_subscribe_with_retry below.
- OKX (wss://ws.okx.com:8443/ws/v5/public, channel "liquidation-orders",
  instType "SWAP") subscribes to the whole market in one frame, no
  per-symbol chunking or rejection handling needed. Confirmed real payload
  shape: {"arg":{...}, "data":[{"instId":"XPL-USDT-SWAP", "details":
  [{"side":"sell","bkPx":"0.0825","sz":"467","ts":"..."}]}]}.
"""
import re

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

_BYBIT_REJECTED_TOPIC_RE = re.compile(r"topic:allLiquidation\.(\S+)")


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if result == result else default  # filters NaN
    except (TypeError, ValueError):
        return default


def _from_okx_symbol(inst_id):
    """"BTC-USDT-SWAP" -> "BTCUSDT". Inverse of cross_exchange_oi._to_okx_
    symbol - the liquidation-orders channel identifies symbols by instId
    in its payload, not via the subscription (which is instType-wide, no
    per-symbol echo). Returns None for anything not matching the expected
    "<BASE>-USDT-SWAP" shape rather than guessing."""
    if not isinstance(inst_id, str):
        return None

    if not inst_id.endswith("-USDT-SWAP"):
        return None

    base = inst_id[: -len("-USDT-SWAP")]

    if not base:
        return None

    return f"{base}USDT"


def bybit_liquidation_stream_names(symbols):
    return [f"allLiquidation.{symbol.upper()}" for symbol in symbols]


def bybit_subscribe_frame(symbols):
    return {"op": "subscribe", "args": bybit_liquidation_stream_names(symbols)}


def bybit_parse_rejected_symbol(reply):
    """reply: a parsed (json.loads'd) Bybit subscribe response. Returns the
    single symbol Bybit rejected the whole batch for (see module docstring
    - one bad topic fails the entire subscribe, not just that topic), or
    None if `reply` isn't a rejection or doesn't match the expected
    "topic:allLiquidation.<SYMBOL>" shape in ret_msg."""
    if not isinstance(reply, dict) or reply.get("success"):
        return None

    match = _BYBIT_REJECTED_TOPIC_RE.search(str(reply.get("ret_msg") or ""))
    return match.group(1) if match else None


def parse_bybit_liquidation(message):
    """message: a parsed (json.loads'd) Bybit `allLiquidation.*` push
    frame. Returns (symbol, side, notional, timestamp_seconds) or None -
    never raises on malformed input. Bybit's S:"Sell"/"Buy" already
    matches LiquidationEngine's SELL/BUY convention directly (Sell = long
    liquidated, Buy = short liquidated), no remapping needed."""
    if not isinstance(message, dict):
        return None

    topic = message.get("topic")

    if not isinstance(topic, str) or not topic.startswith("allLiquidation"):
        return None

    entries = message.get("data")

    if not isinstance(entries, list) or not entries:
        return None

    entry = entries[0]

    if not isinstance(entry, dict):
        return None

    symbol = str(entry.get("s") or "")

    if not symbol:
        return None

    side = str(entry.get("S") or "").upper()
    price = _safe_float(entry.get("p"))
    quantity = _safe_float(entry.get("v"))
    notional = abs(price * quantity)
    timestamp = _safe_float(entry.get("T"), None)
    timestamp = timestamp / 1000 if timestamp is not None else None

    return (symbol, side, notional, timestamp)


def okx_subscribe_frame():
    return {"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}


def parse_okx_liquidation(message):
    """message: a parsed (json.loads'd) OKX liquidation-orders push frame.
    Returns a list of (symbol, side, notional, timestamp_seconds) tuples
    (one frame's `data` can carry multiple instruments, each with multiple
    `details` fills) - never raises, skips anything malformed rather than
    failing the whole frame. OKX's side is lowercase ("sell"/"buy") but the
    same forced-long/forced-short convention as Binance/Bybit - just
    uppercased here to match LiquidationEngine's expectation."""
    if not isinstance(message, dict):
        return []

    entries = message.get("data")

    if not isinstance(entries, list):
        return []

    results = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        symbol = _from_okx_symbol(entry.get("instId"))

        if not symbol:
            continue

        details = entry.get("details")

        if not isinstance(details, list):
            continue

        for detail in details:
            if not isinstance(detail, dict):
                continue

            side = str(detail.get("side") or "").upper()
            price = _safe_float(detail.get("bkPx"))
            quantity = _safe_float(detail.get("sz"))
            notional = abs(price * quantity)
            timestamp = _safe_float(detail.get("ts"), None)
            timestamp = timestamp / 1000 if timestamp is not None else None

            results.append((symbol, side, notional, timestamp))

    return results
