"""Generic Binance USDT-M futures REST plumbing.

Ported from v7/v8's exchange.py: only the parts that are pure exchange
mechanics (rate limiting, backoff, exchange-info/symbol-rules, historical
kline fetch) - not the trend-following strategy's order-lifecycle/DCA
reconciliation logic, which belongs to a strategy-specific execution layer
built in a later phase, not here.
"""
from collections import deque
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
from pathlib import Path
import json
import re
import threading
import time

import pandas as pd
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL

import config
from logger import log_error, log_info, log_warning


# config.BINANCE_REQUEST_TIMEOUT_SECONDS - see its own config.py comment
# for the real incident this fixes (an unbounded REST call could hang the
# entire bot indefinitely; requests_params is python-binance's own hook
# for threading a timeout into every request, spot and futures alike -
# confirmed against the installed library's _get_request_kwargs, which
# merges this into every single call's kwargs).
client = Client(
    config.API_KEY, config.SECRET_KEY, ping=False,
    requests_params={"timeout": config.BINANCE_REQUEST_TIMEOUT_SECONDS},
)

if config.BINANCE_TESTNET:
    # python-binance's own `testnet=` constructor flag does NOT redirect
    # USDS-M futures REST calls in this version - it only affects a
    # separate websocket-API client, and FUTURES_URL is left pointed at
    # production regardless (verified empirically: client.FUTURES_URL
    # stays "https://fapi.binance.com/fapi" even with testnet=True).
    # Overriding it directly is the only way that actually works.
    client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

_exchange_info_cache = None
_exchange_info_cache_at = 0.0
_exchange_info_ttl_seconds = 3600

_public_request_weights = deque()
_public_request_lock = threading.Lock()
_last_public_request_at = 0.0

_public_rest_backoff_until = 0.0
_public_rest_lock = threading.Lock()
_public_rest_log_times = {}

_kline_cache = {}
_kline_cache_lock = threading.RLock()
_kline_cache_ttl_seconds = 5

_leverage_cache = {}  # symbol -> confirmed leverage value
_leverage_cache_lock = threading.RLock()

_BAN_UNTIL_RE = re.compile(r"banned until\s+(\d+)", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"(code=-1003|too many requests)", re.IGNORECASE)


def sync_client_time():
    try:
        server_time = client.get_server_time()
        client.timestamp_offset = (
            server_time["serverTime"] - int(time.time() * 1000)
        )
        return True

    except Exception as exc:
        client.timestamp_offset = 0
        log_warning(
            "Binance startup time sync unavailable; "
            "using zero timestamp offset | "
            f"ERROR={exc}"
        )
        return False


# =========================
# RATE LIMITING / BACKOFF (ported convention from v7/v8)
# =========================
def _rate_limit_public_request(weight=1):
    """Two independent guards, both required before a call proceeds:
    (1) cumulative weight within the sliding window stays under the
    per-minute budget (the original mechanism), and (2) at least
    BINANCE_MIN_REQUEST_GAP_SECONDS has passed since the previous call,
    regardless of weight.

    Real bug found live (2026-08-11): an 80-symbol x 2-timeframe startup
    seed loop (ws_client._seed_history) fired ~160 kline calls back-to-
    back with no pacing between them - its SUMMED weight was nominally
    well under the per-minute budget, so guard (1) alone never blocked
    it, but Binance still returned a hard IP ban (-1003). Binance's real
    abuse detection is evidently sensitive to burst RATE, not just weight
    accounted for over a full minute - a weight budget alone can't catch
    "100 calls in the same second" if their total weight is small. Guard
    (2) is a general defense against that shape of bug in ANY caller
    (present or future), not just the one instance found here."""
    max_weight = float(config.BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE)
    min_gap = max(float(config.BINANCE_MIN_REQUEST_GAP_SECONDS), 0.0)

    if max_weight <= 0 and min_gap <= 0:
        return

    window_seconds = max(float(config.BINANCE_PUBLIC_RATE_WINDOW_SECONDS), 1.0)
    weight = max(float(weight or 1), 0.1)

    global _last_public_request_at

    while True:
        now = time.time()

        with _public_request_lock:
            while (
                _public_request_weights
                and now - _public_request_weights[0][0] >= window_seconds
            ):
                _public_request_weights.popleft()

            used_weight = sum(item[1] for item in _public_request_weights)
            since_last = now - _last_public_request_at
            weight_ok = max_weight <= 0 or used_weight + weight <= max_weight
            gap_ok = since_last >= min_gap

            if weight_ok and gap_ok:
                if max_weight > 0:
                    _public_request_weights.append((now, weight))
                _last_public_request_at = now
                return

            if not gap_ok:
                wait_seconds = min_gap - since_last
            else:
                oldest_at = _public_request_weights[0][0]
                wait_seconds = window_seconds - (now - oldest_at) + 0.05

        time.sleep(min(max(wait_seconds, 0.05), 5.0))


def _public_rest_log_allowed(key, cooldown=30):
    now = time.time()

    with _public_rest_lock:
        last_logged_at = _public_rest_log_times.get(key, 0.0)

        if now - last_logged_at < cooldown:
            return False

        _public_rest_log_times[key] = now
        return True


def _extract_public_rest_backoff_seconds(error):
    message = str(error)
    buffer_seconds = 60.0
    match = _BAN_UNTIL_RE.search(message)

    if match:
        try:
            banned_until_ms = int(match.group(1))
            banned_until_seconds = banned_until_ms / 1000
            return max(
                banned_until_seconds - time.time() + buffer_seconds,
                buffer_seconds,
            )
        except (TypeError, ValueError):
            pass

    if _RATE_LIMIT_RE.search(message):
        return 300.0

    return 0.0


def _set_public_rest_backoff(error, context):
    global _public_rest_backoff_until

    pause_seconds = _extract_public_rest_backoff_seconds(error)

    if pause_seconds <= 0:
        return

    until = time.time() + pause_seconds

    with _public_rest_lock:
        _public_rest_backoff_until = max(_public_rest_backoff_until, until)

    if _public_rest_log_allowed("public_rest_backoff_set"):
        log_warning(
            "Binance public REST backoff active | "
            f"CALL={context} | PAUSE_SECONDS={round(pause_seconds, 1)} | "
            f"ERROR={error}"
        )


def get_public_rest_backoff_remaining():
    with _public_rest_lock:
        return max(_public_rest_backoff_until - time.time(), 0.0)


def is_public_rest_backoff_active():
    return get_public_rest_backoff_remaining() > 0


def _raise_if_public_rest_backoff(context):
    remaining = get_public_rest_backoff_remaining()

    if remaining <= 0:
        return

    raise RuntimeError(
        "Binance public REST backoff active | "
        f"CALL={context} | WAIT_SECONDS={round(remaining, 1)}"
    )


def _is_public_rest_backoff_error(error):
    return "Binance public REST backoff active" in str(error)


def _public_rest_call(context, func, *args, weight=1, **kwargs):
    _raise_if_public_rest_backoff(context)
    _rate_limit_public_request(weight)

    try:
        return func(*args, **kwargs)
    except Exception as exc:
        if not _is_public_rest_backoff_error(exc):
            _set_public_rest_backoff(exc, context)
        raise


# =========================
# EXCHANGE INFO / SYMBOL RULES
# =========================
def get_exchange_info(force=False):
    global _exchange_info_cache, _exchange_info_cache_at

    now = time.time()

    if (
        force
        or _exchange_info_cache is None
        or now - _exchange_info_cache_at >= _exchange_info_ttl_seconds
    ):
        _exchange_info_cache = _public_rest_call(
            "futures_exchange_info",
            client.futures_exchange_info,
            weight=config.EXCHANGE_INFO_REQUEST_WEIGHT,
        )
        _exchange_info_cache_at = now

    return _exchange_info_cache


def get_supported_symbols(quote_asset=None):
    quote_asset = (quote_asset or config.QUOTE_ASSET).upper()

    try:
        symbols = set()

        for item in get_exchange_info().get("symbols", []):
            if item.get("status") != "TRADING":
                continue

            if item.get("contractType") != "PERPETUAL":
                continue

            if item.get("quoteAsset") != quote_asset:
                continue

            symbols.add(item["symbol"])

        return symbols

    except Exception as exc:
        log_error(f"supported symbols error: {exc}")
        return set()


def get_symbol_precision(symbol):
    info = get_exchange_info()

    for item in info.get("symbols", []):
        if item.get("symbol") == symbol:
            return item.get("quantityPrecision", 3)

    return 3


def get_price_precision(symbol):
    info = get_exchange_info()

    for item in info.get("symbols", []):
        if item.get("symbol") == symbol:
            return int(item.get("pricePrecision", 4))

    return 4


def get_symbol_price_rules(symbol):
    try:
        for item in get_exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            filters = {
                entry.get("filterType"): entry
                for entry in item.get("filters", [])
            }
            price_filter = filters.get("PRICE_FILTER") or {}
            return {
                "available": True,
                "tick_size": price_filter.get("tickSize", "0"),
                "min_price": price_filter.get("minPrice", "0"),
                "max_price": price_filter.get("maxPrice", "0"),
                "precision": int(item.get("pricePrecision", 4)),
            }

    except Exception as exc:
        log_warning(f"{symbol} price rule lookup warning: {exc}")

    return {
        "available": False,
        "tick_size": "0",
        "min_price": "0",
        "max_price": "0",
        "precision": 8,
    }


# =========================
# KLINES (REST - used to seed history before the websocket feed catches
# up; the live/forming candle for signal detection comes from ws_client)
# =========================
def _kline_cache_key(symbol, interval, limit):
    return (symbol, interval, int(limit))


def _get_cached_kline_df(key):
    with _kline_cache_lock:
        entry = _kline_cache.get(key)

    if entry is None:
        return None

    stored_at, df = entry

    if time.time() - stored_at >= _kline_cache_ttl_seconds:
        return None

    return df.copy(deep=True)


def _store_cached_kline_df(key, df):
    with _kline_cache_lock:
        _kline_cache[key] = (time.time(), df.copy(deep=True))


def _klines_to_df(klines):
    """Shared parsing for the raw list-of-lists futures_klines returns -
    used by both get_klines (live, most-recent-`limit`) and
    get_historical_klines (backtest.py, arbitrary past range) so both
    produce the exact same DataFrame shape."""
    df = pd.DataFrame(klines, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df


def get_klines(symbol, interval, limit=200):
    try:
        cache_key = _kline_cache_key(symbol, interval, limit)
        cached = _get_cached_kline_df(cache_key)

        if cached is not None:
            return cached

        _raise_if_public_rest_backoff("klines")
        _rate_limit_public_request(config.KLINE_REQUEST_WEIGHT)

        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        df = _klines_to_df(klines)
        _store_cached_kline_df(cache_key, df)
        return df.copy(deep=True)

    except Exception as exc:
        if _is_public_rest_backoff_error(exc):
            return None

        _set_public_rest_backoff(exc, f"futures_klines:{symbol}:{interval}")
        log_error(f"{symbol} klines error: {exc}")
        return None


def get_symbol_price_rules(symbol):
    try:
        for item in get_exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            filters = {
                entry.get("filterType"): entry
                for entry in item.get("filters", [])
            }
            price_filter = filters.get("PRICE_FILTER") or {}
            return {
                "available": True,
                "tick_size": price_filter.get("tickSize", "0"),
                "min_price": price_filter.get("minPrice", "0"),
                "max_price": price_filter.get("maxPrice", "0"),
                "precision": int(item.get("pricePrecision", 4)),
            }

    except Exception as exc:
        log_warning(f"{symbol} price rule lookup warning: {exc}")

    return {
        "available": False,
        "tick_size": "0",
        "min_price": "0",
        "max_price": "0",
        "precision": 8,
    }


def normalize_order_price(symbol, price, rounding="nearest"):
    try:
        rules = get_symbol_price_rules(symbol)

        if not rules.get("available", True):
            return 0.0

        value = Decimal(str(float(price)))
        tick = Decimal(str(rules.get("tick_size") or "0"))
        minimum = Decimal(str(rules.get("min_price") or "0"))
        maximum = Decimal(str(rules.get("max_price") or "0"))

        if value <= 0:
            return 0.0

        if tick <= 0:
            precision = max(int(rules.get("precision", 4)), 0)
            return round(float(value), precision)

        rounding_mode = {
            "down": ROUND_DOWN,
            "up": ROUND_UP,
            "nearest": ROUND_HALF_UP,
        }.get(str(rounding or "nearest").lower(), ROUND_HALF_UP)
        steps = (value / tick).to_integral_value(rounding=rounding_mode)
        normalized = steps * tick

        if minimum > 0:
            normalized = max(normalized, minimum)

        if maximum > 0:
            normalized = min(normalized, maximum)

        return float(normalized)

    except (InvalidOperation, TypeError, ValueError) as exc:
        log_warning(f"{symbol} price normalization warning: {exc}")
        return 0.0


def normalize_trigger_price(symbol, side, order_type, price):
    """Round a stop/TP trigger away from the market on the side that
    would otherwise make it trigger immediately - SL rounds against the
    position, TP rounds against the position too but in the opposite
    direction, matching Binance's own trigger-validation direction."""
    side = str(side or "").upper()
    order_type = str(order_type or "").upper()
    take_profit = "TAKE_PROFIT" in order_type

    if side == SIDE_BUY:
        rounding = "down" if take_profit else "up"
    else:
        rounding = "up" if take_profit else "down"

    return normalize_order_price(symbol, price, rounding=rounding)


def _format_price_for_api(symbol, value):
    """Fixed-point decimal string for a price/trigger field going to the
    exchange - NEVER scientific notation. Real bug found live
    (NEIROUSDT, 2026-08-10, 100% reproducible on every attempt): Python's
    default float-to-string conversion silently switches to scientific
    notation below ~1e-4 (e.g. 6.29e-05 for a low-priced symbol like
    this), and Binance's REST API rejects that outright (-1102
    "mandatory parameter ... was not sent, was empty/null, or
    malformed") - an entire tier of low-priced symbols could never
    successfully place a real SL/TP/limit order, no matter how good the
    signal was. Uses the symbol's own reported pricePrecision so the
    formatted string still respects its tick size."""
    precision = max(int(get_symbol_price_rules(symbol).get("precision", 8)), 0)
    return f"{float(value):.{precision}f}"


def get_symbol_quantity_rules(symbol, order_type="MARKET"):
    try:
        for item in get_exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            filters = {
                entry.get("filterType"): entry
                for entry in item.get("filters", [])
            }
            order_type = str(order_type or "MARKET").upper()
            lot_size = filters.get("LOT_SIZE") or {}

            if order_type == "MARKET":
                lot_size = filters.get("MARKET_LOT_SIZE") or lot_size or {}
                max_qty = lot_size.get("maxQty", "0")
            elif order_type == "LIMIT":
                # A plain resting LIMIT entry is neither a MARKET order nor
                # one of the algo/conditional types below (it never
                # executes-as-MARKET-once-triggered) - LOT_SIZE alone is
                # the correct filter, not the tighter-of-both clamp.
                max_qty = lot_size.get("maxQty", "0")
            else:
                # Algo/conditional orders that carry an explicit quantity
                # (TAKE_PROFIT_MARKET's partial TP1 leg) execute as a
                # MARKET order once triggered - confirmed live (XAIUSDT,
                # 2026-08-08): a TP1 quantity that passed LOT_SIZE's max
                # alone still got rejected server-side with -4005 "Quantity
                # greater than max quantity". Clamp to whichever of
                # LOT_SIZE/MARKET_LOT_SIZE is tighter instead of assuming
                # which one Binance enforces for this undocumented case.
                market_lot_size = filters.get("MARKET_LOT_SIZE") or {}
                candidates = [
                    value for value in (
                        _safe_float_local(lot_size.get("maxQty")),
                        _safe_float_local(market_lot_size.get("maxQty")),
                    )
                    if value > 0
                ]
                max_qty = str(min(candidates)) if candidates else lot_size.get("maxQty", "0")

            return {
                "available": True,
                "step_size": lot_size.get("stepSize", "1"),
                "min_qty": lot_size.get("minQty", "0"),
                "max_qty": max_qty,
                "precision": int(item.get("quantityPrecision", 3)),
            }
    except Exception as exc:
        log_warning(f"{symbol} quantity rule lookup warning: {exc}")

    return {
        "available": False,
        "step_size": "0",
        "min_qty": "0",
        "max_qty": "0",
        "precision": 8,
    }


def get_symbol_max_order_quantity(symbol):
    """The most restrictive max order quantity across every order type
    this bot actually places for a symbol over a trade's lifecycle
    (MARKET for entry, TAKE_PROFIT_MARKET for the TP1 partial). Sizing a
    position against only one of these lets the other get rejected later
    with -4005 once TP1 is a fraction of an entry quantity that was only
    ever checked against its own filter - size against the tightest limit
    up front instead."""
    market_max = _safe_float_local(get_symbol_quantity_rules(symbol, order_type="MARKET").get("max_qty"))
    algo_max = _safe_float_local(get_symbol_quantity_rules(symbol, order_type="CONDITIONAL").get("max_qty"))
    candidates = [value for value in (market_max, algo_max) if value > 0]
    return min(candidates) if candidates else 0.0


def normalize_order_quantity(symbol, quantity, order_type="MARKET"):
    try:
        rules = get_symbol_quantity_rules(symbol, order_type=order_type)

        if not rules.get("available", True):
            return 0.0

        value = Decimal(str(abs(float(quantity))))
        step = Decimal(str(rules.get("step_size") or "1"))
        min_qty = Decimal(str(rules.get("min_qty") or "0"))
        max_qty = Decimal(str(rules.get("max_qty") or "0"))

        if step > 0:
            value = (value / step).to_integral_value(rounding=ROUND_DOWN) * step

        if value < min_qty or value <= 0:
            return 0.0

        if max_qty > 0:
            value = min(value, max_qty)

        precision = max(int(rules.get("precision", 3)), 0)
        quantum = Decimal("1").scaleb(-precision)
        value = value.quantize(quantum, rounding=ROUND_DOWN)
        return float(value)

    except (InvalidOperation, TypeError, ValueError) as exc:
        log_warning(f"{symbol} quantity normalization warning: {exc}")
        return 0.0


def get_mark_price(symbol):
    try:
        mark = _public_rest_call(
            f"futures_mark_price:{symbol}",
            client.futures_mark_price,
            symbol=symbol,
            weight=1,
        )
        return float(mark["markPrice"])

    except Exception as exc:
        if _is_public_rest_backoff_error(exc):
            return None

        _set_public_rest_backoff(exc, f"futures_mark_price:{symbol}")
        log_error(f"{symbol} mark price error: {exc}")
        return None


# =========================
# PRIVATE REST (account/orders) - same weight-budget + backoff convention
# as the public layer above, applied to authenticated endpoints.
# =========================
_private_rest_backoff_until = 0.0
_private_rest_lock = threading.Lock()
_private_rest_log_times = {}


def _private_rest_log_allowed(key, cooldown=30):
    now = time.time()

    with _private_rest_lock:
        last_logged_at = _private_rest_log_times.get(key, 0.0)

        if now - last_logged_at < cooldown:
            return False

        _private_rest_log_times[key] = now
        return True


def _extract_private_rest_backoff_seconds(error):
    message = str(error)
    buffer_seconds = 60.0
    match = _BAN_UNTIL_RE.search(message)

    if match:
        try:
            banned_until_ms = int(match.group(1))
            banned_until_seconds = banned_until_ms / 1000
            return max(
                banned_until_seconds - time.time() + buffer_seconds,
                buffer_seconds,
            )
        except (TypeError, ValueError):
            pass

    if _RATE_LIMIT_RE.search(message):
        return 300.0

    return 0.0


def _set_private_rest_backoff(error, context):
    global _private_rest_backoff_until

    pause_seconds = _extract_private_rest_backoff_seconds(error)

    if pause_seconds <= 0:
        return

    until = time.time() + pause_seconds

    with _private_rest_lock:
        _private_rest_backoff_until = max(_private_rest_backoff_until, until)

    if _private_rest_log_allowed("private_rest_backoff_set"):
        log_warning(
            "Binance private REST backoff active | "
            f"CALL={context} | PAUSE_SECONDS={round(pause_seconds, 1)} | "
            f"ERROR={error}"
        )


def get_private_rest_backoff_remaining():
    with _private_rest_lock:
        return max(_private_rest_backoff_until - time.time(), 0.0)


def is_private_rest_backoff_active():
    return get_private_rest_backoff_remaining() > 0


def _raise_if_private_rest_backoff(context):
    remaining = get_private_rest_backoff_remaining()

    if remaining <= 0:
        return

    raise RuntimeError(
        "Binance private REST backoff active | "
        f"CALL={context} | WAIT_SECONDS={round(remaining, 1)}"
    )


def _private_rest_call(context, func, *args, weight=0, **kwargs):
    """weight=0 (the default) skips the pre-emptive weight-budget throttle
    entirely - only the reactive backoff-after-error below still applies.
    Real bug found live (2026-08-11): every call site in this module used
    to omit `weight`, so despite this function's own claim to use "the
    same weight-budget + backoff convention as the public layer", NONE of
    the account/order REST traffic was ever actually throttled - only
    caught after Binance had already banned the IP (-1003). Every call
    site below now passes a real weight for GET/query endpoints (the ones
    that actually cost IP REQUEST_WEIGHT); order mutations (place/cancel)
    are left at the default since Binance doesn't charge IP weight for
    those (verified: they consume the separate order-count limit
    instead) - don't add a new private call site without setting `weight`
    for anything that's a GET/query call, or this regresses silently
    again."""
    _raise_if_private_rest_backoff(context)

    if weight > 0:
        _rate_limit_public_request(weight)

    try:
        return func(*args, **kwargs)
    except Exception as exc:
        _set_private_rest_backoff(exc, context)
        raise


# =========================
# ACCOUNT / POSITIONS
# =========================
_balance_cache = {"time": 0.0, "value": None}
_balance_cache_lock = threading.Lock()
_balance_cache_ttl_seconds = 5


def get_balance(force=False):
    """USDT futures wallet balance, short-TTL cached so risk sizing and a
    position-manager poll loop don't each trigger their own REST call."""
    now = time.time()

    with _balance_cache_lock:
        cached_at, cached_value = _balance_cache["time"], _balance_cache["value"]

    if not force and cached_value is not None and now - cached_at < _balance_cache_ttl_seconds:
        return cached_value

    try:
        balances = _private_rest_call(
            "futures_account_balance",
            client.futures_account_balance,
            weight=5,
        )

        for item in balances:
            if item.get("asset") == "USDT":
                value = float(item["balance"])

                with _balance_cache_lock:
                    _balance_cache["time"] = now
                    _balance_cache["value"] = value

                return value

        return 0.0

    except Exception as exc:
        log_error(f"balance fetch error: {exc}")
        return cached_value if cached_value is not None else 0.0


def _fetch_open_position_detail(symbol):
    """Raises on a REST failure instead of swallowing it - callers that
    need to tell "confirmed flat" apart from "couldn't check right now"
    (e.g. deciding whether to give up on a stop replacement) should call
    this directly rather than get_open_position_detail, which collapses
    both cases to None."""
    positions = _private_rest_call(
        f"futures_position_information:{symbol}",
        client.futures_position_information,
        symbol=symbol,
        weight=5,
    )

    for position in positions or []:
        amount = float(position.get("positionAmt", 0) or 0)

        if amount == 0:
            continue

        return {
            "symbol": symbol,
            "amount": amount,
            "side": "BUY" if amount > 0 else "SELL",
            "quantity": abs(amount),
            "entry_price": float(position.get("entryPrice", 0) or 0),
            "mark_price": float(position.get("markPrice", 0) or 0),
            "unrealized_pnl": float(position.get("unRealizedProfit", 0) or 0),
        }

    return None


def get_open_position_detail(symbol, force=False):
    """Single-symbol position snapshot, one-way (non-hedge) mode only."""
    try:
        return _fetch_open_position_detail(symbol)
    except Exception as exc:
        log_error(f"{symbol} position fetch error: {exc}")
        return None


def has_open_position(symbol, force=True):
    return get_open_position_detail(symbol, force=force) is not None


def _fetch_all_open_positions():
    """Raises on a REST failure instead of swallowing it - same
    "confirmed empty" vs "couldn't check right now" distinction as
    _fetch_open_position_detail. Callers that would take a destructive
    action on absence (PositionManager.reconcile_closed_positions
    treating a symbol's absence here as "closed externally") must call
    this directly, not get_all_open_positions, which collapses both
    cases to []."""
    positions = _private_rest_call(
        "futures_position_information:all",
        client.futures_position_information,
        weight=5,
    )
    result = []

    for position in positions or []:
        amount = float(position.get("positionAmt", 0) or 0)

        if amount == 0:
            continue

        result.append({
            "symbol": position["symbol"],
            "side": "BUY" if amount > 0 else "SELL",
            "quantity": abs(amount),
            "entry_price": float(position.get("entryPrice", 0) or 0),
        })

    return result


def get_all_open_positions():
    """Every non-zero position on the account, account-wide, in one call
    - used for startup reconciliation instead of checking symbol by
    symbol (and catches a position outside the current watchlist too, if
    SCAN_SYMBOLS changed since it was opened)."""
    try:
        return _fetch_all_open_positions()
    except Exception as exc:
        log_error(f"account-wide open positions fetch error: {exc}")
        return []


_OI_UNAVAILABLE_SYMBOL_RE = re.compile(r"code=-4108|code=-1121")
_oi_unavailable_symbols = {}
_oi_unavailable_lock = threading.Lock()


def get_open_interest(symbol):
    """Current total open interest (in contracts) for a symbol. Binance
    has no public OI websocket stream, so this is REST-polled - see
    open_interest.py, which turns a series of these into a % change.

    A symbol that's delisted/settling/pre-trading (-4108) or simply no
    longer a valid symbol (-1121 - can outlive the hourly exchange-info
    cache) will never answer here - skip it for
    OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS after the first such failure
    instead of hitting the same permanent error on every single poll."""
    now = time.time()

    with _oi_unavailable_lock:
        unavailable_until = _oi_unavailable_symbols.get(symbol, 0.0)

    if now < unavailable_until:
        return None

    try:
        data = _public_rest_call(
            f"futures_open_interest:{symbol}",
            client.futures_open_interest,
            symbol=symbol,
            weight=1,
        )

        with _oi_unavailable_lock:
            _oi_unavailable_symbols.pop(symbol, None)

        return float(data["openInterest"])

    except Exception as exc:
        if _is_public_rest_backoff_error(exc):
            return None

        if _OI_UNAVAILABLE_SYMBOL_RE.search(str(exc)):
            cooldown = max(float(config.OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS), 0)

            with _oi_unavailable_lock:
                _oi_unavailable_symbols[symbol] = now + cooldown

            if _public_rest_log_allowed(f"oi_unavailable:{symbol}", cooldown=cooldown):
                log_warning(
                    f"{symbol} open interest unavailable (delisted/settling/invalid) - "
                    f"skipping for {int(cooldown / 60)}m | ERROR={exc}"
                )

            return None

        _set_public_rest_backoff(exc, f"futures_open_interest:{symbol}")
        log_error(f"{symbol} open interest error: {exc}")
        return None


def get_open_interest_history(symbol, period="5m", limit=288):
    """Historical open interest (in contracts, same unit as get_open_
    interest's current-value field) - backs OpenInterestEngine.seed_from_
    history, which fixes OI_DIVERGENCE_TRIGGER_ENABLED never finding data
    (see open_interest.py's own comment): unlike the REST-seeded candle
    history swing points come from, OI's in-memory history used to start
    empty on every restart. period="5m" matches WS_KLINE_INTERVAL's
    granularity (this endpoint only accepts Binance's fixed enum periods,
    not arbitrary intervals); limit=288 is ~24h of coverage at that
    granularity, matching OI_HISTORY_MAX_SAMPLES's original ~24h intent -
    confirmed live against the real endpoint (2026-08-29): count=288,
    span=~24h, and limit=500 also succeeds, so 288 is comfortably under
    the real per-request cap.

    Shares the same unavailable-symbol cooldown as get_open_interest
    above (_oi_unavailable_symbols) - a symbol already known permanently
    invalid there should skip this endpoint too, not just the live one.

    Returns a list of (timestamp_seconds, oi_value) tuples in ascending
    time order, or None on failure - same failure contract as get_open_
    interest. timestamp is converted from Binance's raw milliseconds to
    real Unix seconds, matching OpenInterestEngine.history()'s own
    documented convention."""
    now = time.time()

    with _oi_unavailable_lock:
        unavailable_until = _oi_unavailable_symbols.get(symbol, 0.0)

    if now < unavailable_until:
        return None

    try:
        data = _public_rest_call(
            f"futures_open_interest_hist:{symbol}",
            client.futures_open_interest_hist,
            symbol=symbol,
            period=period,
            limit=limit,
            weight=1,
        )

        with _oi_unavailable_lock:
            _oi_unavailable_symbols.pop(symbol, None)

        return [
            (float(row["timestamp"]) / 1000, float(row["sumOpenInterest"]))
            for row in data
        ]

    except Exception as exc:
        if _is_public_rest_backoff_error(exc):
            return None

        if _OI_UNAVAILABLE_SYMBOL_RE.search(str(exc)):
            cooldown = max(float(config.OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS), 0)

            with _oi_unavailable_lock:
                _oi_unavailable_symbols[symbol] = now + cooldown

            if _public_rest_log_allowed(f"oi_unavailable:{symbol}", cooldown=cooldown):
                log_warning(
                    f"{symbol} open interest history unavailable (delisted/settling/invalid) - "
                    f"skipping for {int(cooldown / 60)}m | ERROR={exc}"
                )

            return None

        _set_public_rest_backoff(exc, f"futures_open_interest_hist:{symbol}")
        log_error(f"{symbol} open interest history error: {exc}")
        return None


def get_open_algo_orders(symbol):
    """Every open algo/conditional order (our SL/TP1/TP2) for a symbol -
    used to rebuild position tracking on startup reconciliation."""
    method = getattr(client, "futures_get_open_algo_orders", None)

    try:
        if method:
            response = _private_rest_call(
                f"futures_get_open_algo_orders:{symbol}",
                method,
                symbol=symbol,
                weight=1,
            )
        else:
            response = _private_rest_call(
                f"futures_get_open_algo_orders:{symbol}",
                client._request_futures_api,
                "get",
                "openAlgoOrders",
                True,
                data={"symbol": symbol},
                weight=1,
            )

        if isinstance(response, dict):
            # v7's proven-working _normalise_algo_orders checks both keys -
            # the wrapper key isn't consistently "data".
            wrapped = response.get("orders") or response.get("data")
            return wrapped if isinstance(wrapped, list) else []

        return response if isinstance(response, list) else []

    except Exception as exc:
        log_error(f"{symbol} open algo orders fetch error: {exc}")
        return []


def setup_leverage(symbol):
    """Cached per symbol for the life of the process - real bug found
    live (MTLUSDT/JOEUSDT, 2026-08-10): this used to make a real
    futures_change_leverage call on EVERY entry attempt, even for a
    symbol whose leverage was already confirmed minutes/hours earlier,
    burning private REST rate-limit budget for no reason - directly
    causing genuinely valid signals to die with "leverage Nx not
    available" purely because this redundant call happened to land while
    Binance's private REST backoff was active. Only a real leverage
    change would ever need this re-checked, and that never happens
    without a restart (config.LEVERAGE is read once at startup), which
    naturally clears the cache anyway."""
    symbol = symbol.upper()

    with _leverage_cache_lock:
        if _leverage_cache.get(symbol) == config.LEVERAGE:
            return True

    try:
        response = _private_rest_call(
            f"futures_change_leverage:{symbol}",
            client.futures_change_leverage,
            symbol=symbol,
            leverage=config.LEVERAGE,
            weight=1,
        )
        actual = int(response["leverage"])

        if actual != config.LEVERAGE:
            log_warning(f"{symbol} leverage mismatch | REQUESTED={config.LEVERAGE} | ACTUAL={actual}")
            return False

        with _leverage_cache_lock:
            _leverage_cache[symbol] = actual

        log_info(f"{symbol} leverage set: {actual}x")
        return True

    except Exception as exc:
        log_error(f"{symbol} leverage error: {exc}")
        return False


# =========================
# ORDERS - entry, and TP/SL via the algo-order (conditional) endpoint,
# same convention v7/v8 use: STOP_MARKET/TAKE_PROFIT_MARKET, MARK_PRICE
# working type, priceProtect on, one-way mode (reduceOnly instead of a
# positionSide close).
# =========================
def place_market_order(symbol, side, quantity):
    quantity = normalize_order_quantity(symbol, quantity, order_type="MARKET")

    if quantity <= 0:
        raise ValueError(f"{symbol} entry quantity normalized to zero")

    return _private_rest_call(
        f"futures_create_order:{symbol}",
        client.futures_create_order,
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=quantity,
    )


def close_position_market(symbol, position_side, quantity):
    """Immediately market-close `quantity` of an open position.
    `position_side` is the position's own side (BUY/SELL, not the closing
    order's side) - the closing order is placed on the opposite side with
    reduceOnly, so it can never accidentally open a new position in the
    other direction."""
    close_side = SIDE_SELL if position_side == SIDE_BUY else SIDE_BUY
    quantity = normalize_order_quantity(symbol, quantity, order_type="MARKET")

    if quantity <= 0:
        raise ValueError(f"{symbol} close quantity normalized to zero")

    return _private_rest_call(
        f"futures_create_order:{symbol}:close",
        client.futures_create_order,
        symbol=symbol,
        side=close_side,
        type="MARKET",
        quantity=quantity,
        reduceOnly="true",
    )


def place_limit_order(symbol, side, quantity, price, client_order_id=None):
    """Plain resting GTC LIMIT entry - config.LIMIT_ENTRY_MODE_ENABLED.
    Unlike place_market_order this does NOT fill synchronously: it can sit
    unfilled indefinitely, so callers must track and expire/cancel it
    themselves (see position_manager.poll_pending_entry) rather than
    assume a position exists once this call returns. Stays on the plain
    order endpoint (python-binance does not auto-route "LIMIT" to the algo
    namespace the way it does STOP_MARKET/TAKE_PROFIT_MARKET).

    `client_order_id` (optional) tags the real exchange order with a
    caller-chosen newClientOrderId - see config.DCA_RESTING_ORDER_ENABLED
    and position_manager.reconcile_pending_entries_on_startup's own use of
    this (a "dcaAdd..." prefix): a resting DCA-add order for an already-
    open, bot-tracked position is otherwise indistinguishable on Binance's
    account-wide open-orders endpoint from a genuinely orphaned pending-
    entry LIMIT order, which that startup-recovery routine would otherwise
    cancel outright. Mirrors place_stop_loss/place_take_profit_full's own
    client_algo_id tagging, applied here via the plain-order endpoint's
    own distinct param name (clientOrderId, not clientAlgoId)."""
    quantity = normalize_order_quantity(symbol, quantity, order_type="LIMIT")

    if quantity <= 0:
        raise ValueError(f"{symbol} entry quantity normalized to zero")

    # Deliberately normalize_order_price, not normalize_trigger_price: a
    # resting limit has no "must not fire immediately" constraint the way
    # a stop/TP trigger does, so the away-from-market rounding that
    # function applies doesn't belong here.
    price = normalize_order_price(symbol, price, rounding="nearest")

    if price <= 0:
        raise ValueError(f"{symbol} entry price is invalid")

    price = _format_price_for_api(symbol, price)

    params = dict(
        symbol=symbol,
        side=side,
        type="LIMIT",
        quantity=quantity,
        price=price,
        timeInForce="GTC",
    )

    if client_order_id:
        params["newClientOrderId"] = client_order_id

    return _private_rest_call(
        f"futures_create_order:{symbol}:limit",
        client.futures_create_order,
        **params,
    )


def place_algo_order(**params):
    """SL/TP conditional orders go through Binance's algo-order namespace
    (same as v7/v8), not the plain order endpoint - `algoStatus="FINISHED"`
    on this namespace is the only reliable signal that a stop/TP actually
    triggered, vs. `CANCELED`/`EXPIRED` because something else closed the
    position first."""
    method = getattr(client, "futures_create_algo_order", None)

    if method:
        return _private_rest_call(
            f"futures_create_algo_order:{params.get('symbol', 'unknown')}",
            method,
            **params,
        )

    return _private_rest_call(
        f"futures_create_algo_order:{params.get('symbol', 'unknown')}",
        client._request_futures_api,
        "post",
        "algoOrder",
        True,
        data=params,
    )


def _accepted_order_id(order):
    order = order or {}
    status = str(order.get("status") or order.get("algoStatus") or "").upper()

    if status in {"REJECTED", "EXPIRED", "CANCELED", "CANCELLED", "FAILED"}:
        return ""

    return order.get("orderId") or order.get("algoId") or ""


def place_stop_loss(symbol, side, trigger_price, client_algo_id=None):
    """Full-position STOP_MARKET, closePosition=true - always closes
    everything still open on that symbol, so it stays valid through a TP1
    partial fill without needing to be resized.

    `client_algo_id` (optional) tags the real exchange order with a
    caller-chosen clientAlgoId - see position_manager._execute_dca's own
    use of this (a "dcaSL..." prefix) for why: a DCA_ACTIVE position's
    real SL+TP shape (one full-position STOP_MARKET + one full-position
    TAKE_PROFIT_MARKET, no partial TP) is otherwise indistinguishable
    from an ordinary post-TP1 BREAKEVEN_ACTIVE position on a restart -
    this tag is what lets _adopt_position tell the two apart from real
    exchange state instead of guessing."""
    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
    trigger_price = normalize_trigger_price(symbol, side, "STOP_MARKET", trigger_price)

    if trigger_price <= 0:
        raise ValueError(f"{symbol} SL trigger price is invalid")

    trigger_price = _format_price_for_api(symbol, trigger_price)

    params = dict(
        symbol=symbol,
        side=close_side,
        type="STOP_MARKET",
        triggerPrice=trigger_price,
        closePosition="true",
        workingType="MARK_PRICE",
        priceProtect="TRUE",
        timeInForce="GTC",
    )

    if client_algo_id:
        params["clientAlgoId"] = client_algo_id

    return place_algo_order(**params)


def _normalize_callback_rate(callback_rate):
    """config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - Binance requires
    0.1-10 for a TRAILING_STOP_MARKET's callbackRate (python-binance's own
    futures_create_algo_order docstring: "min 0.1, max 10"), but this
    project deliberately caps the top end lower (5) - a callback anywhere
    near Binance's own max would give back far more profit than the flat
    breakeven stop it's replacing was ever designed to risk. No existing
    price-formatting helper applies here (normalize_order_price/
    _format_price_for_api are both tick-size/price-precision based) -
    this is a plain percentage, not a price."""
    return round(max(0.1, min(float(callback_rate), 5.0)), 1)


def place_trailing_stop_loss(symbol, side, quantity, activate_price, callback_rate, client_algo_id=None):
    """config.DCA_BREAKEVEN_TRAILING_STOP_ENABLED - a real, native Binance
    TRAILING_STOP_MARKET. Unlike a plain STOP_MARKET this doesn't fire at
    a fixed level - it only starts trailing once price first reaches
    `activate_price`, then trails behind the best price seen by
    `callback_rate`%, entirely server-side (no bot polling, no cancel/
    replace round trip - see position_manager._execute_dca's own comment
    for the real race this closes).

    Real bug found live (2026-08-26, MEUSDT, the feature's first-ever
    live attempt): closePosition=true - this project's original
    assumption, mirroring place_stop_loss's own shape - is rejected
    outright by Binance for TRAILING_STOP_MARKET with -4136 ("Target
    strategy invalid for orderType TRAILING_STOP_MARKET,closePosition
    true"). Confirmed against a real Binance dev-forum report of the
    identical error text and its working resolution: reduceOnly=true
    with an explicit quantity instead - the exact shape place_take_
    profit_partial already uses successfully. `quantity` must be the
    position's real current size (the post-DCA blended quantity at the
    moment this is placed) - unlike closePosition=true, a reduceOnly
    order with a fixed quantity does NOT automatically track a position
    size that changes afterward.

    Still real, unverified: whether a closePosition=true STOP_MARKET
    (the existing wide post-DCA SL) and this reduceOnly TRAILING_STOP_
    MARKET can rest on the same symbol/side simultaneously has never
    been tested in this codebase. Callers must treat this as best-
    effort/non-fatal, exactly like TP1/TP2 placement failures already
    are."""
    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
    quantity = normalize_order_quantity(symbol, quantity, order_type="CONDITIONAL")
    # Reuses the non-TAKE_PROFIT (SL-direction) rounding branch - a
    # genuine price, same rounding math regardless of order type, just
    # never previously exercised with this literal type string.
    activate_price = normalize_trigger_price(symbol, side, "STOP_MARKET", activate_price)

    if quantity <= 0:
        raise ValueError(f"{symbol} trailing stop quantity normalized to zero")

    if activate_price <= 0:
        raise ValueError(f"{symbol} trailing stop activate price is invalid")

    activate_price = _format_price_for_api(symbol, activate_price)
    callback_rate = _normalize_callback_rate(callback_rate)

    params = dict(
        symbol=symbol,
        side=close_side,
        type="TRAILING_STOP_MARKET",
        activatePrice=activate_price,
        callbackRate=callback_rate,
        quantity=quantity,
        reduceOnly="true",
        workingType="MARK_PRICE",
        priceProtect="TRUE",
        timeInForce="GTC",
    )

    if client_algo_id:
        params["clientAlgoId"] = client_algo_id

    return place_algo_order(**params)


def place_take_profit_partial(symbol, side, quantity, trigger_price):
    """Reduce-only TAKE_PROFIT_MARKET for an exact quantity - this is TP1,
    closing part of the position and leaving the rest open."""
    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
    quantity = normalize_order_quantity(symbol, quantity, order_type="CONDITIONAL")
    trigger_price = normalize_trigger_price(symbol, side, "TAKE_PROFIT_MARKET", trigger_price)

    if quantity <= 0:
        raise ValueError(f"{symbol} TP1 quantity normalized to zero")

    if trigger_price <= 0:
        raise ValueError(f"{symbol} TP1 trigger price is invalid")

    trigger_price = _format_price_for_api(symbol, trigger_price)

    return place_algo_order(
        symbol=symbol,
        side=close_side,
        type="TAKE_PROFIT_MARKET",
        triggerPrice=trigger_price,
        quantity=quantity,
        reduceOnly="true",
        workingType="MARK_PRICE",
        priceProtect="TRUE",
        timeInForce="GTC",
    )


def place_take_profit_full(symbol, side, trigger_price, client_algo_id=None):
    """Full-position TAKE_PROFIT_MARKET, closePosition=true - this is TP2
    (or, post-DCA, the single replacement target), closing whatever
    remains. `client_algo_id` - see place_stop_loss's own docstring."""
    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
    trigger_price = normalize_trigger_price(symbol, side, "TAKE_PROFIT_MARKET", trigger_price)

    if trigger_price <= 0:
        raise ValueError(f"{symbol} TP2 trigger price is invalid")

    trigger_price = _format_price_for_api(symbol, trigger_price)

    params = dict(
        symbol=symbol,
        side=close_side,
        type="TAKE_PROFIT_MARKET",
        triggerPrice=trigger_price,
        closePosition="true",
        workingType="MARK_PRICE",
        priceProtect="TRUE",
        timeInForce="GTC",
    )

    if client_algo_id:
        params["clientAlgoId"] = client_algo_id

    return place_algo_order(**params)


def cancel_algo_order(symbol, algo_id):
    method = getattr(client, "futures_cancel_algo_order", None)

    try:
        if method:
            return _private_rest_call(
                f"futures_cancel_algo_order:{symbol}",
                method,
                symbol=symbol,
                algoId=algo_id,
                weight=1,
            )

        return _private_rest_call(
            f"futures_cancel_algo_order:{symbol}",
            client._request_futures_api,
            "delete",
            "algoOrder",
            True,
            data={"symbol": symbol, "algoId": algo_id},
            weight=1,
        )
    except Exception as exc:
        log_warning(f"{symbol} algo order {algo_id} cancel warning: {exc}")
        return None


def cancel_all_open_orders(symbol):
    """Cancel every open order for a symbol - both regular orders and
    algo/conditional orders, which is what our SL/TP1/TP2 actually are
    (Binance treats these as two separate cancel-all endpoints, not one -
    calling only the regular one silently leaves SL/TP orders in place).
    Used whenever a position closes, so a stray or improperly-tracked
    order can never survive on the exchange regardless of whether this
    bot's own order-id bookkeeping is accurate at that moment."""
    try:
        _private_rest_call(
            f"futures_cancel_all_open_orders:{symbol}",
            client.futures_cancel_all_open_orders,
            symbol=symbol,
            weight=1,
        )
    except Exception as exc:
        log_warning(f"{symbol} cancel-all regular orders warning: {exc}")

    try:
        _private_rest_call(
            f"futures_cancel_all_open_orders:{symbol}:conditional",
            client.futures_cancel_all_open_orders,
            symbol=symbol,
            conditional=True,
            weight=1,
        )
    except Exception as exc:
        log_warning(f"{symbol} cancel-all algo orders warning: {exc}")


def get_algo_order_status(symbol, algo_id):
    """`FINISHED` = genuinely triggered. `CANCELED`/`EXPIRED` = removed
    without triggering (e.g. the other leg of the OCO pair filled first).
    `NOT_FOUND` after a cancel call is the expected/successful outcome."""
    method = getattr(client, "futures_get_algo_order", None)

    try:
        if method:
            order = _private_rest_call(
                f"futures_get_algo_order:{symbol}",
                method,
                symbol=symbol,
                algoId=algo_id,
                weight=1,
            )
        else:
            order = _private_rest_call(
                f"futures_get_algo_order:{symbol}",
                client._request_futures_api,
                "get",
                "algoOrder",
                True,
                data={"symbol": symbol, "algoId": algo_id},
                weight=1,
            )

        data = order.get("data") if isinstance(order, dict) and isinstance(order.get("data"), dict) else order
        return str((data or {}).get("algoStatus") or (data or {}).get("status") or "").upper()

    except Exception as exc:
        message = str(exc).lower()

        if "unknown order" in message or "order does not exist" in message or "not found" in message:
            return "NOT_FOUND"

        log_warning(f"{symbol} algo order {algo_id} status lookup warning: {exc}")
        return "UNKNOWN"


def get_order_status(symbol, order_id):
    """Ground truth for a plain (non-algo) order - config.LIMIT_ENTRY_MODE_ENABLED's
    entry order is placed on this endpoint, not the algo namespace, so it
    needs its own status/cancel pair rather than reusing
    get_algo_order_status/cancel_algo_order. `executed_qty`/`avg_price`
    (not just status) matter here specifically because a LIMIT order can
    partially fill - see position_manager.poll_pending_entry."""
    try:
        order = _private_rest_call(
            f"futures_get_order:{symbol}",
            client.futures_get_order,
            symbol=symbol,
            orderId=order_id,
            weight=1,
        )
        return {
            "status": str(order.get("status") or "").upper(),
            "executed_qty": float(order.get("executedQty") or 0),
            "avg_price": float(order.get("avgPrice") or 0),
            "orig_qty": float(order.get("origQty") or 0),
        }

    except Exception as exc:
        message = str(exc).lower()

        if "unknown order" in message or "order does not exist" in message or "not found" in message:
            return {"status": "NOT_FOUND", "executed_qty": 0.0, "avg_price": 0.0, "orig_qty": 0.0}

        log_warning(f"{symbol} order {order_id} status lookup warning: {exc}")
        return {"status": "UNKNOWN", "executed_qty": 0.0, "avg_price": 0.0, "orig_qty": 0.0}


def resolve_market_fill_price(symbol, order, fallback_price):
    """The real average fill price for a just-placed MARKET order - see
    execution.py's own comment on why this project historically trusted
    the planned (pre-order, signal-time) price instead of ever reading
    this back. `avgPrice` is usually already populated in the create-
    order response itself (market orders fill essentially synchronously),
    but can occasionally still read back as 0/missing if the fill hasn't
    fully propagated server-side yet - falls back to one fresh
    get_order_status lookup (the same ground-truth call LIMIT fills
    already use) in that case, and to `fallback_price` only if that also
    comes back empty. Never raises, never blocks/delays the entry beyond
    this one optional lookup - a real fill price is a bookkeeping
    improvement (accurate entry_price/risk_distance for MAE/MFE and
    R-multiple journaling), not something worth failing an entry over."""
    avg_price = float((order or {}).get("avgPrice") or 0)

    if avg_price > 0:
        return avg_price

    order_id = (order or {}).get("orderId")

    if not order_id:
        return fallback_price

    status = get_order_status(symbol, order_id)
    return status["avg_price"] if status["avg_price"] > 0 else fallback_price


def cancel_order(symbol, order_id):
    """Plain (non-algo) cancel - distinct endpoint from cancel_algo_order.
    Never raises; logs and returns None on failure, same convention as
    cancel_algo_order (callers must always re-check get_order_status after
    calling this rather than assume the cancel succeeded - see
    position_manager.poll_pending_entry's post-cancel race re-check)."""
    try:
        return _private_rest_call(
            f"futures_cancel_order:{symbol}",
            client.futures_cancel_order,
            symbol=symbol,
            orderId=order_id,
            weight=1,
        )
    except Exception as exc:
        log_warning(f"{symbol} order {order_id} cancel warning: {exc}")
        return None


def get_all_open_orders():
    """Every open plain (non-algo) order, account-wide, one call - the
    non-algo analog of get_all_open_positions(). Startup-only use (see
    PositionManager.reconcile_pending_entries_on_startup) - never called
    per-poll-tick, so its higher weight doesn't recur."""
    try:
        response = _private_rest_call(
            "futures_get_open_orders",
            client.futures_get_open_orders,
            weight=40,
        )
        return response if isinstance(response, list) else []

    except Exception as exc:
        log_error(f"open orders fetch error: {exc}")
        return []


def get_all_open_algo_orders():
    """Every open algo/conditional order (SL/TP1/TP2/trailing-stop),
    account-wide, one call - the algo-namespace analog of
    get_all_open_orders(). Startup-only use (see PositionManager.
    reconcile_stray_algo_orders_on_startup) - confirmed live (2026-09-01)
    that the underlying endpoint accepts an account-wide query with no
    symbol filter, same shape as get_open_algo_orders(symbol) without
    the filter."""
    method = getattr(client, "futures_get_open_algo_orders", None)

    try:
        if method:
            response = _private_rest_call(
                "futures_get_open_algo_orders", method, weight=40,
            )
        else:
            response = _private_rest_call(
                "futures_get_open_algo_orders", client._request_futures_api,
                "get", "openAlgoOrders", True, data={}, weight=40,
            )

        if isinstance(response, dict):
            # Same wrapper-key ambiguity get_open_algo_orders already
            # handles per-symbol - not consistently "data".
            wrapped = response.get("orders") or response.get("data")
            return wrapped if isinstance(wrapped, list) else []

        return response if isinstance(response, list) else []

    except Exception as exc:
        log_error(f"open algo orders fetch error: {exc}")
        return []


def get_income_history(symbol=None, income_type=None, start_time=None, end_time=None, limit=1000):
    """Ground-truth realized PNL/commission/funding from Binance's own
    account income ledger (/fapi/v1/income) - unlike the journal's
    planned entry/SL/TP prices, this reflects what actually happened
    (real fills, slippage, fees), so it's the right source for "what did
    this actually make or lose", not an estimate derived from our own
    signal records."""
    params = {"limit": min(max(int(limit), 1), 1000)}

    if symbol:
        params["symbol"] = symbol
    if income_type:
        params["incomeType"] = income_type
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    try:
        records = _private_rest_call(
            "futures_income_history",
            client.futures_income_history,
            weight=30,
            **params,
        )
        return records if isinstance(records, list) else []

    except Exception as exc:
        log_error(f"income history fetch error: {exc}")
        return []


def get_24h_quote_volumes():
    """One cheap REST call for 24h quote volume across every symbol -
    used to rank the universe for the watchlist instead of guessing."""
    try:
        tickers = _public_rest_call(
            "futures_ticker",
            client.futures_ticker,
            weight=40,
        )
        return {
            item["symbol"]: _safe_float_local(item.get("quoteVolume"))
            for item in tickers
            if item.get("symbol")
        }
    except Exception as exc:
        log_error(f"24h quote volume fetch error: {exc}")
        return {}


def get_funding_rates():
    """Bulk snapshot (one call, every symbol) of the current funding
    rate - reuses the same premiumIndex endpoint get_mark_price() already
    calls per-symbol, just without a symbol filter, same pattern as
    get_24h_quote_volumes(). Funding rate reflects how crowded long vs
    short positioning is market-wide for a symbol: strongly positive
    means longs are paying heavily to stay long (a crowded trade, more
    prone to a squeeze/reversal); strongly negative is the mirror image
    for shorts."""
    try:
        data = _public_rest_call(
            "futures_mark_price:all",
            client.futures_mark_price,
            weight=10,
        )
        return {
            item["symbol"]: _safe_float_local(item.get("lastFundingRate"))
            for item in data
            if item.get("symbol")
        }
    except Exception as exc:
        log_error(f"funding rate fetch error: {exc}")
        return {}


def get_long_short_ratio(symbol):
    """Global long/short account ratio for one symbol - who's positioned
    which way, distinct from OI's "how much is open" and funding's "cost
    of holding". Unlike premiumIndex/24h-ticker, Binance has no bulk
    "every symbol" version of this endpoint - only one symbol per call -
    so this is deliberately called on-demand (see config.py's
    LONG_SHORT_RATIO_ENABLED comment), never polled across the whole
    watchlist. Returns None (not 0.0 - a real ratio of exactly 0 doesn't
    happen) if the data isn't available, so a missing value can never be
    misread as a real "even" reading."""
    try:
        data = _public_rest_call(
            f"futures_global_longshort_ratio:{symbol}",
            client.futures_global_longshort_ratio,
            symbol=symbol,
            period="1h",
            limit=1,
            weight=1,
        )

        if not data:
            return None

        value = data[-1].get("longShortRatio")
        return float(value) if value is not None else None

    except Exception as exc:
        log_warning(f"{symbol} long/short ratio fetch error: {exc}")
        return None


def get_taker_longshort_ratio(symbol):
    """Binance's own aggregated taker buy/sell VOLUME ratio for one
    symbol - real executed order flow, distinct from get_long_short_
    ratio above (account/position count, not volume) and from this
    bot's own local CVD calculation (aggTrade-derived, a different real
    data source). config.CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED (see
    that flag's own config.py comment for the real evidence). No bulk
    "every symbol" version of this endpoint exists either, so this is
    on-demand only, same constraint as get_long_short_ratio. Returns
    None (not 0.0) when unavailable - a real ratio of exactly 0 doesn't
    happen."""
    try:
        data = _public_rest_call(
            f"futures_taker_longshort_ratio:{symbol}",
            client.futures_taker_longshort_ratio,
            symbol=symbol,
            period="1h",
            limit=1,
            weight=1,
        )

        if not data:
            return None

        value = data[-1].get("buySellRatio")
        return float(value) if value is not None else None

    except Exception as exc:
        log_warning(f"{symbol} taker long/short ratio fetch error: {exc}")
        return None


# =========================
# HISTORICAL DATA (BACKTESTING ONLY) - not called anywhere in the live
# trading loop. Unlike get_klines/get_24h_quote_volumes/get_funding_rates
# above (which only ever look at "right now"), these fetch an arbitrary
# past [start_time_ms, end_time_ms) range, for backtest.py/backtest_feed.py
# to replay through the real signal_engine/risk_manager/position_manager
# code instead of a separate reimplementation. Each result is disk-cached
# under data/backtest_cache/, keyed to its exact symbol/range - since
# historical data never changes, the cache has no TTL (contrast the live
# _kline_cache above, which deliberately expires after 5s).
# =========================
_BACKTEST_CACHE_DIR = Path(__file__).resolve().parent / "data" / "backtest_cache"

KLINE_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


def _backtest_cache_path(kind, symbol, interval, start_time_ms, end_time_ms):
    _BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    interval_part = f"_{interval}" if interval else ""
    return _BACKTEST_CACHE_DIR / f"{kind}_{symbol}{interval_part}_{int(start_time_ms)}_{int(end_time_ms)}.json"


def _read_backtest_cache(path):
    if not path.exists():
        return None

    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_backtest_cache(path, records):
    try:
        with open(path, "w") as fh:
            json.dump(records, fh)
    except OSError as exc:
        log_warning(f"backtest cache write failed ({path.name}): {exc}")


def get_historical_klines(symbol, interval, start_time_ms, end_time_ms):
    """Same DataFrame shape as get_klines() (see _klines_to_df), but for
    an arbitrary past [start_time_ms, end_time_ms) range instead of "the
    most recent `limit` candles" - backtest_feed.py's primary data
    source. Paginates in Binance's own 1500-candle-per-call max, since a
    multi-month range at a fine interval easily exceeds that in one call.
    None on a genuine fetch error; an empty list means the range legitimately
    has no data (e.g. before the symbol existed)."""
    symbol = symbol.upper()
    interval_ms = KLINE_INTERVAL_MS.get(interval)

    if interval_ms is None:
        log_error(f"{symbol} historical klines error: unknown interval {interval}")
        return None

    cache_path = _backtest_cache_path("klines", symbol, interval, start_time_ms, end_time_ms)
    cached = _read_backtest_cache(cache_path)

    if cached is not None:
        return _klines_to_df(cached)

    all_rows = []
    cursor = int(start_time_ms)
    end_time_ms = int(end_time_ms)

    try:
        while cursor < end_time_ms:
            batch = _public_rest_call(
                f"futures_klines_historical:{symbol}:{interval}",
                client.futures_klines,
                weight=config.KLINE_REQUEST_WEIGHT,
                symbol=symbol, interval=interval,
                startTime=cursor, endTime=end_time_ms, limit=1500,
            )

            if not batch:
                break

            all_rows.extend(batch)
            last_open_time = int(batch[-1][0])

            if len(batch) < 1500 or last_open_time + interval_ms > end_time_ms:
                break

            cursor = last_open_time + interval_ms

        _write_backtest_cache(cache_path, all_rows)
        return _klines_to_df(all_rows)

    except Exception as exc:
        log_error(f"{symbol} historical klines error: {exc}")
        return None


_AGG_TRADES_LOOKBACK_ERROR_HINT = "restricted to recent"
_AGG_TRADES_MAX_LOOKBACK_MS = 2 * 86_400_000  # empirically confirmed below


def get_historical_agg_trades(symbol, start_time_ms, end_time_ms):
    """Raw list of aggTrade dicts (Binance's own shape: p=price, q=qty,
    T=timestamp ms, m=isBuyerMaker) covering [start_time_ms, end_time_ms) -
    backtest_feed.py replays these through the real order_flow.CVDEngine.
    record_trade() to reconstruct historical CVD the same way the live
    feed builds it from the real-time aggTrade stream. Binance's REST
    aggTrades endpoint caps each individual startTime/endTime query at a
    1-hour window regardless of `limit`, so this paginates on two axes:
    1-hour outer windows, and up-to-1000-trades inner pages within a
    window for busy symbols/periods.

    Binance ALSO rejects any startTime older than roughly 2 days before
    the real current time on this endpoint at all (APIError -4166,
    "Search window is restricted to recent 2 days only") - confirmed
    empirically, not documented anywhere obvious. There is no historical
    aggTrade archive reachable through this endpoint beyond that; CVD
    reconstruction is only possible for the most recent ~2 days of any
    backtest range, regardless of how far back start_time_ms asks for.
    When a window hits that specific error, everything older is skipped
    in one jump (not retried hour-by-hour) and whatever portion of the
    requested range WAS within the recent-2-day window is still returned
    - a shorter-than-requested result means exactly this, not a bug.
    Never cached when truncated this way (the "recent 2 days" boundary
    moves with real time, so caching a degraded result under a fixed
    range key would lock in less data than a later run could actually
    fetch for the same nominal range)."""
    symbol = symbol.upper()
    cache_path = _backtest_cache_path("aggtrades", symbol, None, start_time_ms, end_time_ms)
    cached = _read_backtest_cache(cache_path)

    if cached is not None:
        return cached

    one_hour_ms = 3_600_000
    all_trades = []
    window_start = int(start_time_ms)
    end_time_ms = int(end_time_ms)
    truncated = False

    while window_start < end_time_ms:
        window_end = min(window_start + one_hour_ms, end_time_ms)
        cursor = window_start

        try:
            while cursor < window_end:
                batch = _public_rest_call(
                    f"futures_aggregate_trades:{symbol}",
                    client.futures_aggregate_trades,
                    weight=20,
                    symbol=symbol, startTime=cursor, endTime=window_end, limit=1000,
                )

                if not batch:
                    break

                all_trades.extend(batch)
                last_ts = int(batch[-1]["T"])

                if len(batch) < 1000:
                    break

                cursor = last_ts + 1

        except Exception as exc:
            if _AGG_TRADES_LOOKBACK_ERROR_HINT in str(exc):
                skip_to = int(time.time() * 1000) - _AGG_TRADES_MAX_LOOKBACK_MS + one_hour_ms

                if skip_to > window_start:
                    truncated = True
                    window_start = skip_to
                    continue

            log_error(f"{symbol} historical aggTrades error: {exc}")
            return all_trades or None

        window_start += one_hour_ms

    if truncated:
        log_warning(
            f"{symbol} historical aggTrades truncated to Binance's ~2-day "
            "lookback limit - CVD unavailable for the older portion of this range"
        )
    else:
        _write_backtest_cache(cache_path, all_trades)

    return all_trades


def get_historical_funding_rates(symbol, start_time_ms, end_time_ms):
    """List of {"fundingTime": ms, "fundingRate": float} dicts covering
    [start_time_ms, end_time_ms) - funding settles every 8h, so even a
    multi-month range is a handful of records, but this still paginates
    (1000/call) for correctness on very long ranges."""
    symbol = symbol.upper()
    cache_path = _backtest_cache_path("funding", symbol, None, start_time_ms, end_time_ms)
    cached = _read_backtest_cache(cache_path)

    if cached is not None:
        return cached

    all_rows = []
    cursor = int(start_time_ms)
    end_time_ms = int(end_time_ms)

    try:
        while cursor < end_time_ms:
            batch = _public_rest_call(
                f"futures_funding_rate:{symbol}",
                client.futures_funding_rate,
                weight=1,
                symbol=symbol, startTime=cursor, endTime=end_time_ms, limit=1000,
            )

            if not batch:
                break

            all_rows.extend(batch)
            last_ts = int(batch[-1]["fundingTime"])

            if len(batch) < 1000:
                break

            cursor = last_ts + 1

        parsed = [
            {"fundingTime": int(row["fundingTime"]), "fundingRate": _safe_float_local(row.get("fundingRate"))}
            for row in all_rows
        ]
        _write_backtest_cache(cache_path, parsed)
        return parsed

    except Exception as exc:
        log_error(f"{symbol} historical funding rate error: {exc}")
        return None


def _safe_float_local(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
