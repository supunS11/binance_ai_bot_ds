"""Cross-exchange open-interest reads (Bybit, OKX) - a corroboration
signal for Binance's own OI_RISING gate (open_interest.py), see
config.CROSS_EXCHANGE_OI_TRACKING_ENABLED for the informational-only
rationale. Both endpoints are public/unauthenticated REST, no vendor
needed.

Deliberately does NOT reuse exchange.py's rate-limit/backoff machinery
(_rate_limit_public_request, _set_public_rest_backoff,
_is_public_rest_backoff_error) - that state is Binance-specific end to
end (a single shared weight budget, Binance error-code regexes, a
literal "Binance public REST backoff active" string match). Reusing it
here would either silently fail to recognize a Bybit/OKX rate-limit
error, or falsely consume Binance's own budget. This module keeps its
own much simpler, host-keyed pacing/backoff state instead - both venues'
public OI endpoints are cheap, ungated-by-weight market data, not
Binance's weighted request budget.
"""
import re
import threading
import time

import requests

import config
from logger import log_error, log_warning

BYBIT_BASE_URL = "https://api.bybit.com"
OKX_BASE_URL = "https://www.okx.com"

_rate_lock = threading.Lock()
_last_request_at = {}  # host -> monotonic timestamp
_backoff_until = {}  # host -> timestamp
_unavailable_symbols = {}  # (host, symbol) -> until-timestamp

_RATE_LIMIT_RE = re.compile(r"(429|rate limit|too many requests)", re.IGNORECASE)


def _rate_limit(host):
    min_gap = max(float(config.CROSS_EXCHANGE_OI_MIN_REQUEST_GAP_SECONDS), 0.0)

    with _rate_lock:
        last_at = _last_request_at.get(host, 0.0)
        now = time.monotonic()
        wait = (last_at + min_gap) - now

        if wait > 0:
            time.sleep(wait)

        _last_request_at[host] = time.monotonic()


def _is_backing_off(host):
    return time.time() < _backoff_until.get(host, 0.0)


def _set_backoff(host, exc):
    if _RATE_LIMIT_RE.search(str(exc)):
        _backoff_until[host] = time.time() + 60
        log_warning(f"{host} open interest rate-limited - backing off 60s | ERROR={exc}")


def _is_symbol_unavailable(host, symbol):
    return time.time() < _unavailable_symbols.get((host, symbol), 0.0)


def _mark_symbol_unavailable(host, symbol):
    cooldown = max(float(config.CROSS_EXCHANGE_OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS), 0)
    _unavailable_symbols[(host, symbol)] = time.time() + cooldown


def _to_okx_symbol(symbol):
    """"BTCUSDT" -> "BTC-USDT-SWAP". Only defined for USDT-quoted symbols
    (this bot's entire tradable universe) - returns None otherwise."""
    if not symbol.endswith("USDT"):
        return None

    base = symbol[:-4]

    if not base:
        return None

    return f"{base}-USDT-SWAP"


def get_open_interest_bybit(symbol):
    """Current open interest (base-asset units) for symbol on Bybit's
    linear (USDT-margined) perpetual market, or None if unavailable for
    any reason (not listed, network error, rate-limited) - best-effort,
    same shape as exchange.get_open_interest."""
    symbol = symbol.upper()
    host = "bybit"

    if _is_symbol_unavailable(host, symbol) or _is_backing_off(host):
        return None

    try:
        _rate_limit(host)
        response = requests.get(
            f"{BYBIT_BASE_URL}/v5/market/open-interest",
            params={
                "category": "linear",
                "symbol": symbol,
                "intervalTime": "5min",
                "limit": 1,
            },
            timeout=max(float(config.CROSS_EXCHANGE_OI_REQUEST_TIMEOUT_SECONDS), 1.0),
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("retCode") != 0:
            _mark_symbol_unavailable(host, symbol)
            return None

        entries = (payload.get("result") or {}).get("list") or []

        if not entries:
            _mark_symbol_unavailable(host, symbol)
            return None

        return float(entries[0]["openInterest"])

    except Exception as exc:
        _set_backoff(host, exc)
        log_error(f"{symbol} Bybit open interest error: {exc}")
        return None


def get_open_interest_okx(symbol):
    """Current open interest (contracts) for symbol on OKX's USDT-margined
    perpetual swap market, or None if unavailable - same best-effort
    shape as get_open_interest_bybit."""
    symbol = symbol.upper()
    host = "okx"
    inst_id = _to_okx_symbol(symbol)

    if inst_id is None:
        return None

    if _is_symbol_unavailable(host, symbol) or _is_backing_off(host):
        return None

    try:
        _rate_limit(host)
        response = requests.get(
            f"{OKX_BASE_URL}/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": inst_id},
            timeout=max(float(config.CROSS_EXCHANGE_OI_REQUEST_TIMEOUT_SECONDS), 1.0),
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != "0":
            _mark_symbol_unavailable(host, symbol)
            return None

        entries = payload.get("data") or []

        if not entries:
            _mark_symbol_unavailable(host, symbol)
            return None

        return float(entries[0]["oi"])

    except Exception as exc:
        _set_backoff(host, exc)
        log_error(f"{symbol} OKX open interest error: {exc}")
        return None


def compute_agreement(binance_change_pct, bybit_change_pct, okx_change_pct):
    """Sign-only comparison (matches signal_engine.py's own
    oi_rising = oi_change_pct > 0 definition) - answers "do other venues
    agree on direction", not a magnitude/weighting question nothing has
    justified yet. None when Binance's own reading, or every
    cross-exchange reading, is unavailable."""
    if binance_change_pct is None:
        return None

    readings = [value for value in (bybit_change_pct, okx_change_pct) if value is not None]

    if not readings:
        return None

    binance_rising = binance_change_pct > 0

    return all((value > 0) == binance_rising for value in readings)


def reset():
    """Test-only: clears all module-level rate/backoff/unavailability
    state between test cases."""
    with _rate_lock:
        _last_request_at.clear()

    _backoff_until.clear()
    _unavailable_symbols.clear()
