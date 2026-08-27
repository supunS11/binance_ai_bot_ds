import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


def env_bool(name, default="False"):
    return os.getenv(name, default).strip().lower() == "true"


def env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def env_str_list(name, default):
    value = os.getenv(name)

    if value in (None, ""):
        return default

    result = [item.strip().upper() for item in value.split(",") if item.strip()]
    return result or default


# =========================
# EXCHANGE / ACCOUNT
# =========================
API_KEY = os.getenv("API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")
# When True, every REST call and websocket stream targets Binance's
# Futures Testnet (testnet.binancefuture.com / stream.binancefuture.com)
# instead of production - put your testnet API_KEY/SECRET_KEY above (they
# are a different key pair than your real account, generated separately
# at https://testnet.binancefuture.com).
BINANCE_TESTNET = env_bool("BINANCE_TESTNET", "False")

# =========================
# RATE LIMITING (ported convention from v7/v8 exchange.py)
# =========================
BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE = env_float(
    "BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE", 1800
)
BINANCE_PUBLIC_RATE_WINDOW_SECONDS = env_float(
    "BINANCE_PUBLIC_RATE_WINDOW_SECONDS", 60
)
KLINE_REQUEST_WEIGHT = env_float("KLINE_REQUEST_WEIGHT", 2)
EXCHANGE_INFO_REQUEST_WEIGHT = env_float("EXCHANGE_INFO_REQUEST_WEIGHT", 1)
# Minimum time between ANY two rate-limited REST calls (public, or private
# calls that pass a weight), regardless of weight. Real bug found live
# (2026-08-11): an unpaced per-symbol loop (80 symbols x 2 timeframes at
# startup) fired ~160 calls back-to-back - their summed weight was well
# under BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE, so the weight budget alone
# never blocked it, but Binance still hard-banned the IP. A cumulative
# weight-per-minute budget can't catch a burst of many calls landing in
# the same second - this floor protects against that shape of bug in any
# caller, not just the one instance found so far. 0 disables it (weight
# budget alone, the original behavior).
BINANCE_MIN_REQUEST_GAP_SECONDS = env_float("BINANCE_MIN_REQUEST_GAP_SECONDS", 0.05)

# =========================
# SYMBOL UNIVERSE / WATCHLIST
# =========================
# Empty means "every tradable USDT-M perpetual" - resolved at startup via
# exchange.get_supported_symbols().
SCAN_SYMBOLS = env_str_list("SCAN_SYMBOLS", [])
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
# How many symbols get promoted from cheap kline-only scanning to the
# heavier depth+aggTrade websocket tier (mirrors v7's
# ORDER_FLOW_SHADOW_MAX_SYMBOLS pattern).
WATCHLIST_SIZE = env_int("WATCHLIST_SIZE", 15)
WATCHLIST_REFRESH_SECONDS = env_int("WATCHLIST_REFRESH_SECONDS", 300)
# How often ws_client refreshes the 24h-quote-volume map (a single bulk
# REST call covering every symbol, not per-symbol) that backs
# MIN_24H_QUOTE_VOLUME_USDT below.
VOLUME_POLL_INTERVAL_SECONDS = env_int("VOLUME_POLL_INTERVAL_SECONDS", 300)
# Signal-time liquidity floor - independent of watchlist selection, so a
# broad/unfiltered watchlist (e.g. SCAN_SYMBOLS pinned to the full 500+
# symbol universe) can still be scanned for structure without illiquid/
# vanity-ticker symbols actually being tradeable. 0 disables it. Real
# motivation (2026-08-11): reintroducing the full original symbol list
# for broader coverage reintroduced exactly the illiquid-symbol noise
# that narrowing the watchlist had removed - this restores that quality
# filter at signal time instead of watchlist time, so both can be tuned
# independently. Starting value is a reasonable floor, not yet calibrated
# against real trade data - revisit once there's evidence for where the
# real quality cutoff sits. A symbol with no volume data yet (poll hasn't
# completed, or the ticker endpoint has nothing for it) is let through
# rather than blocked - never gate on data we don't actually have.
# Lowered from 3,000,000 -> 1,500,000 (2026-08-14, operator request after
# real evidence): this single reason alone accounted for ~49% of every
# rejection tallied in the live bot.log - and a direct symbol-count check
# (exchange.get_24h_quote_volumes() against the real WATCHLIST_SIZE=400
# top-by-volume selection) showed why: only 209 of those 400 watchlisted
# symbols actually cleared the old 3M floor, with another 115 sitting in
# the 1.5M-3M band that this change now admits (76 remain below 1.5M,
# still blocked). Free in infra terms - every watchlisted symbol already
# gets full websocket data collection regardless of this check, so this
# only stops discarding data already being paid for, it doesn't add any
# load. Real open question, not yet resolved: whether this newly-admitted
# 1.5M-3M cohort converts to trades at a similar rate to the 3M+ cohort,
# or worse given the wider spreads/thinner books smaller-cap symbols
# carry - watch journal_analysis.py for that specific band once trades
# accumulate.
MIN_24H_QUOTE_VOLUME_USDT = env_float("MIN_24H_QUOTE_VOLUME_USDT", 1500000)

# =========================
# WEBSOCKET DATA LAYER
# =========================
WS_ENABLED = env_bool("WS_ENABLED", "True")
WS_KLINE_INTERVAL = os.getenv("WS_KLINE_INTERVAL", "5m")
WS_KLINE_HISTORY_LIMIT = env_int("WS_KLINE_HISTORY_LIMIT", 200)
# HTF bias/dealing-range timeframe - separate stream+buffer from the LTF
# structure-trigger timeframe above.
HTF_KLINE_INTERVAL = os.getenv("HTF_KLINE_INTERVAL", "1h")
HTF_KLINE_HISTORY_LIMIT = env_int("HTF_KLINE_HISTORY_LIMIT", 200)
WS_STREAMS_PER_SOCKET = env_int("WS_STREAMS_PER_SOCKET", 100)
WS_DEPTH_STREAMS_PER_SOCKET = env_int("WS_DEPTH_STREAMS_PER_SOCKET", 50)
WS_DEPTH_LEVELS = os.getenv("WS_DEPTH_LEVELS", "20")
WS_DEPTH_SPEED_MS = os.getenv("WS_DEPTH_SPEED_MS", "100ms")
WS_STALE_SECONDS = env_float("WS_STALE_SECONDS", 45)
WS_WATCHDOG_INTERVAL_SECONDS = env_float("WS_WATCHDOG_INTERVAL_SECONDS", 15)
WS_RESTART_COOLDOWN_SECONDS = env_float("WS_RESTART_COOLDOWN_SECONDS", 30)
# How many LTF candle closes of CVD history order_flow.CVDEngine retains
# per symbol (see CVDEngine.finalize_candle/cvd_history) - backs
# CVD_DIVERGENCE_TRIGGER_ENABLED's swing-point comparison below. Kept
# >= WS_KLINE_HISTORY_LIMIT so CVD history always covers the same span as
# ltf_candles - a swing point still visible in candles but already
# evicted from CVD history would silently and permanently disqualify
# divergence detection for it.
CVD_HISTORY_MAXLEN = env_int("CVD_HISTORY_MAXLEN", 200)

# =========================
# ORDER FLOW (CVD)
# =========================
ORDER_FLOW_MAX_WINDOW_SECONDS = env_int("ORDER_FLOW_MAX_WINDOW_SECONDS", 900)
ORDER_FLOW_MIN_NOTIONAL_USDT = env_float("ORDER_FLOW_MIN_NOTIONAL_USDT", 5000)
# Was defined but never wired to anything (the 5th trigger - CVD/order-flow
# divergence - was originally skipped while the other 4 were built). Now
# backs CVD_DIVERGENCE_TRIGGER_ENABLED below: how stale the qualifying
# swing point is allowed to be before the trigger stops firing on it, same
# shape as CHOCH_TRIGGER_MAX_AGE_CANDLES/OB_FVG_RETEST_MAX_AGE_CANDLES.
ORDER_FLOW_DIVERGENCE_LOOKBACK = env_int("ORDER_FLOW_DIVERGENCE_LOOKBACK", 20)

# =========================
# ORDER-BOOK ABSORPTION (informational)
# =========================
# 2026-08-26 signal-engine "next level" audit: real, one-sided aggressive
# volume (order_flow.CVDEngine's existing ratio_1m/notional_1m) that
# didn't move price (orderbook.DepthImbalanceEngine's short mid-price
# history, built alongside this) - the one thing neither existing
# order-flow gate measures. SIGNAL_MIN_CVD_SCORE is a recent-window
# aggressor LEAN, blind to how much price actually moved for that volume;
# SIGNAL_MIN_DEPTH_IMBALANCE is a snapshot of RESTING size, blind to
# whether real flow is even testing it right now (see absorption.py's own
# docstring). Uses data already flowing for both of those - no new
# market-data subscription. Brand new, zero trade history exists on it by
# construction - informational/journaled only, same "evidence before
# gate" convention as ema_aligned/btc_aligned/funding_favorable. Default
# True (unlike CVD_DIVERGENCE_TRIGGER_ENABLED's own "brand new, default
# OFF" convention): this is a read-only computation, never changes
# entry/exit behavior, same treatment as those informational fields' own
# ENABLED defaults.
ABSORPTION_TRACKING_ENABLED = env_bool("ABSORPTION_TRACKING_ENABLED", "True")
# How far back orderbook.DepthImbalanceEngine retains its own short
# mid-price history to answer "how much did price actually move" - must
# comfortably exceed ABSORPTION_WINDOW_SECONDS below so a real reference
# sample already exists once the engine's been running a while.
ABSORPTION_PRICE_HISTORY_SECONDS = env_int("ABSORPTION_PRICE_HISTORY_SECONDS", 90)
# The lookback window absorption.compute() measures price movement over -
# deliberately matches CVDEngine.snapshot()'s own first window (60s,
# "ratio_1m"/"notional_1m") so the aggressor-volume reading and the
# price-movement reading always describe the exact same interval, not two
# different timeframes compared as if they were one.
ABSORPTION_WINDOW_SECONDS = env_int("ABSORPTION_WINDOW_SECONDS", 60)
# How one-sided the 60s aggressor flow (cvd_snapshot's ratio_1m, -1..1)
# has to be before a reading is trusted at all - same purpose as
# SIGNAL_MIN_CVD_SCORE, kept as its own knob since this is a different
# question (a genuinely lopsided window worth explaining away, not merely
# "confirms this trade's direction"). Starting value, not yet calibrated
# against real trade data - zero trade history exists on this field by
# construction.
ABSORPTION_MIN_CVD_RATIO = env_float("ABSORPTION_MIN_CVD_RATIO", 0.5)
# How little the mid-price is allowed to have actually moved (%, absolute)
# over that same window for the lopsided flow above to count as
# "absorbed" rather than merely "resolved a bit slower than usual".
# Starting value, not yet calibrated against real trade data.
ABSORPTION_MAX_PRICE_MOVE_PCT = env_float("ABSORPTION_MAX_PRICE_MOVE_PCT", 0.05)

# =========================
# CROSS-EXCHANGE OPEN INTEREST (informational)
# =========================
# 2026-08-26: OI_RISING (above) is a real, evidence-backed gate, but it
# only ever sees Binance's own OI - if the rest of the market is closing
# out while Binance-only positioning looks fresh, that's invisible today.
# No diagnosed problem behind this (unlike CROSS_EXCHANGE liquidation
# data, which fixes a proven data-sparsity gap) - this is a speculative
# corroboration signal, informational/journaled only, same "evidence
# before gate" convention as absorption_signal/btc_aligned. Confirmed
# buildable without a paid vendor: Bybit's /v5/market/open-interest and
# OKX's /api/v5/public/open-interest are both public, unauthenticated,
# generously rate-limited REST endpoints (see cross_exchange_oi.py).
# Default OFF (unlike ABSORPTION_TRACKING_ENABLED's default True) -
# unlike absorption, this is genuinely NEW outbound network I/O to two
# hosts the bot has never talked to before (new failure modes, new
# latency, new third-party uptime dependency), not a read-only
# computation over data already flowing locally. Ships gated off, same
# rollout convention as DCA_BREAKEVEN_TRAILING_STOP_ENABLED - turned on
# deliberately once ready to add live network load.
CROSS_EXCHANGE_OI_TRACKING_ENABLED = env_bool("CROSS_EXCHANGE_OI_TRACKING_ENABLED", "False")
# Mirrors OI_POLL_INTERVAL_SECONDS - kept as its own knob since Bybit/OKX
# have their own independent rate limits, not tied to Binance's.
CROSS_EXCHANGE_OI_POLL_INTERVAL_SECONDS = env_int("CROSS_EXCHANGE_OI_POLL_INTERVAL_SECONDS", 60)
# Mirrors OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS's own rationale - a
# symbol simply not listed on Bybit/OKX (common for Binance-only small
# caps) shouldn't be retried every poll cycle forever.
CROSS_EXCHANGE_OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS = env_int("CROSS_EXCHANGE_OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS", 3600)
CROSS_EXCHANGE_OI_REQUEST_TIMEOUT_SECONDS = env_float("CROSS_EXCHANGE_OI_REQUEST_TIMEOUT_SECONDS", 5.0)
# Minimum gap between successive requests to the SAME host - simple
# pacing, not a weight budget (both venues' public OI endpoints are cheap
# / ungated by weight, unlike Binance's - see cross_exchange_oi.py's own
# docstring for why exchange.py's Binance-specific rate/backoff globals
# are deliberately NOT reused here).
CROSS_EXCHANGE_OI_MIN_REQUEST_GAP_SECONDS = env_float("CROSS_EXCHANGE_OI_MIN_REQUEST_GAP_SECONDS", 0.2)
# EXPLICIT LIVE TEST (2026-08-26) - zero resolved-trade evidence behind
# this, unlike every other reject gate in signal_engine.py. User's own
# choice, made with that tradeoff explained - see this flag's use in
# signal_engine.py for the exact reject condition. Implicitly a no-op
# unless CROSS_EXCHANGE_OI_TRACKING_ENABLED is also True (cross_exchange_
# oi_agree only ever computes when that's on).
CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED = env_bool("CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED", "False")

# =========================
# VOLUME PROFILE (informational)
# =========================
# 2026-08-26 signal-engine "next level" audit: no diagnosed problem
# behind this either - a speculative structural read (where has the most
# volume actually traded recently) that nothing here has validated yet.
# Built from data already flowing (the aggTrade stream orderbook.py/
# order_flow.py already consume) - no new market-data subscription,
# unlike CROSS_EXCHANGE_OI_TRACKING_ENABLED above. volume_profile.py is
# the first thing in this codebase to retain raw per-trade PRICE (CVDEngine
# reduces straight to notional and discards it) - a new, small per-tick
# cost (append + occasional prune), same cost class CVDEngine already
# pays on every trade. Default True, same reasoning as
# ABSORPTION_TRACKING_ENABLED: read-only computation, never changes
# entry/exit behavior. Purely descriptive fields only (poc/value-area/
# position) - deliberately NOT inventing a directional "aligned" boolean
# the way absorption_aligned has one, since there's no evidence-backed
# hypothesis yet for which position implies which trade direction.
VOLUME_PROFILE_TRACKING_ENABLED = env_bool("VOLUME_PROFILE_TRACKING_ENABLED", "True")
# Rolling window of raw trades volume_profile.py buckets into a histogram
# - 4h, a reasoned starting point (long enough to span more than one LTF
# swing, short enough to describe "recent" positioning rather than a
# stale multi-day profile), not calibrated against real trade data.
VOLUME_PROFILE_LOOKBACK_SECONDS = env_int("VOLUME_PROFILE_LOOKBACK_SECONDS", 14400)
# Hard per-symbol cap on retained raw trade samples regardless of
# lookback - memory safety valve for a very high-frequency symbol (e.g.
# BTC), mirrors OI_HISTORY_MAX_SAMPLES's own purpose.
VOLUME_PROFILE_MAX_SAMPLES = env_int("VOLUME_PROFILE_MAX_SAMPLES", 20000)
# Bucket width as a % of price, not a fixed $ tick - a fixed tick would be
# meaningless across this bot's symbol range (BTC ~$100k vs a sub-cent
# altcoin). Starting value, not yet calibrated against real trade data.
VOLUME_PROFILE_BUCKET_PCT = env_float("VOLUME_PROFILE_BUCKET_PCT", 0.05)
# % of total windowed notional the "value area" must cover, expanding
# outward from the point of control (POC) by descending bucket notional -
# 70% is the standard volume-profile convention (roughly one standard
# deviation), not something calibrated against this bot's own data.
VOLUME_PROFILE_VALUE_AREA_PCT = env_float("VOLUME_PROFILE_VALUE_AREA_PCT", 70.0)
# EXPLICIT LIVE TEST (2026-08-26) - zero resolved-trade evidence, same as
# CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED above. Mean-reversion hypothesis
# (user's choice): price already outside the value area in the trade's
# own direction reads as "already ran, don't chase" - same spirit as the
# evidence-backed MAX_ENTRY_EXTENSION_R reject, but unproven here.
# Implicitly a no-op unless VOLUME_PROFILE_TRACKING_ENABLED is also True.
VP_EXTENSION_REJECT_ENABLED = env_bool("VP_EXTENSION_REJECT_ENABLED", "False")

# =========================
# MARKET STRUCTURE (ICT/SMC)
# =========================
SWING_LEFT = env_int("SWING_LEFT", 2)
SWING_RIGHT = env_int("SWING_RIGHT", 2)
STRUCTURE_LOOKBACK_CANDLES = env_int("STRUCTURE_LOOKBACK_CANDLES", 150)
FVG_LOOKBACK_CANDLES = env_int("FVG_LOOKBACK_CANDLES", 50)
LIQUIDITY_POOL_TOLERANCE_PCT = env_float("LIQUIDITY_POOL_TOLERANCE_PCT", 0.001)
PREMIUM_DISCOUNT_LOOKBACK_CANDLES = env_int(
    "PREMIUM_DISCOUNT_LOOKBACK_CANDLES", 100
)
# Raised from 0.618 -> 0.705 (2026-08-14, operator feedback): BUY signals
# were seen firing in the discount zone while price kept falling anyway,
# and SELL signals in premium while price kept rising - both consistent
# with 0.618 (the shallow end of the classic Fibonacci OTE band) not
# requiring enough of a pullback before entry to reflect a genuinely
# exhausted move. Narrows the qualifying retracement band (0.705-0.79 vs
# the old 0.618-0.79), requiring a deeper pullback before either the
# zone or OTE gate can pass. signal_engine.py now also journals the real
# retracement depth (zone_retracement_pct) for every signal regardless of
# where it landed in the band, so journal_analysis.py can test whether
# shallower qualifying retracements actually lose more before tightening
# further - this value is a reasoned starting point, not yet calibrated
# against real trade data at the new setting.
OTE_RETRACEMENT_MIN = env_float("OTE_RETRACEMENT_MIN", 0.705)
OTE_RETRACEMENT_MAX = env_float("OTE_RETRACEMENT_MAX", 0.79)
ATR_PERIOD = env_int("ATR_PERIOD", 14)
# Kaufman's Efficiency Ratio lookback - net directional movement over the
# window divided by total path length (1.0 = straight-line trend, near 0
# = chop/round-trip noise). Informational only - computed and journaled
# so a break inside a genuinely low-conviction, choppy market can be told
# apart from one inside a real trend, evidence pending on whether it
# actually separates winners from losers.
CHOP_FILTER_LOOKBACK_CANDLES = env_int("CHOP_FILTER_LOOKBACK_CANDLES", 14)
# BTC correlation - most alts move because BTC moves, not from their own
# structure. Informational only: computed/journaled, not gated, same
# evidence-first treatment as every other confluence field here.
BTC_CORRELATION_ENABLED = env_bool("BTC_CORRELATION_ENABLED", "True")
# Same timeframe-mismatch fix as EMA_ALIGNMENT_PERIOD above, same evidence
# (2026-08-24 underwater-duration audit) - at 20 (on 1h ltf_candles) this
# was a 20-HOUR correlation/return window feeding btc_aligned, far slower
# than the actual trade lifecycle (median 2.6h). Safe to recalibrate this
# constant directly rather than add a parallel one: grep-confirmed, its
# only consumers are btc_aligned and its own journaled/informational value
# - no trigger, gate, or sizing logic reads it any other way. 4 (hours)
# matches EMA_ALIGNMENT_PERIOD for the same "median trade duration" reason
# - reasoned, not yet outcome-validated; revisit once enough trades
# resolve under it.
CORRELATION_LOOKBACK_CANDLES = env_int("CORRELATION_LOOKBACK_CANDLES", 4)
CORRELATION_REFERENCE_SYMBOL = os.getenv("CORRELATION_REFERENCE_SYMBOL", "BTCUSDT").upper()
# =========================
# CRASH DETECTOR
# =========================
# Real incident (2026-08-22): a BTC flash-crash (~2.65% in ~4 minutes,
# 05:07-05:11 UTC) caught one open BUY position mid-DCA - it averaged in
# right at the bottom, then its real SL placement was rejected by the
# exchange ("Order would immediately trigger" - price had already blown
# through the level), forcing an emergency market-close at a real loss.
# Every position on the OTHER side of the same move profited from it.
# config.DCA_PRESSURE_CHECK_ENABLED DID fire correctly at that exact
# moment (reduced the DCA size) - real, working infrastructure, just not
# strong enough alone: volume was already 15-20x normal a full 2 minutes
# before that damaging DCA fired, real lead time crash_detector.py exists
# to use. See crash_detector.py's own docstring for the full mechanism.
#
# Two-phase rollout, same convention as DCA_BREAKEVEN_CONFIRMATION_
# ENABLED/..._WITHHOLD_ENABLED above: detection itself ships on
# immediately (tracking + transition logging only, zero trading behavior
# change on its own) - the two behavior-changing switches below ship OFF
# until the detector's been observed firing correctly against real data
# a few more times.
CRASH_DETECTOR_ENABLED = env_bool("CRASH_DETECTOR_ENABLED", "True")
# The actual behavior-changing switches - both require CRASH_DETECTOR_
# ENABLED ALSO on to do anything. Left independently toggleable so the
# informational phase can run for as long as needed without either being
# touched.
CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED = env_bool("CRASH_DETECTOR_BLOCK_ENTRIES_ENABLED", "False")
CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED = env_bool(
    "CRASH_DETECTOR_FORCE_DCA_PRESSURE_ENABLED", "False"
)
CRASH_DETECTOR_REFERENCE_SYMBOL = os.getenv("CRASH_DETECTOR_REFERENCE_SYMBOL", "BTCUSDT").upper()
# Real incident's damaging leg fit inside ~4 minutes - 3 minutes catches
# a move like that mid-way through, not only after it's already over.
CRASH_DETECTOR_WINDOW_SECONDS = env_int("CRASH_DETECTOR_WINDOW_SECONDS", 180)
# Real incident moved ~2.65% in ~4 min; ordinary 15-min BTC swings around
# it (same day's own price history) were all under 0.6%. 1.5% within a
# 3-min window sits meaningfully above routine noise and below the real
# event, with margin either way. Starting value, not fully calibrated -
# only one real incident exists to check it against so far - revisit
# once it's been observed live a few more times.
CRASH_DETECTOR_MOVE_PCT = env_float("CRASH_DETECTOR_MOVE_PCT", 1.5)
# Stays active this long after the LAST trigger (extends, not resets, on
# repeat triggers - see crash_detector.py's _trigger) so a sustained
# crash doesn't flicker crash-mode off on a single brief bounce.
CRASH_DETECTOR_COOLDOWN_SECONDS = env_int("CRASH_DETECTOR_COOLDOWN_SECONDS", 600)
# HTF bias freshness check. Real motivation (2026-08-15, traced directly
# against real Binance history): htf_structure's trend comes from
# structure_state() applied to the HTF (4h) candles using the SAME
# SWING_LEFT/SWING_RIGHT=4 the fast-reacting LTF uses - on a 4h chart
# that means a swing needs 16 REAL hours on each side before it's even
# confirmed, so the "last confirmed swing" AGAINST_HTF_BIAS compares
# against can be stale by many hours relative to what price is actually
# doing right now. Confirmed live: CATIUSDT's htf_trend read BULLISH for
# a full 21-hour stretch (2026-08-09 20:03 UTC - 2026-08-10 17:16 UTC)
# while real 4h price actually declined ~4.3% (0.04686 -> 0.04484, mostly
# lower closes, not sideways chop) - 6 BUY signals fired into that
# stretch: 4 losses, 2 breakeven scratches, zero wins. This adds a
# second, faster-updating HTF read - a plain EMA on the HTF candles,
# recomputed fresh every candle instead of needing 16h to confirm a new
# swing - and requires REAL, CURRENT price (not the stale swing
# structure) to still agree with it. Risk-REDUCING by construction (only
# ever rejects, never accepts more risk that wasn't already being taken)
# - ships live immediately rather than defaulting off, same as
# MIN_STOP_DISTANCE_ATR_MULTIPLE/MAX_SL_ROI_PCT.
HTF_TREND_FRESHNESS_ENABLED = env_bool("HTF_TREND_FRESHNESS_ENABLED", "True")
# Period (in HTF candles) for the freshness EMA - 20 on the default 4h
# HTF_KLINE_INTERVAL is ~80 real hours (~3.3 days): long enough to still
# be a genuine higher-timeframe read, short enough to update every
# candle instead of needing 16h to confirm a swing. Starting value, not
# yet calibrated against real trade data.
HTF_TREND_EMA_PERIOD = env_int("HTF_TREND_EMA_PERIOD", 20)
# EXPLICIT LIVE TEST (2026-08-27) - real mechanism (structure_state's
# htf_trend has zero time decay - see market_structure.py's own
# _classify_swings), but the systematic evidence behind THIS gate is
# genuinely weak: a 107-trade check (real market_structure.structure_
# state() run against real historical klines) found swing age does NOT
# predict win/loss monotonically - the FRESHEST tercile had the WORST
# win rate (65.7%), while the STALEST tercile still won 78.4% of the
# time. What staleness DOES predict is MAE (0.841R vs 0.58-0.63R) - a
# bumpier ride to the same win, not a more likely loss. This gate could
# plausibly reduce win rate by rejecting the currently-best-performing
# STALE trades while leaving the worst-performing FRESH ones untouched.
# User's own explicit choice to test this live anyway, same category as
# CROSS_EXCHANGE_OI_AGREE_REJECT_ENABLED/VP_EXTENSION_REJECT_ENABLED.
HTF_TREND_SWING_AGE_REJECT_ENABLED = env_bool("HTF_TREND_SWING_AGE_REJECT_ENABLED", "False")
# Starting value, not calibrated - roughly the p75 (62h) of the real
# 107-trade sample's swing-age distribution (median 43h), rounded up to
# a clean 3 days. No evidence this specific cutoff is the right one -
# see the flag's own comment above for the honest caveat.
HTF_TREND_MAX_SWING_AGE_HOURS = env_float("HTF_TREND_MAX_SWING_AGE_HOURS", 72.0)
# EXPLICIT LIVE TEST (2026-08-27) - user's own alternative to structure_
# state()'s swing-confirmed trend: reuses the EMA already computed for
# HTF_TREND_FRESHNESS_ENABLED (its own comment above) as the PRIMARY
# trend read for AGAINST_HTF_BIAS instead of a secondary veto. Zero
# resolved-trade evidence on this specific method - a fundamentally
# different, unvalidated mechanism, not a threshold tweak on a validated
# one. Automatically retires HTF_TREND_STALE/HTF_TREND_SWING_AGE_REJECT_
# ENABLED's own reject checks while this is on (see signal_engine.py) -
# both existed only to catch staleness in the mechanism this replaces.
HTF_TREND_EMA_PRIMARY_ENABLED = env_bool("HTF_TREND_EMA_PRIMARY_ENABLED", "False")

# =========================
# SIGNAL ENGINE - order-flow confirmation thresholds
# =========================
SIGNAL_MIN_CVD_SCORE = env_float("SIGNAL_MIN_CVD_SCORE", 0.15)
SIGNAL_MIN_DEPTH_IMBALANCE = env_float("SIGNAL_MIN_DEPTH_IMBALANCE", 0.10)
REQUIRE_ORDER_BLOCK_OR_FVG = env_bool("REQUIRE_ORDER_BLOCK_OR_FVG", "True")
# EMA alignment - the same direction-confirmation concept v7 uses (its
# EMA_WRONG_SIDE live-entry guard). Informational only, NOT a gate: an
# EMA is a lagging/smoothed indicator by construction, so requiring price
# already on its correct side can delay entry on a sharp move until price
# has run further - a real cost against this bot's real-time premise,
# and not yet backed by evidence it's worth paying. Computed and logged
# on every signal (ema_value, ema_aligned) so that evidence can
# accumulate before this is ever turned into a hard gate.
EMA_CONFIRMATION_ENABLED = env_bool("EMA_CONFIRMATION_ENABLED", "True")
EMA_CONFIRMATION_PERIOD = env_int("EMA_CONFIRMATION_PERIOD", 20)
# Separate, faster EMA used ONLY for the ema_aligned confluence field above
# - independent of EMA_CONFIRMATION_PERIOD, which still drives ema_value
# and therefore EMA_PULLBACK_TRIGGER_ENABLED's own live trigger detection
# (market_structure.detect_ema_pullback) unchanged. Real evidence
# (2026-08-24 underwater-duration audit, 80 resolved trades with real 1m
# price paths): at EMA_CONFIRMATION_PERIOD=20 on the 1h ltf_candles, that's
# a 20-HOUR EMA, while trades resolve on a 30min-5h scale (median 2.6h for
# wins) - a 20h trend can stay fully intact through a multi-hour corrective
# dip, so ema_aligned=True on 6 of 7 inspected non-CHOCH_RETEST losses and
# showed no protective correlation with adverse-excursion duration (if
# anything, aligned=True trades showed LONGER underwater streaks than
# aligned=False ones). 4 (hours, on 1h candles) chosen to match the median
# trade's own duration - reasoned, not yet outcome-validated against
# trades entered under it; revisit once enough resolve.
EMA_ALIGNMENT_PERIOD = env_int("EMA_ALIGNMENT_PERIOD", 4)
# Open Interest - informational only, NOT a gate (same treatment as EMA
# above). OI rising during a directional break points at fresh
# positioning behind the move (new longs on a bullish break, new shorts
# on a bearish one); OI falling on the same break points at the opposite
# side closing out instead (short-covering / long-liquidation) - a real
# distinction, but with no evidence yet on how much it separates winners
# from losers here. Polled via REST since Binance has no public OI
# websocket stream (it changes far slower than kline/aggTrade anyway).
OI_CONFIRMATION_ENABLED = env_bool("OI_CONFIRMATION_ENABLED", "True")
OI_POLL_INTERVAL_SECONDS = env_int("OI_POLL_INTERVAL_SECONDS", 60)
OI_LOOKBACK_SECONDS = env_int("OI_LOOKBACK_SECONDS", 900)
# Raised from 60 -> 1440 (2026-08-14): the old default only retained
# ~1 hour of history (60 samples x OI_POLL_INTERVAL_SECONDS=60s), enough
# for snapshot()'s own OI_LOOKBACK_SECONDS (900s) change-pct read, but
# nowhere near enough for OI_DIVERGENCE_TRIGGER_ENABLED below, which
# needs OI's value AT two swing points that can easily be many hours
# apart on this bot's 1h LTF (WS_KLINE_INTERVAL) with SWING_LEFT/RIGHT=4
# requiring several confirmed candles just to form one swing. 1440
# samples = ~24h at the default poll interval - cheap (one float per
# sample) and a reasoned starting point, not calibrated against how far
# apart real swing pairs actually land.
OI_HISTORY_MAX_SAMPLES = env_int("OI_HISTORY_MAX_SAMPLES", 1440)
# Evidence (2026-08-08, live, WATCHING=519): a large watchlist includes
# symbols the OI endpoint will never answer for - delisted/settling/
# pre-trading (-4108) or simply no longer valid (-1121, e.g. stale
# entries surviving in the hourly exchange-info cache). Left unhandled,
# these get retried and logged on every single poll cycle forever. Skip a
# symbol that just failed with one of these permanent-style errors for
# this long before trying it again, instead of hammering it every
# OI_POLL_INTERVAL_SECONDS indefinitely.
OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS = env_int("OI_UNAVAILABLE_SYMBOL_COOLDOWN_SECONDS", 3600)
# Reject outright when oi_rising is True - a real gate, not informational
# like the rest of OI_CONFIRMATION_ENABLED's fields above. Evidence
# (2026-08-18, 26 resolved binance_ai_bot_ds trades, entry signal recovered
# by symbol-match since a restart-recovered close's trade_id no longer
# links to its original entry row): oi_rising=True at entry saw DCA fire
# 69% of the time (9/13) vs 17% (2/12) when oi_rising was False/unavailable
# - the opposite of oi_rising's original "confirming" assumption baked into
# confluence_fields below (fresh OI piling in right as the trigger fires
# reads more like late, crowded positioning than genuine confirmation in
# this sample). None of the other confluence fields (ema_aligned came
# closer - 53% vs 22% - but not as wide) showed a comparable split, and
# signal_trigger itself didn't - CHOCH_RETEST's high DCA volume was just it
# being the highest-volume trigger overall, not disproportionately risky
# per DCA trade. Small sample (12/13 split inside 26 total, one trigger -
# OI_DIVERGENCE - never live in it) but risk-reducing by construction
# (only ever rejects, same precedent as EFFICIENCY_RATIO_GATE_ENABLED) -
# ships live immediately rather than sitting informational-only first.
# Applied universally (no TRIGGER_GATE_PROFILES entry, same as
# NOT_IN_DISCOUNT/DEPTH_OPPOSING below) since the evidence found no
# per-trigger split to differentiate on. No-ops whenever oi_rising itself
# is unavailable (OI_CONFIRMATION_ENABLED=False, or no OI data for the
# symbol yet).
OI_RISING_REJECT_ENABLED = env_bool("OI_RISING_REJECT_ENABLED", "True")
# Liquidation clustering - informational only, NOT a gate. Real forced
# liquidations at/around a detected sweep are the closest confirmation
# ICT's "stop hunt" concept has to actual ground truth: a BULLISH break
# (stops below a low swept, then price reverses up) should show real
# long-liquidation flow if it was a genuine stop hunt, not just a wick.
# Ported from v7's proven liquidation_shadow.py (`!forceOrder@arr`
# combined stream), simplified to this bot's direct-lock engine style
# (order_flow.py/orderbook.py) instead of v7's queue+worker-thread
# version, which existed for a heavier multi-symbol shadow monitor than
# this bot's single evaluate-on-tick usage needs.
LIQUIDATION_CONFIRMATION_ENABLED = env_bool("LIQUIDATION_CONFIRMATION_ENABLED", "True")
LIQUIDATION_WINDOW_SECONDS = env_int("LIQUIDATION_WINDOW_SECONDS", 120)
LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT = env_float("LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT", 50000)
LIQUIDATION_MAX_EVENTS_PER_SYMBOL = env_int("LIQUIDATION_MAX_EVENTS_PER_SYMBOL", 200)
# Funding rate - reflects how crowded long vs short positioning is
# market-wide for a symbol (strongly positive = longs paying heavily to
# stay long, a crowded trade more prone to a squeeze/reversal). Free:
# reuses the same premiumIndex endpoint exchange.get_mark_price() already
# calls per-position, just polled in bulk (every symbol, one call) like
# 24h volume. Informational only, not gated.
FUNDING_RATE_ENABLED = env_bool("FUNDING_RATE_ENABLED", "True")
FUNDING_POLL_INTERVAL_SECONDS = env_int("FUNDING_POLL_INTERVAL_SECONDS", 300)
# Long/short account ratio - who's positioned which way (distinct from
# OI's "how much is open" and funding's "cost of holding"). Unlike the
# endpoints above, Binance has no bulk "every symbol" version of this one
# - only one symbol per call - so it's deliberately NOT polled across the
# whole watchlist (that would mean one REST call per symbol per poll
# cycle, the exact shape of traffic this bot spent real effort avoiding
# elsewhere - see exchange._rate_limit_public_request). Fetched on-demand
# in main.py, only for a candidate that's already passed every other
# check, right before it would actually trade. Informational only.
LONG_SHORT_RATIO_ENABLED = env_bool("LONG_SHORT_RATIO_ENABLED", "True")
# Boolean "favorable" readings derived from efficiency_ratio/funding_rate/
# long_short_ratio - informational only, journaled but NOT fed into
# confluence_fields/confluence_ratio (see CONFLUENCE_SIZING_ENABLED below:
# that mechanism is disabled on real negative evidence already, so mixing
# new unvalidated fields into it would contaminate any future read of
# either). funding_rate/long_short_ratio stay informational-only for now;
# efficiency_ratio was promoted to a real gate below - see
# EFFICIENCY_RATIO_GATE_ENABLED.
EFFICIENCY_RATIO_CHOP_THRESHOLD = env_float("EFFICIENCY_RATIO_CHOP_THRESHOLD", 0.3)
# 2026-08-16, real evidence: a 29h window where every single signal read
# htf_trend=BULLISH while BTC (and a directly-traced traded symbol,
# POWERUSDT) were both genuinely flat/choppy the whole time, not trending -
# structure_state()'s swing-confirmed HTF trend can only ever report
# BULLISH or BEARISH, never "no real trend right now", so it freezes on a
# stale read through genuine chop instead of recognizing it.
# journal_analysis.py's efficiency-ratio breakdown for that exact window:
# zero trades in the "trending" bucket, 80% loss rate in "choppy" (<0.3),
# 100% in "moderate" (0.3-0.6) though that bucket was only n=3; 70% of
# losses ran real favorable distance (0.2R-1.0R+) before reversing - the
# whipsaw signature of a directional strategy trading a range. Gates only
# the "choppy" (<0.3) bucket for now (explicit operator choice, 2026-08-16)
# - the "moderate" bucket's 100% loss rate is real but too thin (n=3) to
# set a permanent cutoff on yet. Risk-reducing by construction (only ever
# rejects) so ships live immediately, same precedent as
# HTF_TREND_FRESHNESS_ENABLED/MIN_STOP_DISTANCE_ATR_MULTIPLE.
EFFICIENCY_RATIO_GATE_ENABLED = env_bool("EFFICIENCY_RATIO_GATE_ENABLED", "True")
# Real evidence (2026-08-25 signal-engine audit + direct re-check): the old
# default (0.0005) was measured against journaled outcomes and found to
# almost never vary - only 3 of 118 resolved trades ever read "unfavorable"
# under it, so the informational funding_favorable field wasn't measuring
# anything. Pulled the actual signed funding_rate (toward each trade's own
# side, same transformation signal_engine.py already applies below)
# distribution across all 118 resolved trades directly from the journal:
# p50=0.000043, p90=0.000100, p95=0.000245, max=0.000884. Recalibrated to
# 0.0001 - the real p90 of this bot's own trading conditions, not a guess -
# so ~10% of trades now land on the "unfavorable" side (n=11 today) instead
# of ~2.5% (n=3), enough variance to actually evaluate this field once more
# data accumulates. Purely a measurement fix: funding_favorable is
# informational-only (FUNDING_RATE_ENABLED/journaled, never gates a trade -
# see FUNDING_RATE_ENABLED's own comment), so this changes nothing about
# live trading behavior, only what gets recorded for future evaluation.
FUNDING_RATE_ADVERSE_THRESHOLD = env_float("FUNDING_RATE_ADVERSE_THRESHOLD", 0.0001)
LONG_SHORT_RATIO_CROWD_THRESHOLD = env_float("LONG_SHORT_RATIO_CROWD_THRESHOLD", 2.0)
# Require the LTF candle that broke structure to have actually CLOSED
# beyond the level before entering, instead of reacting to a still-forming
# candle's wick - a real behavior change (this bot's original premise was
# reacting before candle close), not just another journaled field.
# Evidence (2026-08-11, 40 resolved trades post-early-breakeven): 60% of
# remaining LOSS trades were near-zero MFE (wrong from the first tick) -
# up from 38% before early breakeven started peeling off the other loss
# population, meaning entry timing is now the dominant unsolved driver.
# The break_confirmed_by_close journal field (observational only, added
# 2026-08-10) showed the same direction on a thin sample: wick-only breaks
# that didn't hold to their candle's close were 4/4 (100%) LOSS vs 53% for
# closed-confirmed breaks. Combined with standard ICT/SMC doctrine (a BOS/
# CHoCH is only real once confirmed by a closed candle), that's enough to
# ship this gated, not just log it. Costs up to one candle of entry
# latency. Reversible at zero cost if the next batch doesn't show
# separation - see market_structure.live_break_check.
REQUIRE_CLOSE_CONFIRMED_BREAK = env_bool("REQUIRE_CLOSE_CONFIRMED_BREAK", "True")
# Second, alternative entry trigger alongside a live LTF structure break -
# a detected liquidity sweep (liquidity_sweep.detect_sweep: a wick through
# a known pool that closes back inside, the "run the stops then reverse"
# pattern), for symbols whose price rarely produces a clean structure
# break but does sweep organized liquidity (equal highs/lows, round
# numbers on major caps). Both triggers feed the SAME downstream pipeline
# (HTF bias, zone/OTE, order block/FVG, CVD/depth, extension cap, sizing)
# - never two independent pipelines, so at most one signal per symbol per
# eval tick regardless of which trigger(s) fire. Every signal is tagged
# signal_trigger=STRUCTURE_BREAK/LIQUIDITY_SWEEP (journaled) so win rate
# can be broken down by trigger before this path is trusted as much as
# the existing one. Default OFF, same convention as every other feature
# this session.
LIQUIDITY_SWEEP_TRIGGER_ENABLED = env_bool("LIQUIDITY_SWEEP_TRIGGER_ENABLED", "False")
# Third entry trigger: an LTF reversal already CONFIRMED (market_structure's
# last_event classified "CHoCH" - a break AGAINST the prior trend, not a
# continuation BOS) within CHOCH_TRIGGER_MAX_AGE_CANDLES candles of now,
# feeding the same downstream pipeline as every other trigger. Different
# from STRUCTURE_BREAK: that requires the CURRENT tick to be breaking a
# level right now; this lets an entry fire on the retracement back into a
# valid OTE/zone AFTER a reversal already confirmed, without needing a
# fresh break. structure_level is deliberately last_swing_low/last_swing_high
# (the current retracement level), NOT last_event["price"] (the NEW pivot
# that caused the event, e.g. a swing HIGH for a bullish reversal) - that
# distinction matters: the latter would place the stop just below a recent
# high instead of below the actual pullback low. Known, accepted tradeoffs
# (not solved here): (1) possible double-entry on the same reversal, since
# last_event only updates once SWING_RIGHT candles confirm the new pivot -
# often several candles (and past SYMBOL_REENTRY_COOLDOWN_SECONDS) after
# STRUCTURE_BREAK already traded the same move; (2) the downstream
# find_order_block gate only scans 10 candles back from NOW, so for a
# retest firing near the max age, the real order block from the original
# move is likely outside that window - weaker/noisier OB/FVG confirmation
# than STRUCTURE_BREAK gets. journal_analysis.py's per-trigger breakdown
# will show whether either is a real problem. Default OFF.
CHOCH_RETEST_TRIGGER_ENABLED = env_bool("CHOCH_RETEST_TRIGGER_ENABLED", "False")
CHOCH_TRIGGER_MAX_AGE_CANDLES = env_int("CHOCH_TRIGGER_MAX_AGE_CANDLES", 10)
# Real evidence (2026-08-21, 13 resolved CHOCH_RETEST trades with real
# setup_age_candles data - the journal field this reuses): every trade at
# age==CHOCH_TRIGGER_MAX_AGE_CANDLES (10) won (4/4); every younger trade
# (age 4-8) combined won only 2/9 (22%). CHOCH_RETEST was also the single
# worst-performing trigger overall in that same pull (47% win rate, 8 of
# the 15 total DCA events project-wide, on 30% of trade volume) - this is
# the one concrete, evidence-backed lever found for it. A CHoCH that's
# still fresh hasn't been retested/proven yet and is disproportionately a
# fakeout; one that's held for nearly the whole lookback window has shown
# it's real. Reject-only (can only make CHOCH_RETEST fire LESS often than
# today, never more) - same "safe to ship on real evidence immediately"
# precedent as OI_RISING_REJECT_ENABLED, not the "earns a live default
# only after evidence on the mechanism itself" class of feature.
#
# 9, not the literal max (10): no trade at age==9 exists yet in the
# evidence to confirm it shares the same pattern, but requiring exactly
# the maximum would reject every CHOCH_RETEST signal except the single
# oldest age the lookback window allows - stricter than the data actually
# demands. Revisit once age==9 trades exist to check against.
CHOCH_TRIGGER_MIN_AGE_CANDLES = env_int("CHOCH_TRIGGER_MIN_AGE_CANDLES", 9)
# Real evidence (2026-08-24, 32 resolved CHOCH_RETEST trades): depth_
# imbalance clearly favorable (signed >=0.10 toward the trade's own side)
# won 75.0% (n=12) vs only 55.0% (n=20) when merely neutral (-0.10..0.10 -
# already clearing the universal DEPTH_OPPOSING gate, which only rejects
# CLEARLY opposing depth, not "not yet favorable"). Tested the trend-
# agreement hypothesis first (does CHOCH_RETEST need AGAINST_HTF_BIAS/
# HTF_TREND_STALE after all) and found it flat - 62.5% win rate whether
# the trade agreed or disagreed with the swing-confirmed HTF trend, so
# that exemption stays as-is. Depth was the one real lead. Reject-only
# (can only make CHOCH_RETEST fire LESS often, never more) - same "safe
# to ship on real evidence immediately" precedent as CHOCH_TRIGGER_MIN_
# AGE_CANDLES above. 0 (or negative) disables, same convention as
# MIN_TP1_R_MULTIPLE/DCA_MIN_TP_R_MULTIPLE.
CHOCH_RETEST_MIN_DEPTH_IMBALANCE = env_float("CHOCH_RETEST_MIN_DEPTH_IMBALANCE", 0.10)
# Real evidence (2026-08-25 signal-engine audit, 33 resolved OB_FVG_RETEST
# trades, re-validated against a fresh VPS pull the same day with zero new
# trades in between): depth_imbalance clearly favorable (signed >=0.10
# toward the trade's own side) won 90.0% (n=10) vs only 73.9% (n=23) when
# merely neutral (-0.10..0.10 - already clearing the universal
# DEPTH_OPPOSING gate, which only rejects CLEARLY opposing depth, not "not
# yet favorable"). Same shape and comparable sample/effect size to
# CHOCH_RETEST_MIN_DEPTH_IMBALANCE above - the audit found this split held
# up specifically for OB_FVG_RETEST while the same aggregate looked flat
# (a trigger-mix confound: EMA_PULLBACK shows the OPPOSITE relationship on
# a thin n=5 sample and must NOT reuse this threshold). Reject-only (can
# only make OB_FVG_RETEST fire LESS often than today, never more) - same
# "safe to ship on real evidence immediately" precedent as the CHOCH
# version. 0 (or negative) disables, same convention as
# CHOCH_RETEST_MIN_DEPTH_IMBALANCE.
OB_FVG_RETEST_MIN_DEPTH_IMBALANCE = env_float("OB_FVG_RETEST_MIN_DEPTH_IMBALANCE", 0.10)
# Fourth entry trigger: a fresh rejection wick into an UNMITIGATED fair
# value gap (market_structure.find_fvg_retest) - independent of any live
# break right now, the classic OB/FVG "retest" entry. Scoped to FVGs only
# (order-block retest would need a new forward-scanning variant of
# find_order_block, meaningfully more engineering for a shape that's less
# clean than the flat FVG list - deferred). A separate, tighter max-age
# than FVG_LOOKBACK_CANDLES (50, fine as loose corroborating evidence for
# STRUCTURE_BREAK's existing gate, too generous for a standalone trigger
# where freshness should matter more). Default OFF.
OB_FVG_RETEST_TRIGGER_ENABLED = env_bool("OB_FVG_RETEST_TRIGGER_ENABLED", "False")
OB_FVG_RETEST_MAX_AGE_CANDLES = env_int("OB_FVG_RETEST_MAX_AGE_CANDLES", 20)
# How far the retest candle's CLOSE has to reclaim back out of the gap
# before the retest qualifies - see market_structure.find_fvg_retest's own
# docstring for the full reasoning. 0.0 = original behavior (close
# anywhere past the gap's far edge, no minimum reclaim depth). 0.5 =
# midpoint (default). Real motivation (2026-08-22): live OB_FVG_RETEST
# trades averaged ~0.68R max adverse excursion even on eventual WINS
# (28% of wins still ran 1R+ against the position first), and the
# original condition (close > bottom for BULLISH / close < top for
# BEARISH) accepted a close barely off the far edge - deep inside the
# zone - as an equally valid "retest" as a strong reclaim near the near
# edge. Not outcome-validated yet (the journal never captured
# close-position-within-gap before this change existed to check it
# against) - a reasoned starting point, not the literal strictest option
# (1.0 would reject genuine retests along with weak ones). Revisit once
# trades post-dating this change have resolved.
OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT = env_float("OB_FVG_RETEST_MIN_CLOSE_THROUGH_PCT", 0.5)
# Extra consecutive SIGNAL_CONFIRM_TICKS (main.py's SignalStabilityTracker)
# required for any trigger except the one proven trigger, STRUCTURE_BREAK -
# LIQUIDITY_SWEEP (already live), CHOCH_RETEST, and OB_FVG_RETEST all get
# the stricter bar. Deliberately includes LIQUIDITY_SWEEP even though it's
# already running: it hasn't been validated against real outcomes any more
# than the brand-new triggers have, so it's held to the same bar starting
# now rather than grandfathered in - a conscious, evidence-first behavior
# change on something already live, not an accident. Real evidence this
# codebase already has (CONFLUENCE_SIZING_ENABLED disabled on flat-to-
# inverse confluence-vs-outcome data; MIN_STOP_DISTANCE_PCT's comment on
# CVD/sweep confirmation being statistically identical between winners and
# losers) means the accuracy gain from adding more triggers has to come
# from here and from each trigger's own detection strictness - NOT from
# raising SIGNAL_MIN_CVD_SCORE/SIGNAL_MIN_DEPTH_IMBALANCE further, which
# isn't evidence-backed.
# Lowered from 2 -> 1 (2026-08-14, operator request): this was a blanket
# "trust newer triggers less" measure added when trigger count was
# growing, never tied to a specific traced incident and not targeted at
# the actual root cause later found for the "wrong direction entries"
# complaint (OTE_RETRACEMENT_MIN/zone_retracement_pct, see above) - now
# that the real fix is in, the operator asked to scale this back rather
# than keep paying its cost on every non-STRUCTURE_BREAK trigger. Real
# effect is small either way: at SIGNAL_EVAL_INTERVAL_SECONDS=3s, the
# difference between +1 and +2 extra ticks is only 3 real seconds on top
# of SIGNAL_CONFIRM_TICKS - negligible next to this bot's 1h LTF, so this
# mainly reduces friction for genuinely-brief flicker, not a major lever
# on its own (see MIN_24H_QUOTE_VOLUME_USDT/the trigger age-window
# settings for the levers with real trade-count leverage).
EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS = env_int("EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS", 1)
# A smaller extra-ticks requirement specifically for STRUCTURE_BREAK - the
# one trigger EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS deliberately left alone
# above. Lowered from 1 -> 0 (2026-08-14, same request as above) - back to
# the original zero-extra-cost behavior for the one trigger with the
# strongest track record, now that 8 trigger types exist and the newer
# ones already carry their own (reduced) extra-ticks cost above; no
# reason for the proven trigger to keep paying any tax on top of that.
# 0 preserves the original zero-extra-cost behavior.
STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS = env_int("STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS", 0)
# Instead of the fixed STRUCTURE_BREAK > OB_FVG_RETEST > LIQUIDITY_SWEEP >
# CHOCH_RETEST priority order (first match wins, nothing else attempted),
# gather every currently-qualifying trigger, gate each one for real (HTF
# bias/zone/OTE/OB-FVG/CVD/depth - most of these are direction-only, so in
# practice this is usually 1-2 full pipeline runs per tick, not one per
# candidate; see OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED below for the one
# gate that can now vary by trigger too), and among the survivors prefer
# whichever candidate's structure_level sits closest to current price
# (least already-chased - same philosophy as MAX_ENTRY_EXTENSION_R)
# INSTEAD of the fixed-priority default, but only when the edge is real -
# see TRIGGER_QUALITY_EDGE_ATR_MULTIPLE. Default OFF: disabled reproduces
# today's fixed-priority behavior byte-for-byte (confirmed via the
# existing test suite requiring zero changes when this stays False).
TRIGGER_QUALITY_RANKING_ENABLED = env_bool("TRIGGER_QUALITY_RANKING_ENABLED", "False")
# How much better (in ATR-multiples of price distance from the trigger
# level) an alternative same-direction candidate must be before it
# overrides the fixed-priority default, when TRIGGER_QUALITY_RANKING_ENABLED
# is on. Exists specifically to prevent a real risk found during design
# review: two close-scoring candidates could flip which one "wins" from
# ordinary tick-to-tick price noise, resetting main.py's
# SignalStabilityTracker streak every time and starving the setup of ever
# confirming - this hysteresis margin keeps the selection sticky unless
# there's a real, not noise-level, quality difference. Starting value, not
# yet calibrated against real trade data. 0 disables the hysteresis
# (always take the best-scored candidate, no margin required) - not
# recommended given the flapping risk above.
TRIGGER_QUALITY_EDGE_ATR_MULTIPLE = env_float("TRIGGER_QUALITY_EDGE_ATR_MULTIPLE", 0.25)
# =========================
# PER-TRIGGER GATE MATCHING
# =========================
# binance_ai_bot_ds's founding architectural difference from binance_ai_
# bot_smc: every trigger runs through only the gates that actually fit its
# own detection logic, by DEFAULT - not a shared cascade with narrow,
# opt-in, default-OFF exemptions bolted on after the fact (the right move
# for patching a LIVE bot without fragmenting its evidence pool, the wrong
# one for a project starting fresh). The 5 flags below still exist
# individually (each independently toggleable/overridable in .env, same
# env_bool convention as everywhere else) but default the way real
# evidence + code-reading already point, and TRIGGER_GATE_PROFILES below
# is the single declarative table signal_engine._evaluate_direction
# actually reads from - reviewable in one place instead of five scattered
# `chop_gate_applies = not (...)` expressions.
#
# Two trigger groupings referenced below:
#  - reversal/exhaustion triggers: bet AGAINST the prevailing move
#    (CVD_DIVERGENCE/OI_DIVERGENCE at a swing point that disagreed with
#    price; LIQUIDATION_SWEEP_CONFIRMED at a forced-liquidation-driven
#    extreme).
#  - CHOCH_RETEST joins them for the two HTF-trend-agreement gates only
#    (not MARKET_CHOPPY) - a CHoCH retest structurally IS "the trend just
#    changed", so requiring agreement with the OLD, lagging swing-
#    confirmed HTF trend is the same category of self-defeat, but chop
#    vs. trend is an orthogonal question CHOCH_RETEST has no special
#    relationship to either way.
_REVERSAL_TRIGGERS = frozenset({
    "CVD_DIVERGENCE", "OI_DIVERGENCE", "LIQUIDATION_SWEEP_CONFIRMED",
})
_TREND_AGREEMENT_EXEMPT_TRIGGERS = _REVERSAL_TRIGGERS | {"CHOCH_RETEST"}
_OB_FVG_TAUTOLOGICAL_TRIGGERS = frozenset({"OB_FVG_RETEST", "ORDER_BLOCK_RETEST"})

# AGAINST_HTF_BIAS requires the swing-confirmed HTF trend to already agree
# with this candidate's direction. A swing-confirmed trend is inherently
# LAGGING (see HTF_TREND_STALE_ENABLED's own rationale) - fine for
# trend-following triggers, but structurally rejects exactly the early
# trend-change signals _TREND_AGREEMENT_EXEMPT_TRIGGERS exist to catch.
# New in this project (not carried from binance_ai_bot_smc, which has no
# equivalent flag - that codebase never questioned this gate's fit for
# reversal triggers). Ships True: same reasoning class as the 3 carried-
# forward flags below, extended one step further - not yet outcome-
# validated against real trade data, since this project has none yet.
AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED = env_bool(
    "AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", "True"
)
# Same reasoning and same exempt-trigger group as AGAINST_HTF_BIAS above -
# HTF_TREND_STALE is just a second, faster-updating measure of the same
# "does this agree with the broader trend" question.
HTF_TREND_STALE_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED = env_bool(
    "HTF_TREND_STALE_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", "True"
)
# Carried forward from binance_ai_bot_smc's 2026-08-16 code-audit finding:
# NOT_IN_OTE requires current price to sit within a Fibonacci retracement
# band of the OVERALL HTF range - the classic "structure break, then
# retrace to OTE" setup, a real fit for STRUCTURE_BREAK specifically.
# Every other trigger already anchors its own entry to a DIFFERENT, more
# specific zone (FVG/order block range, swept liquidity pool, swing point,
# EMA) with no structural reason to also land in that band. Ships True
# here (unlike binance_ai_bot_smc's own default-OFF, evidence-gated
# rollout) since this project's whole premise is shipping the reasoned
# per-trigger default from day one rather than re-deriving it as an
# opt-in.
OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED = env_bool("OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED", "True")
# Carried forward, same audit: CVD_DIVERGENCE fires when price and CVD
# *disagree* at swing points; CVD_NOT_CONFIRMED then separately requires
# the RECENT-WINDOW CVD score to already *agree* with that same new
# direction - potentially self-defeating, filtering this trigger down to
# only its latest, most-obvious (worst risk/reward) setups. Narrow on
# purpose: OI_DIVERGENCE/LIQUIDATION_SWEEP_CONFIRMED don't share this
# specific self-defeating structure (they don't key off CVD at all), so
# they still get gated by CVD_NOT_CONFIRMED normally.
CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED = env_bool(
    "CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED", "True"
)
# Carried forward, same audit: MARKET_CHOPPY (EFFICIENCY_RATIO_GATE_ENABLED
# above) requires the broader market to read as "trending" - a real fit
# for the 6 trend/breakout-following triggers, but reversal-at-a-range-
# extreme is a legitimate setup INSIDE genuine chop, not despite it - this
# gate would reject exactly the setups _REVERSAL_TRIGGERS are best suited
# to catch. Deliberately excludes CHOCH_RETEST (unlike the two HTF-trend
# flags above) - chop vs. trend has no special relationship to a change-
# of-character signal either way.
MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED = env_bool(
    "MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED", "True"
)


def trigger_gate_profiles():
    """{trigger_name: frozenset(gate names that actually apply)} - the
    single source signal_engine._evaluate_direction reads from. Only
    gates with real per-trigger variation are listed; NOT_IN_DISCOUNT/
    NOT_IN_PREMIUM, DEPTH_OPPOSING, and OI_RISING apply to every trigger
    unconditionally (checked directly in signal_engine.py, no profile entry
    needed - see their own comments there for why they fit universally,
    unlike the six below).

    Deliberately a function called fresh on every use, NOT a module-level
    constant computed once at import time - the 5 flags above are exactly
    the kind of setting tests (and, eventually, a hot-reload) override via
    `patch.object(config, "...")` at run time, same as every other flag in
    this file. A cached dict built once from Python-import-time values
    would silently ignore any such override (real bug caught while fixing
    up this project's own test suite, 2026-08-17 - the "skip flag set to
    False" tests kept getting "OK" instead of the gate's rejection,
    because the cached profile never saw the patched value)."""
    all_variable_gates = frozenset({
        "AGAINST_HTF_BIAS", "HTF_TREND_STALE", "MARKET_CHOPPY",
        "NOT_IN_OTE", "NO_ORDER_BLOCK_OR_FVG", "CVD_NOT_CONFIRMED",
    })
    profiles = {}

    for trigger in (
        "STRUCTURE_BREAK", "OB_FVG_RETEST", "LIQUIDITY_SWEEP", "CHOCH_RETEST",
        "CVD_DIVERGENCE", "ORDER_BLOCK_RETEST", "OI_DIVERGENCE",
        "LIQUIDATION_SWEEP_CONFIRMED", "EMA_PULLBACK",
    ):
        gates = set(all_variable_gates)

        if AGAINST_HTF_BIAS_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED and trigger in _TREND_AGREEMENT_EXEMPT_TRIGGERS:
            gates.discard("AGAINST_HTF_BIAS")

        if HTF_TREND_STALE_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED and trigger in _TREND_AGREEMENT_EXEMPT_TRIGGERS:
            gates.discard("HTF_TREND_STALE")

        if MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED and trigger in _REVERSAL_TRIGGERS:
            gates.discard("MARKET_CHOPPY")

        if OTE_GATE_STRUCTURE_BREAK_ONLY_ENABLED and trigger != "STRUCTURE_BREAK":
            gates.discard("NOT_IN_OTE")

        if trigger in _OB_FVG_TAUTOLOGICAL_TRIGGERS:
            gates.discard("NO_ORDER_BLOCK_OR_FVG")

        if CVD_NOT_CONFIRMED_SKIP_FOR_CVD_DIVERGENCE_ENABLED and trigger == "CVD_DIVERGENCE":
            gates.discard("CVD_NOT_CONFIRMED")

        profiles[trigger] = frozenset(gates)

    return profiles
# Fifth entry trigger: price makes a new swing extreme (fractal swing
# point, the same detector STRUCTURE_BREAK/LIQUIDITY_SWEEP already use)
# that the CVD line does NOT confirm - classic order-flow divergence/
# absorption (see cvd_divergence.py). Genuinely different data than the
# existing SIGNAL_MIN_CVD_SCORE gate above: that's a recent 1m/5m/15m
# window, blind to price's own swing history; this compares CVD's value
# AT the last two swing points the same way price structure itself is
# compared. Needs order_flow.CVDEngine's persistent per-candle history
# (finalize_candle/cvd_history, see CVD_HISTORY_MAXLEN above) - the
# existing recent-window trade deque is deliberately pruned too
# aggressively (ORDER_FLOW_MAX_WINDOW_SECONDS) to span multiple swings.
# No candle-close-confirmation concept applies here (same as CHOCH_RETEST)
# - the comparison is between two already-confirmed swing points, not a
# currently-forming candle. Brand new, unvalidated mechanism - default
# OFF, same convention as every other trigger this session.
CVD_DIVERGENCE_TRIGGER_ENABLED = env_bool("CVD_DIVERGENCE_TRIGGER_ENABLED", "False")
# How large a gap (in USDT, cumulative CVD delta between the two compared
# swing points) counts as real divergence rather than noise - mirrors
# ORDER_FLOW_MIN_NOTIONAL_USDT's scale (the existing floor for trusting a
# CVD reading at all). Starting value, not yet calibrated against real
# trade data.
CVD_DIVERGENCE_MIN_DELTA_USDT = env_float("CVD_DIVERGENCE_MIN_DELTA_USDT", 5000)
# Temporary, read-only diagnostic (2026-08-26 signal-engine "next level"
# audit follow-up: CVD_DIVERGENCE has never produced a single trade,
# still true a full day and 12 more resolved trades after the 2026-08-25
# audit first flagged it). Unlike LIQUIDATION_SWEEP_CONFIRMED's own
# silence (see LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED below - that
# one turned out to be a real data-source gap, not a threshold problem),
# CVD's underlying data source is confirmed healthy elsewhere: cvd_score
# gates every trade successfully off the exact same feed. So this logs
# the REAL swing-to-swing CVD delta at every structural candidate (price
# making a new extreme), whether or not it clears CVD_DIVERGENCE_MIN_
# DELTA_USDT and whether or not the CVD-history lookup even found data at
# both swing points - so a future recalibration is evidence-based rather
# than a guess. Never gates anything, never changes any returned field -
# default True purely because it's observability-only, same convention as
# LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED; turn off once enough
# evidence has accumulated to decide.
CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED = env_bool(
    "CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", "True"
)
# Sixth entry trigger: a fresh rejection wick back into a previously-
# formed, UNMITIGATED order block (market_structure.find_order_block_
# retest) - the order-block counterpart to OB_FVG_RETEST_TRIGGER_ENABLED
# above, deliberately deferred when that one was built (see its own
# comment: needed a new forward-scanning variant of find_order_block,
# more engineering than the flat FVG list needed at the time). Now built
# via find_structure_events/find_order_blocks - every historical BOS/
# CHoCH's origin block, not just the single most-recent one
# REQUIRE_ORDER_BLOCK_OR_FVG already reads. Brand new, unvalidated
# mechanism - default OFF, same convention as every other trigger.
ORDER_BLOCK_RETEST_TRIGGER_ENABLED = env_bool("ORDER_BLOCK_RETEST_TRIGGER_ENABLED", "False")
ORDER_BLOCK_RETEST_MAX_AGE_CANDLES = env_int("ORDER_BLOCK_RETEST_MAX_AGE_CANDLES", 20)
# How many of the most recent confirmed BOS/CHoCH events to derive order
# blocks from - bounds both compute cost and staleness (an origin block
# from 30 structure breaks ago is no longer a meaningful retest target).
ORDER_BLOCK_RETEST_LOOKBACK_EVENTS = env_int("ORDER_BLOCK_RETEST_LOOKBACK_EVENTS", 5)
# Seventh entry trigger: price's swing structure vs OPEN INTEREST's value
# at those same swing points (oi_divergence.py) - a new price extreme not
# backed by expanding OI is weaker evidence than one where OI genuinely
# built up alongside it. Same divergence concept as CVD_DIVERGENCE above,
# different metric - reuses open_interest.OpenInterestEngine's existing
# per-symbol history (OpenInterestEngine.history(), see OI_HISTORY_
# MAX_SAMPLES above for why that retention window was widened
# specifically to support this). Needs OI_CONFIRMATION_ENABLED=True (the
# OI poll only runs when that's on - see ws_client._start_oi_poll) or
# this trigger will simply never have data to work with. Brand new,
# unvalidated mechanism - default OFF, same convention as every other
# trigger.
OI_DIVERGENCE_TRIGGER_ENABLED = env_bool("OI_DIVERGENCE_TRIGGER_ENABLED", "False")
# Minimum OI decline (%, across the two compared swing points) that
# counts as real divergence rather than noise. Starting value, not yet
# calibrated against real trade data.
OI_DIVERGENCE_MIN_DELTA_PCT = env_float("OI_DIVERGENCE_MIN_DELTA_PCT", 5.0)
# Same shape as CHOCH_TRIGGER_MAX_AGE_CANDLES/ORDER_FLOW_DIVERGENCE_
# LOOKBACK - how stale the qualifying swing point is allowed to be before
# this trigger stops firing on it. Kept as its own knob (not reusing
# ORDER_FLOW_DIVERGENCE_LOOKBACK) since OI_DIVERGENCE is a genuinely
# distinct trigger from CVD_DIVERGENCE and may need a different staleness
# tolerance once real data exists for both.
OI_DIVERGENCE_TRIGGER_MAX_AGE_CANDLES = env_int("OI_DIVERGENCE_TRIGGER_MAX_AGE_CANDLES", 20)
# Temporary, read-only diagnostic - same shape and same motivation as
# CVD_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED above: OI_DIVERGENCE has never
# produced a single trade either, and OI's underlying data source is also
# confirmed healthy elsewhere (oi_rising, fed by the same OI history, is
# the single strongest gate in the whole system - re-confirmed on 117+
# resolved trades). Logs the real swing-to-swing OI delta (%) at every
# structural candidate, whether or not it clears OI_DIVERGENCE_MIN_
# DELTA_PCT and whether or not the OI-history lookup found a sample at
# both swing points. Never gates anything - default True, observability-
# only, same convention as its CVD sibling.
OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED = env_bool(
    "OI_DIVERGENCE_DIAGNOSTIC_LOGGING_ENABLED", "True"
)
# Eighth entry trigger: promotes a plain LIQUIDITY_SWEEP into a stricter,
# distinct trigger by additionally requiring a REAL clustered forced-
# liquidation event backing it (liquidity_sweep.detect_liquidation_
# confirmed_sweep) - not just the informational-only liquidation_aligned/
# liquidation_cluster fields every trigger already journals, but a
# genuine gating condition: the swept level actually forced real
# positions closed in the sweep's direction. Reuses
# LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT (no new notional threshold - the
# existing "is this liquidation flow big enough to matter at all"
# question is the same question here) and the exact same alignment
# formula signal_engine.py already uses for the informational field, not
# a new definition. Can only ever be MORE selective than LIQUIDITY_SWEEP
# alone, never a relaxation of it. Needs LIQUIDATION_CONFIRMATION_ENABLED=
# True (gates the liquidation websocket stream itself - see ws_client.
# _start_liquidation_stream) or this trigger will never have data. Brand
# new, unvalidated mechanism - default OFF, same convention as every
# other trigger.
LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED = env_bool(
    "LIQUIDATION_SWEEP_CONFIRMED_TRIGGER_ENABLED", "False"
)
# Temporary, read-only diagnostic (2026-08-25 signal-engine audit finding:
# LIQUIDATION_SWEEP_CONFIRMED has never produced a single trade).
# LIQUIDATION_CLUSTER_MIN_NOTIONAL_USDT=50000 within a 120s window
# (LIQUIDATION_WINDOW_SECONDS) is realistic for BTCUSDT/ETHUSDT but may be
# unrealistic for this bot's actual (mostly small/mid-cap) symbol set -
# but there's no historical liquidation-notional data to check that
# against without guessing a new number. Logs the REAL notional magnitude
# seen at every genuine LIQUIDITY_SWEEP event (whether or not it clears
# the threshold), so a future recalibration is evidence-based. Never
# gates anything, never changes any returned field - default True purely
# because it's observability-only; turn off once enough evidence has
# accumulated to decide.
LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED = env_bool(
    "LIQUIDATION_SWEEP_DIAGNOSTIC_LOGGING_ENABLED", "True"
)
# Ninth entry trigger: a pullback to the EMA within an established trend,
# followed by a same-candle reclaim (see market_structure.
# detect_ema_pullback) - the classic trend-continuation entry. Real
# motivation (2026-08-14, direct evidence): with all 8 other triggers
# live for a full session, a live bot.log check showed BTCUSDT/ETHUSDT/
# BNBUSDT/SOLUSDT produced ZERO signal-related log activity (no SIGNAL,
# no plan-rejected, no position-closed - not even a rejection at the
# final gate the way alts get, filtered out before any per-symbol event
# ever fires). Majors trend smoothly with shallow pullbacks and
# naturally balanced, deep order flow, so the deep OTE retracement
# (OTE_RETRACEMENT_MIN=0.705) and CVD/depth imbalance thresholds every
# other trigger's downstream gate requires almost never get satisfied on
# them, no matter how many detector TYPES exist upstream feeding that
# same gate. This one is structurally different: it only needs a
# shallow touch-and-reclaim of the trend's own moving average, not a
# deep retracement or an order-flow imbalance - closer to how majors are
# actually traded. Needs config.EMA_CONFIRMATION_ENABLED=True (already
# on) - reuses the same ema_value already computed for the informational
# ema_aligned field in signal_engine.py, zero extra cost, rather than a
# second EMA calculation. Still subject to every other shared downstream
# gate (HTF bias, zone, OTE, REQUIRE_ORDER_BLOCK_OR_FVG, CVD, depth) like
# every other trigger - this doesn't bypass those, it just adds one more
# way to produce a candidate before they're checked. Brand new,
# unvalidated mechanism - default OFF, same convention as every other
# trigger.
EMA_PULLBACK_TRIGGER_ENABLED = env_bool("EMA_PULLBACK_TRIGGER_ENABLED", "False")

# =========================
# RISK MANAGEMENT (ported convention from v7/v8)
# =========================
MARGIN_PER_TRADE = env_float("MARGIN_PER_TRADE", 10)
LEVERAGE = env_int("LEVERAGE", 10)
MARGIN_TYPE = os.getenv("MARGIN_TYPE", "ISOLATED")
RISK_BASED_POSITION_SIZING_ENABLED = env_bool(
    "RISK_BASED_POSITION_SIZING_ENABLED", "True"
)
POSITION_RISK_PCT = env_float("POSITION_RISK_PCT", 1.0)
POSITION_RISK_MAX_USDT = env_float("POSITION_RISK_MAX_USDT", 0)
STRUCTURE_STOP_ATR_BUFFER = env_float("STRUCTURE_STOP_ATR_BUFFER", 0.5)
# Hard floor: SL is never allowed closer to entry than this % of entry
# price, regardless of how close the structure level happened to land -
# prevents a pathologically tight stop (and the oversized position that
# risk-based sizing would produce to compensate for it). Evidence
# (2026-08-08, 79 resolved live trades): 68% hit SL despite CVD/sweep
# confirmation being statistically identical between winners and losers -
# ruling out signal-selection quality as the driver and pointing at the
# stop still being too tight for normal price movement even with the
# floor active (observed average stop distance on SL-hit trades was only
# ~0.45%). Raised from 0.3 -> 0.6 on that evidence.
MIN_STOP_DISTANCE_PCT = env_float("MIN_STOP_DISTANCE_PCT", 0.6)
# A second minimum-stop-distance floor, in ATR multiples rather than a flat
# percentage of price - risk_manager._apply_min_stop_distance takes
# whichever of the two floors is WIDER. A flat percentage can't be "enough"
# for every symbol at once: real evidence (2026-08-13, a direct log-
# distance trace of 24 real trades plus three consecutive
# journal_analysis.py pulls) showed MIN_STOP_DISTANCE_PCT (0.6%) getting
# hit on ~40% of trades, with every floor-clamped trade resolving as a
# loss or scratch - 0.6% sits inside normal 1h noise for the volatile
# small/mid-cap symbols this watchlist trades most often. This scales the
# floor with each symbol's own measured volatility instead of guessing one
# number for all 400 watchlist symbols. Risk-REDUCING by construction
# (only ever widens the stop further, which only ever shrinks position
# size for the same $ risk) - so unlike every trade-count/entry-logic
# feature this session, this ships live immediately rather than defaulting
# off. Starting value, not yet calibrated against real trade data. 0
# disables it (MIN_STOP_DISTANCE_PCT alone, the original behavior).
MIN_STOP_DISTANCE_ATR_MULTIPLE = env_float("MIN_STOP_DISTANCE_ATR_MULTIPLE", 1.0)
# Rejects an entry that's already run more than this many R beyond the
# structure level that triggered it - chasing an already-extended move
# instead of catching it near the level that made the setup valid. Real
# motivation (2026-08-12, live): REQUIRE_CLOSE_CONFIRMED_BREAK (up to an
# hour of delay on 1h candles) plus SIGNAL_CONFIRM_TICKS (another ~10-15s)
# can let price run well past the break level before entry actually
# fires, and execution.py always market-orders at whatever price exists
# by then - there was no check on how far that already was from the
# level that made the setup valid. Expressed in R (the same risk_distance
# used everywhere else in this file), not raw price/ATR, so it scales
# with each symbol's own volatility. Starting value is a reasonable
# floor, not yet calibrated against real trade data. 0 disables it.
MAX_ENTRY_EXTENSION_R = env_float("MAX_ENTRY_EXTENSION_R", 0.5)
# Rejects an entry whose stop, once LEVERAGE is applied, would lose more
# than this % of the margin actually at risk if hit - independent of
# position sizing mode, since quantity cancels out of the ratio
# (ROI_at_SL = stop_distance_% * LEVERAGE, see
# risk_manager._stop_roi_too_high). Real motivation (2026-08-14, operator
# feedback): risk-based sizing already caps the ACCOUNT-level $ loss per
# trade (POSITION_RISK_PCT) regardless of stop width, but the POSITION-
# level ROI% (what the exchange UI shows - PnL against margin used) is a
# different number entirely, and wasn't capped anywhere - a wide
# structural stop (which MIN_STOP_DISTANCE_ATR_MULTIPLE can now produce
# more of, on purpose, to avoid noise-driven SL hits) can show a large
# ROI% loss on that specific position even though the account-level risk
# never changed. Rejects the trade outright rather than shrinking its
# size, per explicit operator choice - a stop this wide relative to
# leverage is treated as not worth taking at any size. Risk-REDUCING by
# construction (only ever rejects, never accepts more risk), so ships
# live immediately rather than defaulting off, same as
# MIN_STOP_DISTANCE_ATR_MULTIPLE. 0 disables it.
# Real evidence this needs real room (2026-08-14): the operator had
# tightened this to 8, then 10, live - at LEVERAGE=10 that caps any
# accepted stop at 1%/1.2% of entry price. Since MIN_STOP_DISTANCE_
# ATR_MULTIPLE routinely produces wider stops than that on volatile
# symbols (by design - that's the fix it exists for), the two fought each
# other: 147 of 155 plan rejections in one live bot.log pull were
# SL_ROI_TOO_HIGH (94.8%), effectively blocking almost every signal that
# had already cleared every trigger/structural gate. Restored to 30 (the
# original starting value) to give the ATR floor room to actually work -
# still a real cap (rejects anything wider than 3% of price at this
# leverage), just not one that fights an already-proven mechanism by
# default.
MAX_SL_ROI_PCT = env_float("MAX_SL_ROI_PCT", 30)
# Confluence-weighted position sizing - see signal_engine.py's
# confluence_ratio (how many of EMA/OI/BTC agree with the signal, out of
# how many were actually available to check - sweep_confluence and
# liquidation_aligned were removed from this list 2026-08-25, see
# signal_engine.py's own comment on confluence_fields for the evidence).
# Scales the
# risk taken per trade instead of gating entry on any of these
# individually: every signal that qualifies today still trades, a
# 0-confluence one just risks less and a fully-aligned one risks more.
# Chosen over a hard gate specifically because it's testable against the
# existing trade count immediately (every trade gets sized, not just a
# rejected subset), reversible at zero cost if the score turns out
# uncorrelated with outcome, and can extract information from these
# fields collectively even before any one of them individually clears a
# significance bar on its own.
# DISABLED 2026-08-09 on real evidence: a 54-trade journal_analysis.py
# pull showed confluence_score trending flat-to-inverse against outcome
# (score=1 80% loss, score=2 96% loss, score=3 89% loss) - the opposite
# of what this multiplier assumes. Exercising the "reversible at zero
# cost" design above. confluence_score/ratio is still computed and
# journaled either way - re-enable only if a larger, cleaner sample
# (see MAE_TRACKING_ENABLED below) actually shows separation.
CONFLUENCE_SIZING_ENABLED = env_bool("CONFLUENCE_SIZING_ENABLED", "False")
CONFLUENCE_SIZING_MIN_MULTIPLIER = env_float("CONFLUENCE_SIZING_MIN_MULTIPLIER", 0.5)
CONFLUENCE_SIZING_MAX_MULTIPLIER = env_float("CONFLUENCE_SIZING_MAX_MULTIPLIER", 1.25)
MAX_TOTAL_POSITIONS = env_int("MAX_TOTAL_POSITIONS", 2)
# After ANY position closes (win, loss, or breakeven), that symbol is
# skipped for this long before it can be re-entered. Evidence (2026-08-08
# review): RSRUSDT/SANDUSDT/TAIKOUSDT/SUSHIUSDT each hit SL repeatedly
# within seconds-to-minutes of the previous close, at nearly the same
# level - immediate re-entry into a symbol that's actively chopping
# instead of waiting for the picture to change. Raised from 900 -> 3600
# (2026-08-14) on the same pattern recurring at the original value's own
# timescale: MUBARAKUSDT/GRAMUSDT/AEROUSDT each re-entered 2-3 times
# within a few hours, repeatedly losing at nearly the same level - 15
# minutes clearly wasn't long enough to let the picture actually change on
# a 1h LTF. Starting value, not yet calibrated against real trade data.
SYMBOL_REENTRY_COOLDOWN_SECONDS = env_int("SYMBOL_REENTRY_COOLDOWN_SECONDS", 3600)

# =========================
# TP1 / TP2 (mirrors v7's partial-TP + full-close ladder: TP1 closes
# TP1_CLOSE_PCT of the position and moves the remainder's stop to
# breakeven; TP2 closes what's left)
# =========================
TP1_CLOSE_PCT = env_float("TP1_CLOSE_PCT", 50)
# Real bug found live (2026-08-17, operator observation - "TP1 feels too
# high", checked against binance_ai_bot_smc's own real trade history since
# this R-multiple was inherited unchanged from there: 270 real trades, 237
# resolved). Outcome split: SL_HIT 173 (73%), BREAKEVEN_STOP_HIT 35 (15%,
# reached TP1 then scratched), TP2_HIT 29 (12%). Of the 114 SL_HIT losses
# with MFE data, only 3% ever ran far enough to reach the OLD 2.0R TP1
# target before reversing - but 22% ran to at least 1.0R. That's a real,
# sizeable population of trades currently guaranteed a full loss that a
# closer target would convert into a partial win instead. Rough
# expectancy at the old 2.0/4.0 split: 0.73*(-1R) + 0.15*(0) + 0.12*(+4R)
# ~= -0.25R/trade - negative. Of trades that DO reach TP1, 45% (29/64) go
# on to a full TP2 win - that leg isn't obviously broken the same way, so
# TP2_R_MULTIPLE/the MAX multiples below are left alone; this is a TP1-
# specific fix. Evidence is inherited from smc, not binance_ai_bot_ds's
# own trade history yet - revisit once this project has enough of its own.
TP1_R_MULTIPLE = env_float("TP1_R_MULTIPLE", 1.0)
TP2_R_MULTIPLE = env_float("TP2_R_MULTIPLE", 4.0)
# Upper bound on how far a real structure target is allowed to be. The
# R-multiples above are a MINIMUM room requirement - without a maximum
# too, "nearest qualifying pool" can still land absurdly far away if
# nothing closer exists (seen live: a ~20R target that's realistically
# never reached), silently turning TP1 into an unreachable target instead
# of an achievable first partial. Beyond this, the plain R-multiple
# fallback is used instead of the distant pool.
TP1_MAX_R_MULTIPLE = env_float("TP1_MAX_R_MULTIPLE", 6.0)
TP2_MAX_R_MULTIPLE = env_float("TP2_MAX_R_MULTIPLE", 10.0)
# Operator-requested alternative to TP1's own calculation (2026-08-19,
# revised 2026-08-21): TP1 becomes a fixed ROI% target (at LEVERAGE,
# risk_manager.price_at_roi_pct - the same math DCA_TP_STATIC_ROI_ENABLED
# already uses for the post-DCA target) instead of the structure-resolved
# one - TP2 stays exactly the existing real-liquidity-first structure
# target (risk_manager.compute_static_tp1_structure_tp2 reuses compute_
# targets' own TP2 resolution, just fed the static TP1 price as its "at
# least 1R beyond TP1" floor input). Still a normal TP1(partial)+TP2
# (remainder) close with TP1_CLOSE_PCT and breakeven-on-TP1-fill
# promotion - nothing about that machinery changes, only how TP1's PRICE
# gets computed. Applies to a DCA_PENDING position before DCA ever fires -
# see risk_manager.build_trade_plan/position_manager.register_dca_pending.
#
# 2026-08-19 original shipped shape (superseded 2026-08-21 on operator
# request): a single whole-position target instead, no TP2/partial close
# at all. That shape's downstream handling (single_tp - position_manager.
# poll_live/poll_shadow's DCA_PENDING branches, execution.
# place_dca_protection_orders) is NOT removed - DCA_ACTIVE (post-DCA)
# always uses it unconditionally regardless of this flag, and it still
# services any already-open position that entered under the old shape
# before this change. New signals under this flag no longer produce it.
#
# Default False - same "new mechanism earns a live default only after
# real data" rule every other unvalidated mechanism this project ships
# follows.
TP_STATIC_ROI_ENABLED = env_bool("TP_STATIC_ROI_ENABLED", "False")
# Leveraged ROI%, same units as DCA_TP_TARGET_ROI_PCT/
# PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT elsewhere in this file.
# Starting value, not calibrated against real trade outcomes yet.
TP_TARGET_ROI_PCT = env_float("TP_TARGET_ROI_PCT", 40)
MOVE_SL_TO_BREAKEVEN_AFTER_TP1 = env_bool(
    "MOVE_SL_TO_BREAKEVEN_AFTER_TP1", "True"
)
# Real gap found live (2026-08-24, DCA -> breakeven -> SL investigation):
# a breakeven SL still closed at a net loss. Every fill in that flow
# (original entry, DCA fill, SL exit) is a taker order - Binance USDⓈ-M
# futures standard taker fee is ~0.05%/side, so a round trip alone (~0.10%)
# already exceeded the old 0.02% buffer before any slippage. 0.15% covers
# that round-trip fee assumption plus a real slippage margin - a reasoned
# estimate, not tied to this account's exact fee tier (BNB discount/VIP
# tier could lower the real number) - revisit once confirmed. Verified
# mathematically that a flat % buffer scales correctly with the doubled
# post-DCA quantity (fees and buffer both scale with notional together),
# so the same value is correct for both the pre-DCA/TP1 and post-DCA
# breakeven cases - no separate DCA-specific buffer needed.
BREAKEVEN_BUFFER_PCT = env_float("BREAKEVEN_BUFFER_PCT", 0.15)
# Early breakeven - protects profit on a trade before it reaches TP1,
# instead of leaving the original (wider) stop in place the whole way
# there. Originally gated on confluence_ratio (protect low-confidence
# trades faster) and disabled 2026-08-09 when real data showed confluence
# didn't correlate with outcome at all. Re-enabled 2026-08-10 with a
# different, evidence-backed trigger: a journal_analysis.py MAE/MFE
# distribution pull (61 resolved LOSS trades) showed a clear bimodal
# split - 38% of losses were near-zero MFE (wrong from the first tick,
# nothing here helps them), but 28% ran 1.0R+ in profit before fully
# reversing to a full loss, completely unprotected the whole way down
# since nothing moves the stop until TP1 formally triggers at 2R. This
# now applies to every trade still waiting on TP1 (not just low-
# confluence ones): once price has moved EARLY_BREAKEVEN_R_MULTIPLE R in
# its favor, the SL moves to breakeven. Known tradeoff: a genuine winner
# that dips back through breakeven on its way to a real TP1/TP2 would
# close early instead of running - real cost, not yet measured, weighed
# against the 28% of losses this targets. Does not change entry/trade
# count - same principle as the sizing feature: adapt what happens to a
# trade that's already happening, not whether it happens.
EARLY_BREAKEVEN_ENABLED = env_bool("EARLY_BREAKEVEN_ENABLED", "True")
# Lowered from 1.0 -> 0.5 (2026-08-14) on real evidence: a 20-trade
# journal_analysis.py pull showed a stark, clean split - trades that
# reached early_breakeven_applied=True had a 0% loss rate (0 of 5 - 3 WIN,
# 2 BREAKEVEN), while trades that never reached it had a 77% loss rate (10
# of 13). Once a trade gets ANY real room, this mechanism protects it
# almost perfectly - the entire loss problem is concentrated in trades
# that never reach the trigger point at all. Lowering the bar gets more
# trades protected sooner, before they've had a chance to fully reverse.
# Known tradeoff, same shape as the original 2026-08-10 rationale below: a
# genuine winner that dips back through a NOW-CLOSER breakeven level on
# its way to TP1/TP2 closes early instead of running - a real cost that
# grows as this value shrinks, not yet measured against the loss
# reduction. Not yet calibrated against real trade data at this new value.
EARLY_BREAKEVEN_R_MULTIPLE = env_float("EARLY_BREAKEVEN_R_MULTIPLE", 0.5)
# How much profit (as an R-multiple) to lock in when early breakeven
# promotes a trade, instead of moving the stop to flat entry (a scratch).
# 0 preserves the original flat-breakeven behavior (see
# risk_manager.compute_early_breakeven_price). Rationale (2026-08-11): the
# early-breakeven population so far is roughly half WIN, half BREAKEVEN
# with zero LOSS - a modest lock converts some of those scratches into
# small realized wins instead of leaving them at exactly zero. Known
# tradeoff, not yet measured: a genuine TP1/TP2 runner that dips slightly
# on its way to target now gets stopped out at this smaller locked amount
# instead of running further. Watch the WIN vs BREAKEVEN split among
# early_breakeven_applied=True trades after this ships - see the
# EARLY_BREAKEVEN_PROFIT_HIT outcome in journal_analysis.py.
EARLY_BREAKEVEN_LOCK_R_MULTIPLE = env_float("EARLY_BREAKEVEN_LOCK_R_MULTIPLE", 0.3)
# Profit protection - a SECOND, independent early-promotion mechanism
# alongside EARLY_BREAKEVEN above, measured differently: instead of an
# R-multiple of risk_distance, this is a % of what TP1 itself would pay
# out in ROI (at LEVERAGE) - see risk_manager.compute_profit_protection_
# lock_price. Real motivation (2026-08-14, operator feedback): TP1/TP2
# can take a long time to actually fill, and EARLY_BREAKEVEN's fixed
# 0.5R/0.3R trigger/lock is a small, flat amount regardless of how big
# TP1's actual target is on a given trade - a trade already sitting on
# real, meaningful profit (a real fraction of TP1's own payout) has
# earned more protection than a flat 0.3R lock gives it, and this bot's
# only other profit-protection mechanism between entry and TP1
# (STRUCTURE_STOP_MANAGEMENT_ENABLED's trailing stop) only moves when a
# NEW confirmed swing forms - on this 1h LTF with SWING_LEFT/RIGHT=4 that
# can lag many hours behind real, growing unrealized profit. Mutually
# exclusive with EARLY_BREAKEVEN at the promotion moment (whichever
# threshold is reached first wins, both check position stage==TP1_PENDING
# and stop applying once promoted) but STRUCTURE_STOP_MANAGEMENT_ENABLED's
# trailing stop still runs on TOP of either afterward, same as today.
# Locks in the SAME ROI% that triggered activation (not a smaller
# cushion, not continued trailing past it) - explicit operator choice.
# Brand new, unvalidated mechanism - default OFF.
PROFIT_PROTECTION_ENABLED = env_bool("PROFIT_PROTECTION_ENABLED", "False")
# What % of TP1's own ROI counts as "enough profit to protect". E.g. at
# 60: if TP1 would pay out 50% ROI, protection activates once unrealized
# ROI reaches 30% (60% of 50) and locks the stop at that same 30% ROI
# level. Starting value, not yet calibrated against real trade data.
PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1 = env_float(
    "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1", 60
)
# Real operator concern (2026-08-18): waiting for 80% of TP1's own ROI
# feels safe when TP1 only pays out a modest amount, but not when TP1
# itself is a big ROI move - 80% of a large number is still a large
# amount of unrealized profit sitting completely unprotected while
# waiting for activation. When TP1's own ROI exceeds this threshold, a
# separate (lower) activation fraction is used instead - see
# PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI below. Below the
# threshold, behavior is unchanged (plain PROFIT_PROTECTION_ACTIVATION_
# PCT_OF_TP1 still applies). Only the ARM trigger is tiered this way -
# PROFIT_PROTECTION_LOCK_PCT_OF_TP1/RETRACE_PCT (what happens once
# already armed) are untouched, same for every trade regardless of TP1's
# own ROI.
PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT = env_float(
    "PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT", 50
)
# The activation fraction used instead of PROFIT_PROTECTION_ACTIVATION_
# PCT_OF_TP1 once TP1's own ROI clears the threshold above - deliberately
# lower, so a big-ROI TP1 still arms protection at a comparable absolute
# ROI level instead of making the operator wait through a much larger
# unrealized-profit swing first. Starting value, not yet calibrated
# against real trade data.
PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI = env_float(
    "PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1_HIGH_ROI", 50
)
# 2026-08-15, explicit operator report: locking the SL at the exact
# activation price left ~zero room between the stop and current price
# the moment it fired, so ordinary noise closed the trade right at
# activation instead of letting it run. Fix (mirrors v7's
# evaluate_route_profit_protection trigger_roi/lock_roi/retrace_pct
# shape): once armed, the SL now trails a cushion behind the best price
# reached since arming instead of jumping once to the activation price
# and stopping. PROFIT_PROTECTION_LOCK_PCT_OF_TP1 is the worst-case
# floor (same TP1-relative scaling as ACTIVATION_PCT_OF_TP1, deliberately
# smaller); PROFIT_PROTECTION_RETRACE_PCT is how much of the gain since
# entry to the peak is allowed to give back before the trailing stop
# catches up. Starting values, not yet calibrated against real trade data.
PROFIT_PROTECTION_LOCK_PCT_OF_TP1 = env_float(
    "PROFIT_PROTECTION_LOCK_PCT_OF_TP1", 25
)
PROFIT_PROTECTION_RETRACE_PCT = env_float("PROFIT_PROTECTION_RETRACE_PCT", 50)
# Real gap found live (2026-08-22, operator observation): every tiered
# ACTIVATION_PCT_OF_TP1/_HIGH_ROI knob above assumes TP1 itself is a
# variable, potentially large target - structure-based TP1 always was.
# Under TP_STATIC_ROI_ENABLED, TP1 is a small, FIXED ROI% (TP_TARGET_
# ROI_PCT, typically ~10%) well under PROFIT_PROTECTION_HIGH_TP1_ROI_
# THRESHOLD_PCT, so it always lands on the plain (80%) activation branch -
# arming at ~8% ROI and then locking/retracing from there, on a target
# that was only ever 10% away in the first place. Real trade (TREEUSDT,
# 2026-08-21): armed at 8% ROI, price ran to ~9% (never reached the 10%
# TP1), reversed, and closed via the trailing stop for ~4% gross - a
# fraction of a target that was already small and often achievable in
# full. Unlike EARLY_BREAKEVEN (a flat R-multiple of risk_distance,
# genuinely a different, still-relevant metric), profit protection's
# whole premise - protecting a potentially-large, slow-to-resolve TP1 -
# doesn't fit a small, fast, fixed target the same way; letting it run to
# the real TP1 or the original SL is a tighter, faster resolution either
# way. Scoped to the pre-TP1 leg only (_is_profit_protection_candidate,
# TP1_PENDING/DCA_PENDING) - the post-TP1 remainder still running toward
# a real, structure-based TP2 is untouched by THIS flag, since TP2 is
# never static and keeps the same "can be far away, worth protecting
# early" justification as before (see PROFIT_PROTECTION_TP2_LEG_ENABLED
# below for that leg's own, separate protection). Evidence is thin (5
# pre-TP1 trades matching this pattern, no counterfactual showing what
# would have happened without it) but consistent and directly requested -
# default ON.
PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED = env_bool(
    "PROFIT_PROTECTION_SKIP_FOR_STATIC_TP1_ENABLED", "True"
)
# Real gap found live (2026-08-23, operator observation): once a genuine
# TP1 fill promotes the remainder to BREAKEVEN_ACTIVE, profit protection
# has never had a fresh-arm path for that leg - only a position that
# ALREADY armed protection pre-TP1 keeps trailing it
# (_trail_profit_protection_if_improved's own `if position.get(
# "profit_protection_applied")` guard). A position that reached
# BREAKEVEN_ACTIVE via a genuine TP1 fill starts with that flag False and
# stays that way for the rest of its life - only STRUCTURE_STOP_
# MANAGEMENT_ENABLED's trailing stop protects it, and only once a NEW
# confirmed swing has formed (can lag hours, see that setting's own
# comment). TP2 is always real, structure-resolved - never a small fixed
# target - so it fits this mechanism's own "can be far away, worth
# protecting early" premise exactly. Reuses PROFIT_PROTECTION_ACTIVATION_
# PCT_OF_TP1/_HIGH_ROI/LOCK_PCT_OF_TP1/RETRACE_PCT unchanged (same "% of
# whichever target actually applies" math the DCA_ACTIVE case already
# established - the "TP1" in those names is legacy from before that
# reuse existed). Requires PROFIT_PROTECTION_ENABLED also on. Brand new,
# zero live track record of its own yet - default OFF.
PROFIT_PROTECTION_TP2_LEG_ENABLED = env_bool("PROFIT_PROTECTION_TP2_LEG_ENABLED", "False")
# Replaces the fixed EARLY_BREAKEVEN_LOCK_R_MULTIPLE distance with the
# most recent CONFIRMED swing point in the trade's favor
# (market_structure.structure_state's last_swing_low/last_swing_high),
# clamped so it can never sit worse than flat breakeven - and, once a
# position is BREAKEVEN_ACTIVE (post-genuine-TP1 or post-early-lock),
# additionally trails the stop to that same structure level on every
# poll, ratchet-only (never loosens; TP2 stays a fixed target, only SL
# ever moves). Falls back to the existing fixed-distance calculation
# whenever no confirmed swing is available yet. Real motivation
# (2026-08-13, operator feedback): the fixed 0.3R lock was getting
# clipped by ordinary pullback noise on trades that then continued on to
# a real TP1/TP2 win. Trade-off, explicitly accepted: this REPLACES the
# old guarantee ("at least EARLY_BREAKEVEN_LOCK_R_MULTIPLE locked") with
# a weaker one ("never worse than breakeven scratch") - NOT proven more
# profitable yet, ship-and-measure same as every other feature here (see
# journal_analysis.py's TRAILING_STOP_PROFIT_HIT outcome). Real caveat:
# SWING_LEFT/SWING_RIGHT=4 on the 1h LTF means a swing needs ~4
# confirming candles after it forms - many trades resolve before any NEW
# swing has formed since entry, so in practice this often just falls
# back to plain MOVE_SL_TO_BREAKEVEN_AFTER_TP1 behavior; the real
# improvement mainly shows up on longer-running trades. Default OFF.
STRUCTURE_STOP_MANAGEMENT_ENABLED = env_bool("STRUCTURE_STOP_MANAGEMENT_ENABLED", "False")
# MAE/MFE (max adverse/favorable excursion) tracking - the diagnostic
# that's actually missing right now. A plain WIN/LOSS outcome can't tell
# apart a trade that was wrong from the first tick (near-zero MFE, went
# straight to the stop) from one that moved solidly in its favor and
# still reversed all the way back to a loss (large MFE) - those need
# completely different fixes (rework entry timing vs. tighten profit-
# taking/trailing). Tracks the worst/best price seen over a trade's life
# and journals both as an R-multiple of the original risk distance, so
# they're comparable across symbols/volatility regimes. Purely
# observational - never gates or sizes anything.
MAE_TRACKING_ENABLED = env_bool("MAE_TRACKING_ENABLED", "True")

# =========================
# DCA (single average-in, no SL before it fires)
# =========================
# binance_ai_bot_ds's other founding difference from binance_ai_bot_smc:
# the initial entry places TP1/TP2 as usual but NO stop loss. Whichever
# happens first wins the race - TP1 filling (the position graduates into
# the normal BREAKEVEN_ACTIVE lifecycle, a real SL finally gets placed via
# the existing breakeven-promotion path, DCA never fires) or price
# reaching the next real structure level beyond entry in the adverse
# direction (a single DCA fires: adds to the position, cancels TP1+TP2,
# places ONE new single-target TP plus the first real SL, both computed
# from the new blended entry price). See position_manager.py's
# DCA_PENDING/DCA_ACTIVE stages and risk_manager.py's compute_dca_target.
#
# Real, accepted risk (not something this code tries to design away): the
# window between entry and TP1-or-DCA is protected only by the bot's own
# poll loop (POSITION_POLL_INTERVAL_SECONDS), not a resting exchange
# order - a gap or violent single-candle move through both levels before
# the next poll tick has no circuit breaker, and in cross margin that
# exposure isn't contained to this one position. Keep EXECUTION_MODE on
# SHADOW (see below) until this has real track record.
DCA_ENABLED = env_bool("DCA_ENABLED", "True")
# Additional position size at DCA, as a multiplier of the ORIGINAL entry's
# quantity - 1.0 (default) is classic equal-size DCA, landing the blended
# entry exactly halfway between the two fills.
DCA_SIZE_MULTIPLIER = env_float("DCA_SIZE_MULTIPLIER", 1.0)
# The single post-DCA TP's minimum/maximum room, in R-multiples of the
# NEW risk distance (blended entry to the new post-DCA SL).
#
# Real bug found live (2026-08-17, operator observation - "new TP feels
# too high once DCA happened", confirmed against 3 real DCA fires):
# these originally mirrored TP2_R_MULTIPLE/TP2_MAX_R_MULTIPLE's raw
# numbers (4.0/10.0), on the reasoning that a post-DCA position "needs at
# least as much room as the ordinary second target already required." That
# reasoning missed that the DENOMINATOR changed too - the post-DCA risk
# distance (blended entry to the structure-anchored stop beyond the DCA
# fill) isn't the same size as the original entry's risk distance, it's
# structurally wider, since it already spans both the entry-to-DCA leg AND
# the DCA-fill-to-new-SL leg. Measured directly across the first 3 real
# DCA fires: post-DCA risk distance came out ~3.5-4.4x (avg ~3.8x) the
# original pre-DCA risk distance every time. Reusing TP2's raw R-multiple
# against that already-3.8x-wider risk therefore put the target ~3.8x
# further in absolute price terms than TP2 itself would ever have been -
# not "at least as much room as TP2", several multiples beyond it.
#
# Fixed by dividing TP2's original multiples by that same ~3.8x factor
# (4.0/3.8 and 10.0/3.8, rounded) - preserves the ORIGINAL intent (this
# target's absolute distance should be comparable to what TP2 would have
# been) instead of the accidental compounding. Still real-structure-first
# via _resolve_target/compute_dca_target - these are only the floor/cap a
# real liquidity pool gets clamped between, same as TP1/TP2 always were.
# Only 3 real data points behind the 3.8x factor - directionally
# consistent but not tightly calibrated; revisit once more DCA_TP_HIT/
# DCA_SL_HIT trades exist to check against real outcomes.
DCA_TP_R_MULTIPLE = env_float("DCA_TP_R_MULTIPLE", 1.0)
DCA_TP_MAX_R_MULTIPLE = env_float("DCA_TP_MAX_R_MULTIPLE", 2.5)
# Operator-requested alternative to the real-liquidity/R-multiple target
# above (2026-08-19, no evidence yet either way): a fixed ROI% target
# (at LEVERAGE) computed directly from the blended post-DCA entry price -
# "static" in that once DCA fires, this target depends only on the fill
# price and this one number, not structure/pools/risk_distance at all.
# See risk_manager.compute_dca_target/_price_at_roi_pct. Default False -
# same "new mechanism earns a live default only after real data" rule
# every other unvalidated mechanism this project ships follows.
DCA_TP_STATIC_ROI_ENABLED = env_bool("DCA_TP_STATIC_ROI_ENABLED", "False")
# Leveraged ROI%, same units as PROFIT_PROTECTION_HIGH_TP1_ROI_THRESHOLD_PCT
# elsewhere in this file - e.g. 50 means the position's unrealized ROI (at
# LEVERAGE) reaches 50% of margin. Starting value, not calibrated against
# real DCA_TP_HIT/DCA_SL_HIT outcomes yet.
DCA_TP_TARGET_ROI_PCT = env_float("DCA_TP_TARGET_ROI_PCT", 50)
# ATR buffer beyond the structure level used for the first-ever real SL,
# placed once DCA fires - mirrors STRUCTURE_STOP_ATR_BUFFER's own role for
# the (unused, pre-DCA) original stop calculation.
DCA_STRUCTURE_STOP_ATR_BUFFER = env_float("DCA_STRUCTURE_STOP_ATR_BUFFER", 0.5)
# Operator request (2026-08-20, motivated by a real trade - LITUSDT,
# 2026-08-19: DCA fired while price was still running hard in the
# adverse direction with zero sign of turning, then hit its post-DCA SL
# ~2 minutes later): "avoid triggering DCA until buy/sell pressure turns
# in the right direction... otherwise there's no point doing DCA."
#
# Real constraint this project's whole no-SL-before-DCA design imposes:
# DCA_PENDING has NO resting stop at all until DCA fires - delaying the
# fire itself would leave the position naked for LONGER, precisely while
# price is already past the level that currently triggers protection.
# That's a worse trade than the one being complained about, not a
# better one - so this does NOT delay when DCA fires. DCA still executes
# the instant dca_price is touched, exactly as before; the first real SL
# still arrives on schedule every time.
#
# Instead this checks signal_engine.direction_still_confirmed (the same
# HTF-trend/CVD/efficiency-ratio primitive DCA_BREAKEVEN_CONFIRMATION_
# ENABLED already reuses, here against the position's OWN side) at the
# instant DCA fires, and uses the read to size the RESPONSE rather than
# the timing: confirmed (order flow still favors the original side
# despite the adverse price move - looks like a pullback, not a genuine
# reversal) keeps today's unchanged behavior (DCA_SIZE_MULTIPLIER,
# DCA_STRUCTURE_STOP_ATR_BUFFER). Not confirmed (order flow has turned
# against the original side too - the adverse move looks real, not
# noise) commits LESS size (DCA_PRESSURE_SIZE_MULTIPLIER) at a TIGHTER
# stop (DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER) instead - smaller loss if
# wrong again, same protection timing either way.
#
# Can only ever make a DCA fire MORE conservative than today, never less
# (the confirmed branch is identical to current behavior) - unlike
# DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED, this can't leave a
# position less protected than it already is. Still defaults False: new,
# untested logic touching real order sizing/placement, no DCA_TP_HIT/
# DCA_SL_HIT evidence yet on whether the confirmed/not-confirmed split
# actually discriminates real outcomes - same "earns a live default only
# after real data" rule as every other unvalidated mechanism here.
DCA_PRESSURE_CHECK_ENABLED = env_bool("DCA_PRESSURE_CHECK_ENABLED", "False")
# Applied instead of DCA_SIZE_MULTIPLIER for this one fire when the
# pressure check above comes back NOT confirmed. 0.5 = half the usual
# DCA size - a starting, uncalibrated value, not derived from evidence.
DCA_PRESSURE_SIZE_MULTIPLIER = env_float("DCA_PRESSURE_SIZE_MULTIPLIER", 0.5)
# Applied instead of DCA_STRUCTURE_STOP_ATR_BUFFER for this one fire
# under the same not-confirmed condition - a smaller buffer beyond the
# post-DCA structure level, same "less room, smaller loss if wrong
# again" reasoning as the size cut above. Starting value (half of
# DCA_STRUCTURE_STOP_ATR_BUFFER's own 0.5 default), not calibrated.
DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER = env_float("DCA_PRESSURE_TIGHT_STOP_ATR_BUFFER", 0.25)
# Real gap named by this project's own founding design (see DCA_ENABLED's
# comment above): a DCA_PENDING position's add-in is detected by the poll
# loop watching candle ranges, not a resting exchange order - a gap or
# violent single-candle move through dca_price before the next poll tick
# has no circuit breaker at all. Real incident this connects to
# (2026-08-22, see CRASH_DETECTOR_ENABLED's comment): a DCA fired
# correctly but the SUBSEQUENT SL placement was REJECTED ("would
# immediately trigger") because price had already blown through the new
# level by the time the poll-driven code got to it.
#
# When on: the DCA add itself is placed as a real resting LIMIT order at
# dca_price the moment a position enters DCA_PENDING (see execution.
# place_dca_protection_orders), so Binance's own matching engine
# protects it continuously instead of once per
# POSITION_POLL_INTERVAL_SECONDS. poll_live then detects the FILL via
# exchange.get_order_status instead of watching candle ranges for the
# trigger - the candle-range check (position_manager.
# _dca_price_reached_in_range) stays as the fallback for any position
# without a resting order (flag off, or registered before it was turned
# on).
#
# Sizing (operator-confirmed, 2026-08-25): the resting order can't do
# DCA_PRESSURE_CHECK_ENABLED's real-time "last look" before filling -
# whatever size is resting fills unconditionally the instant price
# reaches it. Always rests at DCA_PRESSURE_SIZE_MULTIPLIER (the pressure
# check's existing "not confirmed" size) with DCA_PRESSURE_TIGHT_STOP_
# ATR_BUFFER (its existing tighter stop) - never the full DCA_SIZE_
# MULTIPLIER size, even when order flow would have confirmed. Consistent
# with the pressure check's own stated principle ("can only make DCA
# more conservative than today, never less") and with the real evidence
# that dca_applied=True trades already skew losing system-wide (~6.5%
# win rate, 2026-08-25 signal-engine audit) - the "full size when
# confirmed" upside is rarely realized anyway, so it isn't worth keeping
# at the cost of leaving the resting order at full size during a real
# gap. position_manager._execute_dca's own DCA_PRESSURE_CHECK_ENABLED
# block is skipped entirely when a resting-order fill is what triggered
# it (sizing already happened at placement time) - unaffected for the
# candle-range fallback path, which still runs the real-time check
# exactly as before.
#
# Default False: new, unvalidated mechanism touching real order
# placement - same "earns a live default only after real data" rule as
# DCA_PRESSURE_CHECK_ENABLED itself and every other unvalidated
# mechanism here.
DCA_RESTING_ORDER_ENABLED = env_bool("DCA_RESTING_ORDER_ENABLED", "False")
# Real gap found live (2026-08-17, operator observation): a DCA_ACTIVE
# position's only two protection mechanisms are PROFIT_PROTECTION_ENABLED
# (needs PROFIT_PROTECTION_ACTIVATION_PCT_OF_TP1 of the way to the single,
# wide post-DCA TP - a deep move) and STRUCTURE_STOP_MANAGEMENT_ENABLED
# (needs a NEW confirmed swing, which can lag hours). Between those, a
# trade that recovers all the way to breakeven and then reverses has
# nothing moving its stop - it rides back down to the original post-DCA
# SL (a real loss) with zero protection in between. This closes that gap
# the same way MOVE_SL_TO_BREAKEVEN_AFTER_TP1 already does for a genuine
# TP1 fill: the moment price reaches position["breakeven_price"], replace
# the SL there. One-time arm per position (see dca_breakeven_applied) -
# profit protection/structure trailing still run on top of it afterward.
# Default ON: same "turns a full loss into a scratch" reasoning already
# behind EARLY_BREAKEVEN_ENABLED's default in this project, just applied
# to the post-DCA stage instead of the pre-DCA one.
DCA_BREAKEVEN_ENABLED = env_bool("DCA_BREAKEVEN_ENABLED", "True")
# Real gap in the gap-closer above: DCA_BREAKEVEN_ENABLED moves the stop to
# breakeven unconditionally the instant price reaches it, even when the
# broader trend/order-flow picture still strongly favors the position -
# turning what could have run to the real (wide) post-DCA target into a
# guaranteed scratch. This makes that conditional: re-runs the trend/order-
# flow-health subset of evaluate()'s own entry gates (AGAINST_HTF_BIAS,
# HTF_TREND_STALE, CVD confirmation, MARKET_CHOPPY - see signal_engine.
# direction_still_confirmed for exactly what's checked and why the entry-
# timing-specific gates, zone/OTE/order block, are deliberately excluded)
# against the position's OWN side. Two separate flags, not one, for a
# deliberate two-phase rollout (2026-08-19, zero trades have ever used
# this): this flag alone only COMPUTES and JOURNALS the confirmation
# verdict (signal_journal.py's dca_breakeven_direction_confirmed) - the
# real breakeven move still happens exactly as it does today underneath,
# unconditionally. Only once DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED is
# ALSO on does a confirmed verdict actually withhold the move. Default
# False - unlike OI_RISING_REJECT_ENABLED (reject-only, safe to ship live
# immediately), this feature can WITHHOLD a protective SL move if wrong,
# which increases risk rather than reduces it - it earns a live default
# only after real evidence, same as every other unvalidated mechanism.
DCA_BREAKEVEN_CONFIRMATION_ENABLED = env_bool("DCA_BREAKEVEN_CONFIRMATION_ENABLED", "False")
# The actual behavior-changing switch - see DCA_BREAKEVEN_CONFIRMATION_
# ENABLED's comment above. Requires that flag ALSO on to do anything;
# left independently toggleable so the informational phase can run for as
# long as needed without this ever being touched. Fails safe on its own
# terms too: while price sits at/above breakeven the check re-runs every
# poll tick (not just once) - the instant confirmation fails, the stop
# moves to breakeven immediately, same protection as today, never later.
DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED = env_bool(
    "DCA_BREAKEVEN_CONFIRMATION_WITHHOLD_ENABLED", "False"
)
# Real gap in DCA_BREAKEVEN_ENABLED's own mechanism, not what it protects
# against: it's a poll-and-replace design (detect price reached breakeven,
# cancel the old SL, place a new one at breakeven) - two real sources of
# delay (POSITION_POLL_INTERVAL_SECONDS=10s detection lag, then the
# cancel/place round trip itself). Real incident (2026-08-26, MEUSDT SELL):
# price recovered to breakeven, the bot detected it and tried to move the
# SL there, but by the time the replacement order reached the exchange
# price had already reversed back past breakeven again - Binance rejected
# it with -2021 ("would immediately trigger"), and _replace_sl_order's
# existing fallback closed the position at market anyway. grep of the VPS
# log confirms this isn't a one-off: 5 DCA-breakeven races, 2 TP1-breakeven
# races, 1 profit-protection race, all within an 8-day window (see
# _dca_breakeven_price_reached_in_range's own docstring, which already
# named this exact race before this flag existed).
#
# When on: the moment DCA fires, a real Binance TRAILING_STOP_MARKET order
# is placed alongside the existing wide hard SL, activatePrice=breakeven_
# price, resting dormant. Binance itself arms it server-side the instant
# price reaches breakeven and trails it from there - zero bot polling,
# zero cancel/replace round trip, so the specific MEUSDT race can't repeat
# for a position on this path. It also improves on the OLD mechanism's
# flat stop: once armed, it keeps tightening as price extends further
# favorably instead of sitting flat at breakeven waiting for
# PROFIT_PROTECTION_ENABLED's much deeper threshold or STRUCTURE_STOP_
# MANAGEMENT_ENABLED's next confirmed swing to catch up.
#
# Real incident (2026-08-26, MEUSDT, the feature's first-ever live
# attempt): the ORIGINAL design used closePosition=true for the trailing
# stop (mirroring place_stop_loss's own shape) - Binance rejected every
# attempt outright with -4136 ("Target strategy invalid for orderType
# TRAILING_STOP_MARKET,closePosition true"). Confirmed against a real
# Binance dev-forum report of the identical error and its working fix:
# TRAILING_STOP_MARKET doesn't support closePosition=true at all, full
# stop - not a coexistence question. Now fixed (exchange.place_trailing_
# stop_loss) to use reduceOnly=true with an explicit quantity instead,
# the same shape place_take_profit_partial already uses successfully.
#
# Still real, unverified: whether a closePosition=true STOP_MARKET (the
# existing wide post-DCA SL) and this reduceOnly TRAILING_STOP_MARKET can
# rest on the same symbol/side simultaneously - this project has only
# proven DIFFERENT-type closePosition orders coexist (SL + TP - see
# place_stop_loss/place_take_profit_full) and that same-type closePosition
# duplicates are REJECTED (-4130, a real 2026-08-08 incident - see
# execution.py's own -4130 handling); a reduceOnly+quantity order is a
# different shape from either of those, still untested here. The
# placement is deliberately best-effort/non-fatal (same treatment as
# TP1/TP2 placement failures in _execute_dca already) - a rejection just
# means the position keeps its existing wide SL exactly as today, nothing
# else breaks. DCA_BREAKEVEN_ENABLED's own poll-based mechanism stays
# fully intact as the fallback for whenever this either isn't enabled or
# its placement failed for this specific position (see _is_dca_breakeven_
# candidate). Default False: new, unvalidated mechanism touching real
# order placement - same "earns a live default only after real data" rule
# as every other unvalidated mechanism here.
DCA_BREAKEVEN_TRAILING_STOP_ENABLED = env_bool("DCA_BREAKEVEN_TRAILING_STOP_ENABLED", "False")
# Binance's real minimum is 0.1 (python-binance's own futures_create_algo_
# order docstring: "min 0.1, max 10" for callbackRate). 0.2 is a small
# safety margin above that floor, not outcome-validated - a starting
# value, same convention as RETRACEMENT_ENTRY_OFFSET_R etc. Deliberately
# capped well below Binance's own max of 10 by _normalize_callback_rate
# (exchange.py) - a callback anywhere near that wide would give back far
# more profit than the flat breakeven stop it's replacing was ever
# designed to risk.
DCA_BREAKEVEN_TRAILING_CALLBACK_RATE = env_float("DCA_BREAKEVEN_TRAILING_CALLBACK_RATE", 0.2)

# =========================
# EXECUTION
# =========================
# SHADOW: evaluate signals, size them, log to the journal - place no real
# orders. LIVE: actually enter/attach TP1/TP2/SL. Defaults to SHADOW so
# running this bot for the first time cannot place a real order by
# accident - flip explicitly once shadow signal quality has been reviewed.
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "SHADOW").strip().upper()
POSITION_POLL_INTERVAL_SECONDS = env_int("POSITION_POLL_INTERVAL_SECONDS", 10)
SIGNAL_EVAL_INTERVAL_SECONDS = env_int("SIGNAL_EVAL_INTERVAL_SECONDS", 5)
# A signal must keep qualifying for this many consecutive eval ticks
# before it's acted on - not just the single instant it first appears.
# Real motivation (2026-08-12, live): IOTXUSDT was rejected for
# CVD_NOT_CONFIRMED, then passed 16 seconds later on a marginal 0.29
# score, then sat flat for 90+ minutes before losing - CVD is computed
# over 1m/5m/15m windows (order_flow.py), so it can flip pass/fail within
# seconds, meaning a single-instant pass can be noise rather than genuine
# sustained order flow. Structure/OTE/HTF/order-block/FVG are all derived
# from the last CLOSED candle (see REQUIRE_CLOSE_CONFIRMED_BREAK), so they
# don't change tick-to-tick - CVD and depth imbalance are the only
# genuinely volatile inputs, so requiring the full signal to hold for a
# few ticks in a row is effectively a CVD/depth stability filter. 1
# disables it (act on the first qualifying tick, original behavior).
SIGNAL_CONFIRM_TICKS = env_int("SIGNAL_CONFIRM_TICKS", 3)
# Used only in SHADOW mode so risk-based position sizing has a balance to
# size against without needing real API keys / an authenticated call.
SHADOW_ACCOUNT_BALANCE_USDT = env_float("SHADOW_ACCOUNT_BALANCE_USDT", 1000)
# Enables per-signal market-vs-limit ROUTING (main.py), not "always place
# a limit order" - a signal whose entry_extension_r (risk_manager.build_trade_plan,
# how far price already ran from the structure level, in R) is at or
# below ENTRY_ROUTING_EXTENSION_THRESHOLD_R still gets a market order
# (price is close enough to the ideal level that chase cost is minimal,
# so a guaranteed fill beats limit fill-uncertainty); only a signal
# that's already moderately extended (above the threshold, but still
# under the hard MAX_ENTRY_EXTENSION_R reject) gets routed to a resting
# GTC LIMIT at entry_price instead, so it either fills at (or better
# than) the real level or is walked away from - never chased further by
# a market order. This keeps trade count close to the market-only
# baseline (most signals aren't extended enough to route to a limit) while
# still fixing the late-chase problem for the subset that is. Deliberately
# has NO market-order fallback on a limit's expiry/invalidation (see
# position_manager.poll_pending_entry) - that's the entire point of
# routing those specific entries away from market in the first place.
# Default OFF, same "don't silently change a running bot's behavior"
# convention as every other feature this session.
LIMIT_ENTRY_MODE_ENABLED = env_bool("LIMIT_ENTRY_MODE_ENABLED", "False")
# Must be less than MAX_ENTRY_EXTENSION_R for the "route to limit" band
# to be non-empty (below this: market; between this and
# MAX_ENTRY_EXTENSION_R: limit; above MAX_ENTRY_EXTENSION_R: rejected
# outright, unchanged). Starting value, not yet calibrated against real
# fill-rate/outcome data.
ENTRY_ROUTING_EXTENSION_THRESHOLD_R = env_float("ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2)
# Wall-clock, not tick-based - deliberately decoupled from
# SIGNAL_EVAL_INTERVAL_SECONDS (see poll_every_ticks in main.py; tying
# expiry to eval-ticks would make it silently rescale if that's ever
# tuned, the same hidden-coupling bug already found once with
# _current_balance()). 600s chosen against WS_KLINE_INTERVAL=1h /
# HTF_KLINE_INTERVAL=4h - roughly 1/6 of the triggering hourly candle,
# long enough for a genuine OTE retracement without the setup going
# stale mid-candle. Starting value, not yet calibrated against real
# fill-rate data.
LIMIT_ENTRY_EXPIRY_SECONDS = env_int("LIMIT_ENTRY_EXPIRY_SECONDS", 600)
# Real evidence (2026-08-20, 24 resolved trades cross-checked against
# real 1-minute Binance price data around each entry): 75% dipped at
# least 0.1R against the position within 5 minutes of entry, regardless
# of trigger type (even retest-style triggers, not just breakouts) -
# entries fire at market the instant a trigger condition is detected,
# which is disproportionately often a local extreme that immediately
# mean-reverts before the real move starts. Most of these trades still
# won once they recovered - the direction was right, the entry price/
# timing wasn't.
#
# Unlike LIMIT_ENTRY_MODE_ENABLED above (which routes only already-
# extended entries, and deliberately has no market fallback - walking
# away is the point there), this applies to EVERY entry regardless of
# entry_extension_r and NEVER skips a signal: places a resting limit at
# a small pullback toward the stop (risk_manager.compute_retracement_
# price) instead of paying the trigger-instant price, but always falls
# back to a market order for whatever didn't fill once RETRACEMENT_
# ENTRY_TIMEOUT_SECONDS elapses (see position_manager.RETRACEMENT_PENDING/
# _finalize_retracement_entry) - so every signal still results in a
# position exactly as today, just later and (on the ~75% of trades that
# dip) at a real, better price. Takes priority over both DCA_ENABLED's
# and LIMIT_ENTRY_MODE_ENABLED's own routing when on (see main.py) -
# after the fill/fallback resolves, it hands off into DCA_PENDING or
# TP1_PENDING exactly like a direct entry would have.
#
# A resting, unfilled retracement limit has no matching real exchange
# position, so it can't be recovered by reconcile_on_startup (which only
# walks REAL open positions) - position_manager.reconcile_pending_entries_
# on_startup covers this instead (the same account-wide "cancel any
# stray resting LIMIT order, re-evaluate fresh" sweep LIMIT_ENTRY_MODE_
# ENABLED already relies on), so a restart mid-window loses at most that
# one pending signal, never leaves an order truly orphaned.
#
# Default False: the entry-timing PROBLEM is evidence-backed, but this
# specific FIX has zero live track record of its own yet - same "earns a
# live default only after real data on the mechanism itself" rule as
# every other unvalidated mechanism here.
RETRACEMENT_ENTRY_ENABLED = env_bool("RETRACEMENT_ENTRY_ENABLED", "False")
# In units of the planned risk distance (entry to sl_price) - 0.1 means
# a limit resting 10% of the way from the trigger price toward the stop.
# Chosen directly off the measured data: 75% of trades already reach at
# least this much adverse excursion within 5 minutes, so a fill is
# likely without resting deep enough to risk missing a genuine
# continuation entirely. Starting value, not yet calibrated against real
# RETRACEMENT-mechanism fill-rate/outcome data of its own.
RETRACEMENT_ENTRY_OFFSET_R = env_float("RETRACEMENT_ENTRY_OFFSET_R", 0.1)
# How long the resting limit waits before falling back to market for
# whatever didn't fill. Short by design (minutes, not
# LIMIT_ENTRY_EXPIRY_SECONDS' 600s) - the measured adverse dip already
# happens fast (average worst point ~16 minutes in, but most of the
# first move happens within 5) and this mechanism's whole point is
# capturing that early pullback, not waiting out a slower one. Starting
# value, not yet calibrated.
RETRACEMENT_ENTRY_TIMEOUT_SECONDS = env_int("RETRACEMENT_ENTRY_TIMEOUT_SECONDS", 300)
# RETRACEMENT_ENTRY_OFFSET_R above is a pure calculation, not tied to any
# real market structure. Investigated (2026-08-22, 21 real retracement
# signals): a real structural level (FVG edge, liquidity pool) sat
# strictly between entry and SL for 14/21 (67%) of them, but its own
# distance varied wildly (0.024R to 0.717R, avg 0.197R) - not clustered
# near RETRACEMENT_ENTRY_OFFSET_R's own value. When on, risk_manager.
# compute_retracement_price prefers the nearest real level within
# RETRACEMENT_STRUCTURE_MAX_R of entry over the fixed-R calculation;
# falls back to it otherwise (flag off, no real level present, or the
# nearest one is deeper than the cap allows). Default OFF: unlike the
# other fixes in this file, this one is explicitly exploratory - the
# investigation itself found real levels present only 2/3 of the time
# with no outcome evidence either approach wins - same "earns a live
# default only after real data on the mechanism itself" rule as every
# other unvalidated mechanism here.
RETRACEMENT_STRUCTURE_TARGET_ENABLED = env_bool("RETRACEMENT_STRUCTURE_TARGET_ENABLED", "False")
# Real investigation data: the "similar to RETRACEMENT_ENTRY_OFFSET_R's
# own default" band was 0.05-0.30R (10/21 signals). 0.35 gives a little
# headroom above that without accepting the 0.717R outlier a real level
# occasionally produces - a real level deeper than this is treated as not
# "sane" and the fixed-R fallback is used instead. Starting value, not
# outcome-validated - revisit once this has run live.
RETRACEMENT_STRUCTURE_MAX_R = env_float("RETRACEMENT_STRUCTURE_MAX_R", 0.35)

# =========================
# LOGGING / ALERTING
# =========================
TELEGRAM_ENABLED = env_bool("TELEGRAM_ENABLED", "False")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_TIMEOUT_SECONDS = env_float("TELEGRAM_TIMEOUT_SECONDS", 10)
