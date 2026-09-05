import unittest
from collections import Counter
from unittest.mock import MagicMock, patch

import pandas as pd

import config
import exchange
import execution
import main
import market_structure
import risk_manager
import signal_engine
import signal_journal


class _FakeSnapshotSource:
    def snapshot(self, symbol):
        return {"available": False}


class _FakeCrashDetector:
    def snapshot(self, now=None):
        return {"available": False, "active": False, "direction": None, "pct_move": 0.0}


class _FakeCandleSource:
    def __init__(self, candles=None):
        self._candles = candles if candles is not None else [{"open_time": 0, "close": 1}]

    def get(self, symbol):
        return self._candles

    def latest(self, symbol):
        return self._candles[-1] if self._candles else None


class _FakeFeed:
    def __init__(self, ltf_candles=None, htf_candles=None, volumes=None, funding_rates=None):
        self.candles = _FakeCandleSource(ltf_candles)
        self.htf_candles = _FakeCandleSource(htf_candles)
        # config.EMA_TREND_MIXED_REJECT_ENABLED - the deeper EMA50/200
        # buffers. Mirroring the same candles is fine here: signal_engine
        # falls back to the structure buffers when these are empty, and
        # _ema_regime returns None either way in these fixtures, so the gate
        # stays inert exactly as it does with the flag off.
        self.trend_candles = _FakeCandleSource(ltf_candles)
        self.htf_trend_candles = _FakeCandleSource(htf_candles)
        self.cvd = _FakeSnapshotSource()
        self.depth = _FakeSnapshotSource()
        self.open_interest = _FakeSnapshotSource()
        self.open_interest_bybit = _FakeSnapshotSource()
        self.open_interest_okx = _FakeSnapshotSource()
        self.volume_profile = _FakeSnapshotSource()
        self.liquidations = _FakeSnapshotSource()
        self.liquidations_bybit = _FakeSnapshotSource()
        self.liquidations_okx = _FakeSnapshotSource()
        self.crash_detector = _FakeCrashDetector()
        self.volumes = volumes if volumes is not None else {}
        self.funding_rates = funding_rates if funding_rates is not None else {}


class _FakePositions:
    def __init__(self, has_open=False, in_cooldown=False, count=0, shadow_count=0):
        self._has_open = has_open
        self._in_cooldown = in_cooldown
        self._count = count
        self._shadow_count = shadow_count
        self.registered = []
        self.registered_pending = []
        self.registered_dca_pending = []
        self.registered_retracement_pending = []
        self.positions = {}

    def has_open_position(self, symbol):
        return self._has_open

    def is_in_cooldown(self, symbol):
        return self._in_cooldown

    def open_count(self):
        return self._count

    def real_open_count(self):
        return self._count

    def shadow_open_count(self):
        return self._shadow_count

    def mark_entry_failure(self, symbol):
        pass

    def register(self, plan, execution_result, trade_id=None):
        self.registered.append((plan, execution_result, trade_id))

    def register_pending_entry(self, plan, execution_result, trade_id=None):
        self.registered_pending.append((plan, execution_result, trade_id))

    def register_dca_pending(self, plan, execution_result, trade_id=None):
        self.registered_dca_pending.append((plan, execution_result, trade_id))

    def register_retracement_pending(self, plan, execution_result, trade_id=None):
        self.registered_retracement_pending.append((plan, execution_result, trade_id))


class EvaluateSymbolRejectCountsTests(unittest.TestCase):
    """Real gap found live (2026-08-11): every rejection reason from
    signal_engine.evaluate() was silently discarded, so "no entries" was
    unexplainable from the logs alone - no way to tell "genuinely no
    qualifying setups yet" apart from "something is over-restrictive".
    These lock in that reject_counts actually captures the reason."""

    def setUp(self):
        # config.DCA_ENABLED defaults True in this project and takes
        # priority over the plain market/limit entry path these tests
        # exercise (see main.py's routing comment) - their own minimal
        # `plan` fixtures predate the DCA fields that path now requires.
        # Not what these tests are about, so pinned off here.
        # config.RETRACEMENT_ENTRY_ENABLED - takes priority over EVERYTHING
        # else in that same routing (see main.py's use_retracement) - a
        # real .env flip to True (the operator's own live setting) routes
        # these same minimal fixtures into enter_trade_retracement instead,
        # which needs plan["side"] these fixtures don't carry. Pinned off
        # for the same isolation reason as DCA_ENABLED above.
        # CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED/CVD_DIVERGENCE_PRICE_
        # HOLD_WEAK_REJECT_ENABLED pinned off too: test_signal_trigger_is_
        # copied_onto_plan_before_execution_and_journaling below uses a
        # real CVD_DIVERGENCE signal_trigger and falls through past
        # build_trade_plan - not what that test is about, and these
        # gates' on-demand exchange calls aren't mocked there.
        for name, value in (
            ("DCA_ENABLED", False), ("RETRACEMENT_ENTRY_ENABLED", False),
            ("CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False),
            ("CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_candle_data_is_tallied(self):
        feed = _FakeFeed(ltf_candles=[], htf_candles=[])
        positions = _FakePositions()
        reject_counts = Counter()

        main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(reject_counts["NO_CANDLE_DATA"], 1)

    def test_signal_engine_rejection_reason_is_tallied(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "NOT_IN_OTE"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(reject_counts["NOT_IN_OTE"], 1)

    def test_missing_reason_falls_back_to_unknown(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()

        with patch.object(signal_engine, "evaluate", return_value={"signal": None}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(reject_counts["UNKNOWN"], 1)

    def test_plan_rejection_is_tallied_with_its_status(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()

        with patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(reject_counts["PLAN_REJECTED:SL_TOO_TIGHT"], 1)

    def test_accepted_signal_does_not_touch_reject_counts(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        plan = {
            "symbol": "BTCUSDT", "entry_price": 100, "sl_price": 98,
            "tp1_price": 102, "tp2_price": 104,
        }

        with patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(plan, "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}), \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(len(reject_counts), 0)
        self.assertEqual(len(positions.registered), 1)

    def test_signal_trigger_is_copied_onto_plan_before_execution_and_journaling(self):
        # config.SHADOW_ONLY_TRIGGERS - execution._is_shadow_mode reads
        # plan["signal_trigger"], not result["signal_trigger"] - this
        # must be populated before execution.enter_trade is called, same
        # precedent as the existing structure_level/trigger_candle_open_time
        # copy-through.
        feed = _FakeFeed()
        positions = _FakePositions()
        plan = {
            "symbol": "BTCUSDT", "entry_price": 100, "sl_price": 98,
            "tp1_price": 102, "tp2_price": 104,
        }
        execution_result = {"ok": True, "shadow": True}
        result = {"signal": "BUY", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "LONG_SHORT_RATIO_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(risk_manager, "build_trade_plan", return_value=(plan, "OK")), \
             patch.object(execution, "enter_trade", return_value=execution_result) as enter_trade, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123") as append_signal:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        entered_plan = enter_trade.call_args.args[0]
        self.assertEqual(entered_plan["signal_trigger"], "CVD_DIVERGENCE")
        append_signal.assert_called_once()
        journaled_result, journaled_plan, journaled_execution_result = append_signal.call_args.args
        self.assertIs(journaled_plan, plan)
        self.assertEqual(journaled_plan["signal_trigger"], "CVD_DIVERGENCE")
        self.assertIs(journaled_execution_result, execution_result)

    def test_quote_volume_is_passed_through_to_signal_engine(self):
        feed = _FakeFeed(volumes={"BTCUSDT": 42_000_000})
        positions = _FakePositions()

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "X"}) as evaluate_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        _, kwargs = evaluate_mock.call_args
        self.assertEqual(kwargs["quote_volume_usdt"], 42_000_000)

    def test_btc_candles_and_funding_rate_are_passed_through_to_signal_engine(self):
        btc_candles = [{"open_time": 0, "close": 100}]
        feed = _FakeFeed(funding_rates={"BTCUSDT": 0.0002})
        feed.candles = _FakeCandleSource([{"open_time": 0, "close": 1}])
        positions = _FakePositions()

        with patch.object(config, "CORRELATION_REFERENCE_SYMBOL", "REFUSDT"), \
             patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "X"}) as evaluate_mock:
            feed.candles.get = lambda symbol: (
                btc_candles if symbol == "REFUSDT" else [{"open_time": 0, "close": 1}]
            )
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        _, kwargs = evaluate_mock.call_args
        self.assertEqual(kwargs["btc_candles"], btc_candles)
        self.assertEqual(kwargs["funding_rate"], 0.0002)

    def test_missing_volume_data_is_passed_through_as_none(self):
        feed = _FakeFeed(volumes={})
        positions = _FakePositions()

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "X"}) as evaluate_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        _, kwargs = evaluate_mock.call_args
        self.assertIsNone(kwargs["quote_volume_usdt"])

    def test_reject_counts_none_is_safe_and_does_not_raise(self):
        feed = _FakeFeed(ltf_candles=[], htf_candles=[])
        positions = _FakePositions()

        main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts=None)

    def test_reject_symbols_records_which_symbol_triggered_the_reason(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        reject_symbols = {}

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "NOT_IN_OTE"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols)

        self.assertEqual(reject_symbols["NOT_IN_OTE"], ["BTCUSDT"])

    def test_reject_symbols_sample_is_capped_but_the_count_keeps_growing(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        reject_symbols = {}
        symbols = [f"SYM{i}USDT" for i in range(8)]

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "NOT_IN_OTE"}):
            for symbol in symbols:
                main._evaluate_symbol(feed, symbol, positions, 1000, reject_counts, reject_symbols)

        self.assertEqual(reject_counts["NOT_IN_OTE"], 8)
        self.assertEqual(len(reject_symbols["NOT_IN_OTE"]), main._MAX_REJECT_SAMPLE_SYMBOLS)

    def test_same_symbol_rejected_twice_is_not_duplicated_in_the_sample(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        reject_symbols = {}

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "NOT_IN_OTE"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols)
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols)

        self.assertEqual(reject_counts["NOT_IN_OTE"], 2)
        self.assertEqual(reject_symbols["NOT_IN_OTE"], ["BTCUSDT"])

    def test_reject_symbols_none_is_safe_and_does_not_raise(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "NOT_IN_OTE"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), reject_symbols=None)

    def test_operational_skips_are_not_tallied(self):
        # Routine skips (already has a position, cooldown, at capacity)
        # aren't signal-quality rejections - tallying them would dilute
        # the reason breakdown with noise that's expected, not actionable.
        feed = _FakeFeed()
        reject_counts = Counter()

        main._evaluate_symbol(feed, "BTCUSDT", _FakePositions(has_open=True), 1000, reject_counts)
        main._evaluate_symbol(feed, "BTCUSDT", _FakePositions(in_cooldown=True), 1000, reject_counts)
        main._evaluate_symbol(feed, "BTCUSDT", _FakePositions(count=999), 1000, reject_counts)

        self.assertEqual(len(reject_counts), 0)

    def test_shadow_positions_do_not_count_toward_max_total_positions_capacity(self):
        # config.SHADOW_ONLY_TRIGGERS - a shadow trade never touches the
        # exchange, so it must not compete with real trades for real
        # capacity. ltf_candles=[] forces NO_CANDLE_DATA once past the
        # capacity gate - reaching it proves evaluation wasn't blocked
        # there, since a block is a silent operational skip (see
        # test_operational_skips_are_not_tallied above).
        feed = _FakeFeed(ltf_candles=[])
        reject_counts = Counter()
        positions = _FakePositions(count=1, shadow_count=5)

        with patch.object(config, "MAX_TOTAL_POSITIONS", 2):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(reject_counts.get("NO_CANDLE_DATA"), 1)

    def test_real_positions_alone_still_hit_the_capacity_cap(self):
        feed = _FakeFeed(ltf_candles=[])
        reject_counts = Counter()
        positions = _FakePositions(count=2, shadow_count=0)

        with patch.object(config, "MAX_TOTAL_POSITIONS", 2):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts)

        self.assertEqual(len(reject_counts), 0)

    def test_triggers_field_is_tallied_separately_when_present(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        reject_trigger_counts = Counter()

        with patch.object(
            signal_engine, "evaluate",
            return_value={"signal": None, "reason": "AGAINST_HTF_BIAS", "triggers": ["STRUCTURE_BREAK"]},
        ):
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, reject_counts, None, None,
                reject_trigger_counts,
            )

        self.assertEqual(reject_counts["AGAINST_HTF_BIAS"], 1)
        self.assertEqual(reject_trigger_counts["AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK"], 1)

    def test_multiple_triggers_are_joined_in_the_trigger_tally_key(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_trigger_counts = Counter()

        with patch.object(
            signal_engine, "evaluate",
            return_value={
                "signal": None, "reason": "AGAINST_HTF_BIAS",
                "triggers": ["LIQUIDITY_SWEEP", "STRUCTURE_BREAK"],
            },
        ):
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, Counter(), None, None,
                reject_trigger_counts,
            )

        self.assertEqual(
            reject_trigger_counts["AGAINST_HTF_BIAS | triggers=LIQUIDITY_SWEEP,STRUCTURE_BREAK"], 1
        )

    def test_missing_triggers_field_does_not_touch_the_trigger_tally(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_trigger_counts = Counter()

        with patch.object(
            signal_engine, "evaluate", return_value={"signal": None, "reason": "NO_LIVE_STRUCTURE_BREAK"}
        ):
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, Counter(), None, None,
                reject_trigger_counts,
            )

        self.assertEqual(len(reject_trigger_counts), 0)

    def test_reject_trigger_counts_none_is_safe_and_does_not_raise(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(
            signal_engine, "evaluate",
            return_value={"signal": None, "reason": "AGAINST_HTF_BIAS", "triggers": ["STRUCTURE_BREAK"]},
        ):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

    def test_trigger_sample_symbols_are_tracked_when_provided(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_trigger_counts = Counter()
        reject_trigger_symbols = {}

        with patch.object(
            signal_engine, "evaluate",
            return_value={"signal": None, "reason": "AGAINST_HTF_BIAS", "triggers": ["STRUCTURE_BREAK"]},
        ):
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, Counter(), None, None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        self.assertEqual(
            reject_trigger_symbols["AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK"], ["BTCUSDT"]
        )


class SignalStabilityTrackerTests(unittest.TestCase):
    """Real motivation (2026-08-12, live): IOTXUSDT was rejected for
    CVD_NOT_CONFIRMED, then passed 16 seconds later on a marginal score,
    then sat flat for 90+ minutes before losing - CVD is computed over 1m/
    5m/15m windows, so it can flip pass/fail within seconds, meaning a
    single-instant pass can be noise rather than genuine sustained order
    flow. These lock in that a signal must hold for
    config.SIGNAL_CONFIRM_TICKS consecutive calls before it's confirmed."""

    def test_first_call_is_not_yet_confirmed(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 3):
            tracker = main.SignalStabilityTracker()
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY"))

    def test_confirmed_after_the_required_number_of_consecutive_calls(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 3):
            tracker = main.SignalStabilityTracker()
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY"))
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY"))

    def test_flipping_side_restarts_the_streak(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 3):
            tracker = main.SignalStabilityTracker()
            tracker.confirm("BTCUSDT", "BUY")
            tracker.confirm("BTCUSDT", "BUY")
            self.assertFalse(tracker.confirm("BTCUSDT", "SELL"))
            self.assertFalse(tracker.confirm("BTCUSDT", "SELL"))
            self.assertTrue(tracker.confirm("BTCUSDT", "SELL"))

    def test_reset_clears_the_streak(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2):
            tracker = main.SignalStabilityTracker()
            tracker.confirm("BTCUSDT", "BUY")
            tracker.reset("BTCUSDT")
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY"))

    def test_streaks_are_tracked_independently_per_symbol(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2):
            tracker = main.SignalStabilityTracker()
            tracker.confirm("BTCUSDT", "BUY")
            self.assertFalse(tracker.confirm("ETHUSDT", "BUY"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY"))

    def test_confirm_ticks_of_one_confirms_immediately(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 1):
            tracker = main.SignalStabilityTracker()
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY"))

    # trigger param - every trigger now carries some extra-ticks
    # requirement: config.STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS (small, the
    # one proven trigger) vs config.EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS
    # (larger, everything else) - except the back-compat untagged/None
    # case, for callers that don't pass trigger at all.

    def test_structure_break_trigger_requires_its_own_smaller_extra_ticks(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2), \
             patch.object(config, "STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS", 1), \
             patch.object(config, "EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS", 3):
            tracker = main.SignalStabilityTracker()
            # base(2) + structure_break_extra(1) = 3 required, NOT base(2)
            # + the larger new-trigger extra(3) = 5.
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))

    def test_structure_break_extra_ticks_of_zero_preserves_base_behavior(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2), \
             patch.object(config, "STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS", 0):
            tracker = main.SignalStabilityTracker()
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))

    def test_untagged_none_trigger_uses_base_ticks_only(self):
        # Back-compat: existing callers that never pass trigger (main.py's
        # test doubles, or any future caller not yet trigger-aware) must
        # not silently start requiring extra ticks.
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2), \
             patch.object(config, "STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS", 1), \
             patch.object(config, "EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS", 2):
            tracker = main.SignalStabilityTracker()
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY"))

    def test_non_structure_break_trigger_requires_more_extra_ticks_than_structure_break(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2), \
             patch.object(config, "STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS", 1), \
             patch.object(config, "EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS", 2):
            tracker = main.SignalStabilityTracker()
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "LIQUIDITY_SWEEP"))
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "LIQUIDITY_SWEEP"))
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "LIQUIDITY_SWEEP"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY", "LIQUIDITY_SWEEP"))

    def test_choch_retest_and_ob_fvg_retest_also_require_extra_ticks(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 1), \
             patch.object(config, "EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS", 1):
            tracker = main.SignalStabilityTracker()
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "CHOCH_RETEST"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY", "CHOCH_RETEST"))

            tracker2 = main.SignalStabilityTracker()
            self.assertFalse(tracker2.confirm("ETHUSDT", "BUY", "OB_FVG_RETEST"))
            self.assertTrue(tracker2.confirm("ETHUSDT", "BUY", "OB_FVG_RETEST"))

    def test_switching_trigger_type_restarts_the_streak_even_with_same_side(self):
        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2), \
             patch.object(config, "STRUCTURE_BREAK_EXTRA_CONFIRM_TICKS", 0), \
             patch.object(config, "EXTRA_CONFIRM_TICKS_FOR_NEW_TRIGGERS", 0):
            tracker = main.SignalStabilityTracker()
            tracker.confirm("BTCUSDT", "BUY", "LIQUIDITY_SWEEP")
            self.assertFalse(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))
            self.assertTrue(tracker.confirm("BTCUSDT", "BUY", "STRUCTURE_BREAK"))


class EvaluateSymbolStabilityTests(unittest.TestCase):
    def setUp(self):
        # See EvaluateSymbolRejectCountsTests.setUp - same reason, both flags.
        # MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED pinned off too: several
        # tests below set a real (non-reversal) signal_trigger and fall
        # through past build_trade_plan, which would otherwise reach the
        # new on-demand exchange.get_open_interest_history call with
        # nothing mocking it - not what those tests are about. The
        # dedicated tests near the bottom of this class re-enable it
        # explicitly per test, same convention as OB_FVG_RETEST_PRICE_WEAK.
        # CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED/CVD_DIVERGENCE_PRICE_
        # HOLD_WEAK_REJECT_ENABLED pinned off for the identical reason -
        # test_oi_regime_not_checked_for_reversal_triggers uses a real
        # CVD_DIVERGENCE signal_trigger and falls through past
        # build_trade_plan, which would otherwise reach these two
        # on-demand exchange calls with nothing mocking them (real bug
        # found while implementing an unrelated later feature - both
        # default True and were added after this class's setUp already
        # existed, so this class was never updated for them).
        for name, value in (
            ("DCA_ENABLED", False), ("RETRACEMENT_ENTRY_ENABLED", False),
            ("MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", False),
            ("CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False),
            ("CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _plan(self):
        return {
            "symbol": "BTCUSDT", "entry_price": 100, "sl_price": 98,
            "tp1_price": 102, "tp2_price": 104,
        }

    def test_signal_is_rejected_as_not_yet_stable_before_the_required_ticks(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        stability = main.SignalStabilityTracker()

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 3), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts, {}, stability)

        self.assertEqual(reject_counts["SIGNAL_NOT_YET_STABLE"], 1)
        self.assertEqual(len(positions.registered), 0)

    def test_entry_fires_once_the_signal_has_held_for_enough_ticks(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter()
        stability = main.SignalStabilityTracker()

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 3), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}), \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            for _ in range(3):
                main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, reject_counts, {}, stability)

        self.assertEqual(len(positions.registered), 1)
        # Only the first two ticks were rejected as not-yet-stable; the
        # third is the one that actually enters.
        self.assertEqual(reject_counts["SIGNAL_NOT_YET_STABLE"], 2)

    def test_a_gap_in_qualifying_resets_the_streak(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        stability = main.SignalStabilityTracker()

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 2):
            with patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}):
                main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

            with patch.object(signal_engine, "evaluate", return_value={"signal": None, "reason": "NOT_IN_OTE"}):
                main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

            with patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}):
                # Would be the 2nd consecutive BUY (and confirm) if the gap
                # hadn't reset the streak - it's only the 1st again.
                main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

        self.assertEqual(len(positions.registered), 0)

    def test_streak_resets_after_a_successful_entry(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        stability = main.SignalStabilityTracker()

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 1), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}), \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

        self.assertNotIn("BTCUSDT", stability._streaks)

    def test_long_short_ratio_is_fetched_only_once_the_signal_confirms(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        stability = main.SignalStabilityTracker()
        result = {"signal": "BUY", "symbol": "BTCUSDT"}

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 1), \
             patch.object(config, "LONG_SHORT_RATIO_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_long_short_ratio", return_value=1.8) as ratio_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(), "OK")) as plan_mock, \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}), \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

        ratio_mock.assert_called_once_with("BTCUSDT")
        passed_signal, _ = plan_mock.call_args.args
        self.assertEqual(passed_signal["long_short_ratio"], 1.8)

    def test_long_short_ratio_is_not_fetched_before_the_signal_is_stable(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        stability = main.SignalStabilityTracker()

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 3), \
             patch.object(config, "LONG_SHORT_RATIO_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(exchange, "get_long_short_ratio") as ratio_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

        ratio_mock.assert_not_called()

    def test_long_short_ratio_disabled_by_config_is_never_fetched(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        stability = main.SignalStabilityTracker()

        with patch.object(config, "SIGNAL_CONFIRM_TICKS", 1), \
             patch.object(config, "LONG_SHORT_RATIO_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(exchange, "get_long_short_ratio") as ratio_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}), \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability)

        ratio_mock.assert_not_called()

    # config.LONG_SHORT_FAVORABLE_REJECT_ENABLED - 2026-08-31, real
    # evidence (see config.py's own comment).

    def test_long_short_unfavorable_rejects_before_building_a_plan(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}
        reject_counts = Counter()
        reject_symbols = {}
        reject_trigger_counts = Counter()
        reject_trigger_symbols = {}

        with patch.object(config, "LONG_SHORT_RATIO_ENABLED", True), \
             patch.object(config, "LONG_SHORT_FAVORABLE_REJECT_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_long_short_ratio", return_value=5.0), \
             patch.object(signal_engine, "long_short_favorable", return_value=False), \
             patch.object(risk_manager, "build_trade_plan") as plan_mock:
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols, None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        plan_mock.assert_not_called()
        self.assertEqual(len(positions.registered), 0)
        self.assertEqual(reject_counts["LONG_SHORT_UNFAVORABLE"], 1)
        self.assertIn("BTCUSDT", reject_symbols["LONG_SHORT_UNFAVORABLE"])
        self.assertEqual(reject_trigger_counts["LONG_SHORT_UNFAVORABLE | triggers=EMA_PULLBACK"], 1)

    def test_long_short_favorable_none_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "LONG_SHORT_RATIO_ENABLED", True), \
             patch.object(config, "LONG_SHORT_FAVORABLE_REJECT_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_long_short_ratio", return_value=None), \
             patch.object(signal_engine, "long_short_favorable", return_value=None), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_long_short_favorable_reject_disabled_by_flag(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "LONG_SHORT_RATIO_ENABLED", True), \
             patch.object(config, "LONG_SHORT_FAVORABLE_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_long_short_ratio", return_value=5.0), \
             patch.object(signal_engine, "long_short_favorable", return_value=False), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    # config.OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED - 2026-09-02, real
    # evidence (see config.py's own comment). Scoped to OB_FVG_RETEST
    # only - same on-demand-after-signal shape as LONG_SHORT_RATIO_ENABLED
    # above.

    def _fake_klines_df(self):
        return pd.DataFrame({
            "time": range(15), "open": [100.0] * 15, "high": [101.0] * 15,
            "low": [99.0] * 15, "close": [101.0] * 15, "volume": [10.0] * 15,
        })

    def test_price_hold_weak_rejects_before_building_a_plan(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "OB_FVG_RETEST"}
        reject_counts = Counter()
        reject_symbols = {}
        reject_trigger_counts = Counter()
        reject_trigger_symbols = {}

        with patch.object(config, "OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_PRICE_HOLD_PCT", 0.5), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines", return_value=self._fake_klines_df()), \
             patch.object(market_structure, "price_hold_consistency", return_value=0.3), \
             patch.object(risk_manager, "build_trade_plan") as plan_mock:
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols, None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        plan_mock.assert_not_called()
        self.assertEqual(len(positions.registered), 0)
        self.assertEqual(reject_counts["OB_FVG_RETEST_PRICE_WEAK"], 1)
        self.assertIn("BTCUSDT", reject_symbols["OB_FVG_RETEST_PRICE_WEAK"])
        self.assertEqual(reject_trigger_counts["OB_FVG_RETEST_PRICE_WEAK | triggers=OB_FVG_RETEST"], 1)

    def test_price_hold_strong_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "OB_FVG_RETEST"}

        with patch.object(config, "OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_MIN_PRICE_HOLD_PCT", 0.5), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines", return_value=self._fake_klines_df()), \
             patch.object(market_structure, "price_hold_consistency", return_value=0.8), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_price_hold_consistency_none_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "OB_FVG_RETEST"}

        with patch.object(config, "OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines", return_value=None), \
             patch.object(market_structure, "price_hold_consistency", return_value=None), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_price_hold_weak_reject_disabled_by_flag(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "OB_FVG_RETEST"}

        with patch.object(config, "OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines") as klines_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        klines_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_price_hold_not_checked_for_other_triggers(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        # CHOCH_RETEST - tested identically in the real replay and came
        # back flat, deliberately not applied there.
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CHOCH_RETEST"}

        with patch.object(config, "OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines") as klines_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        klines_mock.assert_not_called()
        plan_mock.assert_called_once()

    # config.MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED - 2026-09-02, real
    # evidence (see config.py's own comment). Same on-demand-after-signal
    # shape as OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED above, but scoped
    # to MARKET_CHOPPY's own full trigger population (config.
    # trigger_gate_profiles()), not just one trigger.

    def test_oi_crowded_rejects_before_building_a_plan(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}
        reject_counts = Counter()
        reject_symbols = {}
        reject_trigger_counts = Counter()
        reject_trigger_symbols = {}

        with patch.object(config, "MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", True), \
             patch.object(config, "MARKET_CHOPPY_OI_REGIME_MAX_PERCENTILE", 0.75), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_open_interest_history", return_value=[(0.0, 1.0)] * 30), \
             patch.object(market_structure, "oi_percentile", return_value=0.9), \
             patch.object(risk_manager, "build_trade_plan") as plan_mock:
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols, None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        plan_mock.assert_not_called()
        self.assertEqual(len(positions.registered), 0)
        self.assertEqual(reject_counts["MARKET_CHOPPY_OI_CROWDED"], 1)
        self.assertIn("BTCUSDT", reject_symbols["MARKET_CHOPPY_OI_CROWDED"])
        self.assertEqual(reject_trigger_counts["MARKET_CHOPPY_OI_CROWDED | triggers=EMA_PULLBACK"], 1)

    def test_oi_below_cutoff_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", True), \
             patch.object(config, "MARKET_CHOPPY_OI_REGIME_MAX_PERCENTILE", 0.75), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_open_interest_history", return_value=[(0.0, 1.0)] * 30), \
             patch.object(market_structure, "oi_percentile", return_value=0.5), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_oi_percentile_none_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_open_interest_history", return_value=None), \
             patch.object(market_structure, "oi_percentile", return_value=None), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_oi_crowded_reject_disabled_by_flag(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_open_interest_history") as history_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        history_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_oi_regime_not_checked_for_reversal_triggers(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        # CVD_DIVERGENCE - one of the 3 reversal triggers MARKET_CHOPPY
        # itself is exempt for (config.
        # MARKET_CHOPPY_SKIP_FOR_REVERSAL_TRIGGERS_ENABLED) - this gate
        # reuses that same exemption via config.trigger_gate_profiles().
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_open_interest_history") as history_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        history_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_oi_regime_uses_the_configured_lookback(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "MARKET_CHOPPY_OI_REGIME_REJECT_ENABLED", True), \
             patch.object(config, "MARKET_CHOPPY_OI_REGIME_LOOKBACK_SAMPLES", 60), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_open_interest_history", return_value=None) as history_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        history_mock.assert_called_once_with("BTCUSDT", period="1h", limit=60)

    # config.CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED /
    # config.CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED - 2026-09-02,
    # real evidence (see each flag's own config.py comment). Both scoped
    # to CVD_DIVERGENCE only, same on-demand-after-signal shape as every
    # other gate above.

    def test_taker_flow_disagree_rejects_before_building_a_plan(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}
        reject_counts = Counter()
        reject_symbols = {}
        reject_trigger_counts = Counter()
        reject_trigger_symbols = {}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_taker_longshort_ratio", return_value=0.8), \
             patch.object(signal_engine, "taker_flow_agrees", return_value=False), \
             patch.object(risk_manager, "build_trade_plan") as plan_mock:
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols, None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        plan_mock.assert_not_called()
        self.assertEqual(len(positions.registered), 0)
        self.assertEqual(reject_counts["CVD_DIVERGENCE_TAKER_FLOW_DISAGREE"], 1)
        self.assertIn("BTCUSDT", reject_symbols["CVD_DIVERGENCE_TAKER_FLOW_DISAGREE"])
        self.assertEqual(
            reject_trigger_counts["CVD_DIVERGENCE_TAKER_FLOW_DISAGREE | triggers=CVD_DIVERGENCE"], 1
        )

    def test_taker_flow_agrees_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_taker_longshort_ratio", return_value=1.5), \
             patch.object(signal_engine, "taker_flow_agrees", return_value=True), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_taker_flow_none_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_taker_longshort_ratio", return_value=None), \
             patch.object(signal_engine, "taker_flow_agrees", return_value=None), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_taker_flow_reject_disabled_by_flag(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_taker_longshort_ratio") as ratio_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        ratio_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_taker_flow_not_checked_for_other_triggers(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "EMA_PULLBACK"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_taker_longshort_ratio") as ratio_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        ratio_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_cvd_price_hold_weak_rejects_before_building_a_plan(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}
        reject_counts = Counter()
        reject_symbols = {}
        reject_trigger_counts = Counter()
        reject_trigger_symbols = {}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_MIN_PRICE_HOLD_PCT", 0.4), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_LOOKBACK_MINUTES", 10), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines", return_value=self._fake_klines_df()), \
             patch.object(market_structure, "price_hold_consistency", return_value=0.3), \
             patch.object(risk_manager, "build_trade_plan") as plan_mock:
            main._evaluate_symbol(
                feed, "BTCUSDT", positions, 1000, reject_counts, reject_symbols, None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        plan_mock.assert_not_called()
        self.assertEqual(len(positions.registered), 0)
        self.assertEqual(reject_counts["CVD_DIVERGENCE_PRICE_HOLD_WEAK"], 1)
        self.assertIn("BTCUSDT", reject_symbols["CVD_DIVERGENCE_PRICE_HOLD_WEAK"])
        self.assertEqual(
            reject_trigger_counts["CVD_DIVERGENCE_PRICE_HOLD_WEAK | triggers=CVD_DIVERGENCE"], 1
        )

    def test_cvd_price_hold_strong_does_not_reject(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", True), \
             patch.object(config, "CVD_DIVERGENCE_MIN_PRICE_HOLD_PCT", 0.4), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines", return_value=self._fake_klines_df()), \
             patch.object(market_structure, "price_hold_consistency", return_value=0.8), \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        plan_mock.assert_called_once()

    def test_cvd_price_hold_weak_reject_disabled_by_flag(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "CVD_DIVERGENCE"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines") as klines_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        klines_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_cvd_price_hold_not_checked_for_other_triggers(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        result = {"signal": "BUY", "symbol": "BTCUSDT", "signal_trigger": "OB_FVG_RETEST"}

        with patch.object(config, "CVD_DIVERGENCE_TAKER_FLOW_REJECT_ENABLED", False), \
             patch.object(config, "CVD_DIVERGENCE_PRICE_HOLD_WEAK_REJECT_ENABLED", True), \
             patch.object(config, "OB_FVG_RETEST_PRICE_WEAK_REJECT_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value=result), \
             patch.object(exchange, "get_klines") as klines_mock, \
             patch.object(risk_manager, "build_trade_plan", return_value=(None, "SL_TOO_TIGHT")) as plan_mock:
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000)

        klines_mock.assert_not_called()
        plan_mock.assert_called_once()

    def test_stability_none_behaves_like_the_original_ungated_behavior(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}), \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter(), {}, stability=None)

        self.assertEqual(len(positions.registered), 1)


class LogHeartbeatRejectSummaryTests(unittest.TestCase):
    def test_reject_counts_are_logged_sorted_by_frequency(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({"NOT_IN_OTE": 5, "CVD_NOT_CONFIRMED": 12, "NO_LIVE_STRUCTURE_BREAK": 80})

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, reject_counts)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("NO_LIVE_STRUCTURE_BREAK=80", logged)
        self.assertIn("CVD_NOT_CONFIRMED=12", logged)
        self.assertIn("NOT_IN_OTE=5", logged)
        # Most frequent reason appears before less frequent ones.
        self.assertLess(logged.index("NO_LIVE_STRUCTURE_BREAK=80"), logged.index("NOT_IN_OTE=5"))

    def test_empty_reject_counts_logs_no_summary_line(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, Counter())

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertNotIn("REJECTED", logged)

    def test_reject_counts_defaults_to_none_without_raising(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        main._log_heartbeat(feed, ["BTCUSDT"], positions)

    def test_symbol_sample_is_included_next_to_its_reason(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({"NOT_IN_OTE": 2})
        reject_symbols = {"NOT_IN_OTE": ["BTCUSDT", "ETHUSDT"]}

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, reject_counts, reject_symbols)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("NOT_IN_OTE=2[BTCUSDT,ETHUSDT]", logged)

    def test_truncated_sample_gets_an_ellipsis_marker(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({"NO_LIVE_STRUCTURE_BREAK": 679})
        reject_symbols = {"NO_LIVE_STRUCTURE_BREAK": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, reject_counts, reject_symbols)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("NO_LIVE_STRUCTURE_BREAK=679[BTCUSDT,ETHUSDT,SOLUSDT,...]", logged)

    def test_reason_without_a_sample_has_no_bracket_suffix(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({"UNKNOWN": 3})

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, reject_counts, reject_symbols=None)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("UNKNOWN=3 ", logged + " ")
        self.assertNotIn("UNKNOWN=3[", logged)

    def test_reject_trigger_counts_get_their_own_separate_line(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({"AGAINST_HTF_BIAS": 5})
        reject_trigger_counts = Counter({"AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK": 3})

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(
                feed, ["BTCUSDT"], positions, reject_counts, None,
                reject_trigger_counts, None,
            )

        lines = [call.args[0] for call in log_mock.call_args_list]
        self.assertTrue(any(line.startswith("  REJECTED since last heartbeat") for line in lines))
        trigger_lines = [line for line in lines if line.startswith("  REJECTED BY TRIGGER")]
        self.assertEqual(len(trigger_lines), 1)
        self.assertIn("AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK=3", trigger_lines[0])

    def test_empty_reject_trigger_counts_logs_no_trigger_line(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, Counter(), None, Counter())

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertNotIn("REJECTED BY TRIGGER", logged)

    def test_reject_trigger_counts_defaults_to_none_without_raising(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        main._log_heartbeat(feed, ["BTCUSDT"], positions, Counter())

    def test_oi_rising_gate_gets_its_own_dedicated_line(self):
        # Real gap this was built for: OI_RISING sits behind AGAINST_HTF_
        # BIAS/zone/OTE/order-block, so far fewer candidates ever reach it
        # than reasons like NO_LIVE_STRUCTURE_BREAK - buried under 8 more
        # frequent reasons, it would never appear in the top-8-only summary
        # line above even though reject_counts tallied it correctly.
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({
            "NO_LIVE_STRUCTURE_BREAK": 900, "AGAINST_HTF_BIAS": 800,
            "MARKET_CHOPPY": 700, "NOT_IN_OTE": 600, "HTF_TREND_STALE": 500,
            "NOT_IN_DISCOUNT": 400, "NOT_IN_PREMIUM": 300, "CVD_NOT_CONFIRMED": 200,
            "OI_RISING": 3,
        })

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, reject_counts)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertNotIn("OI_RISING", logged.split("OI_RISING gate")[0])  # not in the top-8 line
        self.assertIn("OI_RISING gate | blocked 3 since last heartbeat | 0 total since bot start", logged)

    def test_oi_rising_line_shows_the_running_total_since_start(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_counts = Counter({"OI_RISING": 2})

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, reject_counts, oi_rising_block_total=15)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("OI_RISING gate | blocked 2 since last heartbeat | 15 total since bot start", logged)

    def test_oi_rising_line_still_shows_a_quiet_window_against_a_nonzero_total(self):
        # Nothing blocked THIS heartbeat, but the running total is real -
        # must not go silent just because this particular window was quiet.
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, Counter(), oi_rising_block_total=7)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("OI_RISING gate | blocked 0 since last heartbeat | 7 total since bot start", logged)

    def test_no_oi_rising_line_when_nothing_has_ever_blocked(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions, Counter())

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertNotIn("OI_RISING gate", logged)

    def test_trigger_symbol_sample_is_included(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        reject_trigger_counts = Counter({"AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK": 1})
        reject_trigger_symbols = {"AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK": ["BTCUSDT"]}

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(
                feed, ["BTCUSDT"], positions, Counter(), None,
                reject_trigger_counts, reject_trigger_symbols,
            )

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("AGAINST_HTF_BIAS | triggers=STRUCTURE_BREAK=1[BTCUSDT]", logged)


class EvaluateSymbolLimitEntryModeTests(unittest.TestCase):
    """config.LIMIT_ENTRY_MODE_ENABLED is a per-signal ROUTING switch
    (main._evaluate_symbol), not "always place a limit order" - a signal
    at or below ENTRY_ROUTING_EXTENSION_THRESHOLD_R still gets a market
    order; only one above that threshold (but under the hard
    MAX_ENTRY_EXTENSION_R reject risk_manager already applied) gets
    routed to enter_trade_limit + register_pending_entry. The
    has_open_position/cooldown/MAX_TOTAL_POSITIONS guards above need no
    changes for this (see position_manager.PENDING_LIMIT_FILL design
    notes) - not re-proven here with a fake, since that's only meaningful
    against the real PositionManager dict (see test_position_manager.py's
    RegisterPendingEntryTests.test_reserves_a_max_total_positions_slot_immediately).

    config.DCA_ENABLED and config.RETRACEMENT_ENTRY_ENABLED both take
    priority over this routing entirely (see main.py's own comment above
    use_limit) - pinned off below so this class keeps testing LIMIT_ENTRY_
    MODE_ENABLED's routing in isolation, the same path it was written
    against."""

    def setUp(self):
        for name, value in (("DCA_ENABLED", False), ("RETRACEMENT_ENTRY_ENABLED", False)):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _plan(self, entry_extension_r=0.5):
        return {
            "symbol": "BTCUSDT", "entry_price": 100, "sl_price": 98,
            "tp1_price": 102, "tp2_price": 104, "entry_extension_r": entry_extension_r,
        }

    def test_extended_signal_uses_the_limit_entry_path(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.5), "OK")), \
             patch.object(execution, "enter_trade_limit", return_value={"ok": True, "shadow": True}) as enter_limit, \
             patch.object(execution, "enter_trade") as enter_market, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_limit.assert_called_once()
        enter_market.assert_not_called()
        self.assertEqual(len(positions.registered_pending), 1)
        self.assertEqual(len(positions.registered), 0)

    def test_unextended_signal_still_uses_the_market_entry_path(self):
        # Below the routing threshold - guaranteed fill beats limit
        # fill-uncertainty when the chase cost is minimal anyway, even
        # with LIMIT_ENTRY_MODE_ENABLED=True.
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.1), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}) as enter_market, \
             patch.object(execution, "enter_trade_limit") as enter_limit, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_market.assert_called_once()
        enter_limit.assert_not_called()
        self.assertEqual(len(positions.registered), 1)
        self.assertEqual(len(positions.registered_pending), 0)

    def test_exactly_at_the_threshold_uses_the_market_entry_path(self):
        # Boundary is inclusive on the market side (">" not ">="),
        # matching risk_manager._entry_too_extended's own convention.
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.2), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}) as enter_market, \
             patch.object(execution, "enter_trade_limit") as enter_limit, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_market.assert_called_once()
        enter_limit.assert_not_called()

    def test_limit_entry_mode_disabled_always_uses_the_market_entry_path(self):
        # Even a heavily-extended signal stays on the market path when
        # the feature is off entirely.
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.9), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}) as enter_market, \
             patch.object(execution, "enter_trade_limit") as enter_limit, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_market.assert_called_once()
        enter_limit.assert_not_called()
        self.assertEqual(len(positions.registered), 1)
        self.assertEqual(len(positions.registered_pending), 0)

    def test_missing_extension_data_defensively_falls_back_to_market(self):
        # entry_extension_r is always a real float in practice
        # (build_trade_plan only returns "OK" once it's computable) - this
        # is a defensive-fallback test, not an expected real path.
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=None), "OK")), \
             patch.object(execution, "enter_trade", return_value={"ok": True, "shadow": True}) as enter_market, \
             patch.object(execution, "enter_trade_limit") as enter_limit, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_market.assert_called_once()
        enter_limit.assert_not_called()

    def test_limit_entry_failure_marks_entry_failure_not_registered(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.5), "OK")), \
             patch.object(execution, "enter_trade_limit", return_value={"ok": False, "error": "boom"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        self.assertEqual(len(positions.registered_pending), 0)


class LogHeartbeatRetracementPendingTests(unittest.TestCase):
    """config.RETRACEMENT_ENTRY_ENABLED - this stage's position dict keeps
    the whole plan nested under "plan" instead of flattening entry_price/
    sl_price/tp1_price/tp2_price/tp_price/single_tp the way every other
    stage's shape does (see register_retracement_pending's own docstring) -
    the general OPEN-position log line would KeyError on it without its
    own branch."""

    def test_does_not_raise_and_logs_from_the_nested_plan(self):
        feed = _FakeFeed()
        positions = _FakePositions()
        positions.positions = {
            "BTCUSDT": {
                "side": "BUY", "stage": main.RETRACEMENT_PENDING,
                "retracement_price": 99.8,
                "plan": {"entry_price": 100, "sl_price": 98},
            },
        }

        with patch.object(main, "log_info") as log_mock:
            main._log_heartbeat(feed, ["BTCUSDT"], positions)

        logged = " ".join(call.args[0] for call in log_mock.call_args_list)
        self.assertIn("stage=RETRACEMENT_PENDING", logged)
        self.assertIn("retracement=99.8", logged)
        self.assertIn("trigger=100", logged)
        self.assertIn("sl=98", logged)


class PollPositionsDispatchTests(unittest.TestCase):
    """main._poll_positions dispatches on stage == PENDING_LIMIT_FILL
    first, before the existing shadow/live dispatch - a resting entry
    must never be polled by poll_live/poll_shadow (which assume a real
    fill already exists), and a filled position must never be polled by
    the pending-entry methods."""

    def _positions_with(self, stage, shadow):
        positions = MagicMock()
        positions.positions = {"BTCUSDT": {"shadow": shadow, "stage": stage}}
        return positions

    def test_pending_live_dispatches_to_poll_pending_entry(self):
        feed = _FakeFeed()
        positions = self._positions_with(main.PENDING_LIMIT_FILL, shadow=False)

        main._poll_positions(feed, positions)

        positions.poll_pending_entry.assert_called_once()
        positions.poll_live.assert_not_called()
        positions.poll_shadow.assert_not_called()
        positions.poll_shadow_pending_entry.assert_not_called()

    def test_pending_shadow_dispatches_to_poll_shadow_pending_entry(self):
        feed = _FakeFeed()
        positions = self._positions_with(main.PENDING_LIMIT_FILL, shadow=True)

        main._poll_positions(feed, positions)

        positions.poll_shadow_pending_entry.assert_called_once()
        positions.poll_live.assert_not_called()
        positions.poll_shadow.assert_not_called()
        positions.poll_pending_entry.assert_not_called()

    def test_retracement_live_dispatches_to_poll_retracement_pending(self):
        feed = _FakeFeed()
        positions = self._positions_with(main.RETRACEMENT_PENDING, shadow=False)

        main._poll_positions(feed, positions)

        positions.poll_retracement_pending.assert_called_once()
        positions.poll_live.assert_not_called()
        positions.poll_shadow.assert_not_called()
        positions.poll_shadow_retracement_pending.assert_not_called()

    def test_retracement_shadow_dispatches_to_poll_shadow_retracement_pending(self):
        feed = _FakeFeed()
        positions = self._positions_with(main.RETRACEMENT_PENDING, shadow=True)

        main._poll_positions(feed, positions)

        positions.poll_shadow_retracement_pending.assert_called_once()
        positions.poll_live.assert_not_called()
        positions.poll_shadow.assert_not_called()
        positions.poll_retracement_pending.assert_not_called()

    def test_filled_live_dispatches_to_poll_live_unchanged(self):
        feed = _FakeFeed()
        positions = self._positions_with("TP1_PENDING", shadow=False)

        main._poll_positions(feed, positions)

        positions.poll_live.assert_called_once_with(
            "BTCUSDT", candles=feed.candles.get("BTCUSDT"),
            htf_candles=feed.htf_candles.get("BTCUSDT"), cvd_snapshot=feed.cvd.snapshot("BTCUSDT"),
            crash_snapshot=feed.crash_detector.snapshot(),
            btc_low=None, btc_high=None,
        )
        positions.poll_pending_entry.assert_not_called()
        positions.poll_shadow_pending_entry.assert_not_called()

    def test_filled_shadow_dispatches_to_poll_shadow_unchanged(self):
        feed = _FakeFeed()
        positions = self._positions_with("TP1_PENDING", shadow=True)

        main._poll_positions(feed, positions)

        positions.poll_shadow.assert_called_once()
        positions.poll_pending_entry.assert_not_called()
        positions.poll_shadow_pending_entry.assert_not_called()


class RefreshWatchlistTests(unittest.TestCase):
    """config.WATCHLIST_REFRESH_SECONDS - main._refresh_watchlist re-ranks
    by volume and soft-restarts the feed (RealtimeMarketData has no live
    add/remove-symbol path) only when the resulting symbol set actually
    changed, always keeping any symbol with an open position."""

    def _positions_with(self, open_symbols):
        positions = MagicMock()
        positions.positions = {symbol: {} for symbol in open_symbols}
        return positions

    def test_pinned_scan_symbols_is_a_noop(self):
        old_feed = MagicMock()
        positions = self._positions_with([])

        with patch.object(config, "SCAN_SYMBOLS", ["BTCUSDT"]), \
             patch.object(main, "_select_symbols") as select:
            symbols, feed = main._refresh_watchlist(old_feed, positions, ["BTCUSDT"])

        select.assert_not_called()
        old_feed.stop.assert_not_called()
        self.assertIs(feed, old_feed)
        self.assertEqual(symbols, ["BTCUSDT"])

    def test_empty_selection_keeps_the_current_list(self):
        old_feed = MagicMock()
        positions = self._positions_with([])

        with patch.object(config, "SCAN_SYMBOLS", []), \
             patch.object(main, "_select_symbols", return_value=[]):
            symbols, feed = main._refresh_watchlist(old_feed, positions, ["BTCUSDT"])

        old_feed.stop.assert_not_called()
        self.assertIs(feed, old_feed)
        self.assertEqual(symbols, ["BTCUSDT"])

    def test_unchanged_symbol_set_does_not_rebuild_the_feed(self):
        old_feed = MagicMock()
        positions = self._positions_with([])

        with patch.object(config, "SCAN_SYMBOLS", []), \
             patch.object(main, "_select_symbols", return_value=["BTCUSDT", "ETHUSDT"]):
            symbols, feed = main._refresh_watchlist(
                old_feed, positions, ["ETHUSDT", "BTCUSDT"]  # same set, different order
            )

        old_feed.stop.assert_not_called()
        self.assertIs(feed, old_feed)

    def test_changed_selection_rebuilds_the_feed(self):
        old_feed = MagicMock()
        new_feed = MagicMock()
        positions = self._positions_with([])

        with patch.object(config, "SCAN_SYMBOLS", []), \
             patch.object(main, "_select_symbols", return_value=["BTCUSDT", "ETHUSDT"]), \
             patch.object(main, "RealtimeMarketData", return_value=new_feed) as ctor:
            symbols, feed = main._refresh_watchlist(old_feed, positions, ["BTCUSDT"])

        old_feed.stop.assert_called_once()
        ctor.assert_called_once_with(["BTCUSDT", "ETHUSDT"], shutdown_event=main.shutdown_event)
        new_feed.start.assert_called_once()
        self.assertIs(feed, new_feed)
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])

    def test_open_position_symbol_is_kept_even_if_it_dropped_out_on_volume(self):
        old_feed = MagicMock()
        new_feed = MagicMock()
        positions = self._positions_with(["OLDCOINUSDT"])

        with patch.object(config, "SCAN_SYMBOLS", []), \
             patch.object(main, "_select_symbols", return_value=["BTCUSDT", "ETHUSDT"]), \
             patch.object(main, "RealtimeMarketData", return_value=new_feed) as ctor:
            symbols, feed = main._refresh_watchlist(
                old_feed, positions, ["BTCUSDT", "OLDCOINUSDT"]
            )

        ctor.assert_called_once_with(
            ["BTCUSDT", "ETHUSDT", "OLDCOINUSDT"], shutdown_event=main.shutdown_event
        )
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT", "OLDCOINUSDT"])

    def test_open_position_symbol_already_in_the_fresh_list_is_not_duplicated(self):
        old_feed = MagicMock()
        new_feed = MagicMock()
        positions = self._positions_with(["BTCUSDT"])

        with patch.object(config, "SCAN_SYMBOLS", []), \
             patch.object(main, "_select_symbols", return_value=["BTCUSDT", "ETHUSDT"]), \
             patch.object(main, "RealtimeMarketData", return_value=new_feed) as ctor:
            symbols, feed = main._refresh_watchlist(old_feed, positions, ["ETHUSDT"])

        ctor.assert_called_once_with(["BTCUSDT", "ETHUSDT"], shutdown_event=main.shutdown_event)
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])


class EvaluateSymbolDcaRoutingTests(unittest.TestCase):
    """config.DCA_ENABLED takes priority over LIMIT_ENTRY_MODE_ENABLED's
    own market-vs-limit routing entirely (see main.py's comment above
    use_limit) - a DCA-enabled signal always enters via enter_trade_dca_
    pending/register_dca_pending, regardless of entry_extension_r."""

    def setUp(self):
        # config.RETRACEMENT_ENTRY_ENABLED takes priority over even
        # DCA_ENABLED's own routing (see main.py's use_retracement) - a
        # real .env flip to True would route these fixtures (missing
        # plan["side"]) into enter_trade_retracement instead of the DCA/
        # limit paths this class is actually testing. Pinned off, same
        # isolation reason as every other routing test class here.
        patcher = patch.object(config, "RETRACEMENT_ENTRY_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _plan(self, entry_extension_r=0.5):
        return {
            "symbol": "BTCUSDT", "entry_price": 100, "sl_price": 98,
            "tp1_price": 102, "tp2_price": 104, "entry_extension_r": entry_extension_r,
            "dca_price": 96,
        }

    def test_dca_enabled_routes_to_dca_pending_even_when_extended(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "DCA_ENABLED", True), \
             patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.5), "OK")), \
             patch.object(execution, "enter_trade_dca_pending", return_value={"ok": True, "shadow": True}) as enter_dca, \
             patch.object(execution, "enter_trade_limit") as enter_limit, \
             patch.object(execution, "enter_trade") as enter_market, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_dca.assert_called_once()
        enter_limit.assert_not_called()
        enter_market.assert_not_called()
        self.assertEqual(len(positions.registered_dca_pending), 1)

    def test_dca_disabled_falls_back_to_limit_routing(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "DCA_ENABLED", False), \
             patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.5), "OK")), \
             patch.object(execution, "enter_trade_dca_pending") as enter_dca, \
             patch.object(execution, "enter_trade_limit", return_value={"ok": True, "shadow": True}) as enter_limit, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_limit.assert_called_once()
        enter_dca.assert_not_called()


class EvaluateSymbolRetracementRoutingTests(unittest.TestCase):
    """config.RETRACEMENT_ENTRY_ENABLED takes priority over BOTH DCA_ENABLED
    and LIMIT_ENTRY_MODE_ENABLED's own routing entirely - it applies
    regardless of entry_extension_r, and hands off into DCA_PENDING/
    TP1_PENDING itself once resolved (position_manager.
    _finalize_retracement_entry), not at routing time."""

    def _plan(self, entry_extension_r=0.5, dca_price=96):
        return {
            "symbol": "BTCUSDT", "entry_price": 100, "sl_price": 98,
            "tp1_price": 102, "tp2_price": 104, "entry_extension_r": entry_extension_r,
            "dca_price": dca_price,
        }

    def test_retracement_enabled_routes_there_even_when_dca_is_also_on(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "RETRACEMENT_ENTRY_ENABLED", True), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(config, "LIMIT_ENTRY_MODE_ENABLED", True), \
             patch.object(config, "ENTRY_ROUTING_EXTENSION_THRESHOLD_R", 0.2), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.5), "OK")), \
             patch.object(execution, "enter_trade_retracement", return_value={"ok": True, "shadow": True}) as enter_retracement, \
             patch.object(execution, "enter_trade_dca_pending") as enter_dca, \
             patch.object(execution, "enter_trade_limit") as enter_limit, \
             patch.object(execution, "enter_trade") as enter_market, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_retracement.assert_called_once()
        enter_dca.assert_not_called()
        enter_limit.assert_not_called()
        enter_market.assert_not_called()
        self.assertEqual(len(positions.registered_retracement_pending), 1)
        self.assertEqual(len(positions.registered_dca_pending), 0)

    def test_retracement_enabled_routes_there_even_for_an_unextended_signal(self):
        # Unlike LIMIT_ENTRY_MODE_ENABLED's own routing, this applies
        # regardless of entry_extension_r - it's not an extension-based
        # switch.
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "RETRACEMENT_ENTRY_ENABLED", True), \
             patch.object(config, "DCA_ENABLED", False), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.05), "OK")), \
             patch.object(execution, "enter_trade_retracement", return_value={"ok": True, "shadow": True}) as enter_retracement, \
             patch.object(execution, "enter_trade") as enter_market, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_retracement.assert_called_once()
        enter_market.assert_not_called()

    def test_disabled_falls_through_to_the_existing_dca_routing(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "RETRACEMENT_ENTRY_ENABLED", False), \
             patch.object(config, "DCA_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(entry_extension_r=0.5), "OK")), \
             patch.object(execution, "enter_trade_retracement") as enter_retracement, \
             patch.object(execution, "enter_trade_dca_pending", return_value={"ok": True, "shadow": True}) as enter_dca, \
             patch.object(signal_journal, "append_signal", return_value="BTCUSDT_123"):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        enter_retracement.assert_not_called()
        enter_dca.assert_called_once()

    def test_entry_failure_marks_entry_failure_not_registered(self):
        feed = _FakeFeed()
        positions = _FakePositions()

        with patch.object(config, "RETRACEMENT_ENTRY_ENABLED", True), \
             patch.object(signal_engine, "evaluate", return_value={"signal": "BUY"}), \
             patch.object(risk_manager, "build_trade_plan", return_value=(self._plan(), "OK")), \
             patch.object(execution, "enter_trade_retracement", return_value={"ok": False, "error": "boom"}):
            main._evaluate_symbol(feed, "BTCUSDT", positions, 1000, Counter())

        self.assertEqual(len(positions.registered_retracement_pending), 0)


if __name__ == "__main__":
    unittest.main()
