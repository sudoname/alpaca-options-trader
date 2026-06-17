"""
Offline tests for Phase 10G-E — Candidate Resolution Store.

No creds, no network, no broker. Covers:
  - recorder writes EVERY evaluated candidate (dicts and to_dict objects)
  - selected vs not-selected marking (and stickiness across passes)
  - upsert by (symbol, strategy, day) — no duplicates, fill-None merge
  - disabled config and malformed rows (never raises)
  - selection_context extraction from paper positions
  - hold-to-expiry payoff math (rising / falling / iron condor)
  - resolve_pending (expiry gating, price lookup failures, dte fallback)
  - record_paper_outcome by candidate id and paper position id
  - summarize counts
  - fail-open hooks present in the ranker and paper runner
  - no execution path touched (static guards)

Phase 11A additions:
  - stamp_candidate / stamp_candidates append + Triple-Gap stamping
  - load_jsonl_records folds last-write-wins, tolerates missing/malformed/partial
  - resolve_jsonl_candidates statuses (expiry / partial / missing / unresolved)
    over BOTH selected and rejected candidates; actual_move + hold-to-expiry sign
  - record_candidates also stamps the JSONL layer (fail-open)
  - SPREAD_PROPOSAL / ADVISORY_CHECK stamp hooks + the two analytics commands
"""

import json
import os
import tempfile
import unittest
from datetime import date, datetime, timezone

import candidate_resolution as cr
from candidate_resolution import CandidateResolutionConfig
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR, NO_TRADE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NOW = datetime(2026, 6, 1, 15, 0, 0, tzinfo=timezone.utc)
DAY = "2026-06-01"


def cand(symbol="SPY", strategy=BULLISH_PUT_CREDIT_SPREAD, **kw):
    d = {"symbol": symbol, "strategy": strategy, "expected_value": 25.0,
         "probability_of_profit": 0.7, "oracle_score": 6.5,
         "volatility_edge": 0.12, "ev_per_dollar_risk": 0.06,
         "max_profit": 60.0, "max_loss": 440.0, "days": 4,
         "recommendation": "TRADE"}
    d.update(kw)
    return d


class _Result:
    """EVResult stand-in exposing only to_dict()."""

    def __init__(self, **kw):
        self._d = cand(**kw)

    def to_dict(self):
        return dict(self._d)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = CandidateResolutionConfig(
            enabled=True, file=os.path.join(self.tmp.name, "cand.json"))
        # Isolate the additive JSONL layer (record_candidates stamps it via the
        # env-resolved default path) so tests never pollute the repo.
        self.jsonl = os.path.join(self.tmp.name, "cand.jsonl")
        self._prev_jsonl_env = os.environ.get("CANDIDATE_RESOLUTION_JSONL")
        os.environ["CANDIDATE_RESOLUTION_JSONL"] = self.jsonl

    def tearDown(self):
        if self._prev_jsonl_env is None:
            os.environ.pop("CANDIDATE_RESOLUTION_JSONL", None)
        else:
            os.environ["CANDIDATE_RESOLUTION_JSONL"] = self._prev_jsonl_env
        self.tmp.cleanup()

    def rows(self):
        return cr.load_records(self.cfg)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
class TestRecording(StoreCase):
    def test_records_every_candidate(self):
        n = cr.record_candidates([cand("SPY"), cand("QQQ")],
                                 source="best_ev_ranker",
                                 config=self.cfg, now=NOW)
        self.assertEqual(n, 2)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        spy = next(r for r in rows if r["symbol"] == "SPY")
        self.assertFalse(spy["selected_for_paper_trade"])
        self.assertFalse(spy["resolved"])
        self.assertEqual(spy["dte"], 4)            # mapped from EVResult days
        self.assertEqual(spy["expected_value"], 25.0)
        self.assertEqual(spy["sources"], ["best_ev_ranker"])
        self.assertTrue(spy["candidate_id"])
        self.assertIsNone(spy["underlying_price_at_resolution"])
        self.assertIsNone(spy["actual_paper_pnl"])

    def test_to_dict_objects_supported(self):
        n = cr.record_candidates([_Result(symbol="IWM")],
                                 config=self.cfg, now=NOW)
        self.assertEqual(n, 1)
        self.assertEqual(self.rows()[0]["symbol"], "IWM")

    def test_upsert_same_day_no_duplicates(self):
        cr.record_candidates([cand()], source="best_ev_ranker",
                             config=self.cfg, now=NOW)
        cr.record_candidates([cand()], source="best_ev_paper_runner",
                             config=self.cfg, now=NOW)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sources"],
                         ["best_ev_ranker", "best_ev_paper_runner"])

    def test_selected_marking_is_sticky(self):
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        cr.record_candidates([cand("SPY"), cand("QQQ")],
                             selected_keys=[key], config=self.cfg, now=NOW)
        rows = {r["symbol"]: r for r in self.rows()}
        self.assertTrue(rows["SPY"]["selected_for_paper_trade"])
        self.assertFalse(rows["QQQ"]["selected_for_paper_trade"])
        # A later pass without the selection must not unset the flag.
        cr.record_candidates([cand("SPY")], config=self.cfg, now=NOW)
        rows = {r["symbol"]: r for r in self.rows()}
        self.assertTrue(rows["SPY"]["selected_for_paper_trade"])

    def test_fill_none_merge_never_overwrites(self):
        cr.record_candidates([cand(max_profit=None)],
                             config=self.cfg, now=NOW)
        self.assertIsNone(self.rows()[0]["max_profit"])
        cr.record_candidates([cand(max_profit=60.0)],
                             config=self.cfg, now=NOW)
        self.assertEqual(self.rows()[0]["max_profit"], 60.0)
        cr.record_candidates([cand(max_profit=99.0)],
                             config=self.cfg, now=NOW)
        self.assertEqual(self.rows()[0]["max_profit"], 60.0)  # first wins

    def test_disabled_config_writes_nothing(self):
        off = CandidateResolutionConfig(
            enabled=False, file=os.path.join(self.tmp.name, "off.json"))
        self.assertEqual(cr.record_candidates([cand()], config=off), 0)
        self.assertFalse(os.path.exists(off.file))

    def test_malformed_rows_skipped(self):
        n = cr.record_candidates(
            [None, "junk", 7, {"symbol": "SPY"},  # no strategy
             {"strategy": BULLISH_PUT_CREDIT_SPREAD},  # no symbol
             cand("DIA")], config=self.cfg, now=NOW)
        self.assertEqual(n, 1)
        self.assertEqual(len(self.rows()), 1)


# --------------------------------------------------------------------------- #
# Selection context from paper positions
# --------------------------------------------------------------------------- #
class TestSelectionContext(unittest.TestCase):
    def test_extracts_execution_facts(self):
        pos = {"id": "pos1", "symbol": "SPY",
               "strategy": BULLISH_PUT_CREDIT_SPREAD,
               "legs": [{"action": "sell", "type": "put", "strike": 100,
                         "expiration": "2026-06-05"},
                        {"action": "buy", "type": "put", "strike": 95}],
               "dte": 4, "entry_underlying_price": 102.5,
               "expected_move": 0.03, "market_expected_move": 0.025}
        selected, extras = cr.selection_context([pos, "junk"], now=NOW)
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        self.assertEqual(selected, [key])
        x = extras[key]
        self.assertEqual(x["strikes"], [95.0, 100.0])
        self.assertEqual(x["expiry"], "2026-06-05")
        self.assertEqual(x["underlying_price_at_entry"], 102.5)
        self.assertEqual(x["paper_position_id"], "pos1")

    def test_empty_positions(self):
        self.assertEqual(cr.selection_context([], now=NOW), ([], {}))


# --------------------------------------------------------------------------- #
# Hold-to-expiry payoff
# --------------------------------------------------------------------------- #
class TestHoldToExpiryPnl(unittest.TestCase):
    def test_rising_payoff_bull_put_credit(self):
        args = (BULLISH_PUT_CREDIT_SPREAD, [95, 100])
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 105, 60, 440), 60.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 90, 60, 440), -440.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 97.5, 60, 440), -190.0)

    def test_falling_payoff_debit_put(self):
        args = (DEBIT_PUT_SPREAD, [95, 100])
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 90, 60, 440), 60.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 105, 60, 440), -440.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 97.5, 60, 440), -190.0)

    def test_iron_condor_payoff(self):
        args = (IRON_CONDOR, [90, 95, 105, 110])
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 100, 50, 450), 50.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 85, 50, 450), -450.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 112, 50, 450), -450.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 92.5, 50, 450), -200.0)
        self.assertEqual(cr.hold_to_expiry_pnl(*args, 107.5, 50, 450), -200.0)

    def test_unknown_inputs_return_none(self):
        self.assertIsNone(cr.hold_to_expiry_pnl(
            BULLISH_PUT_CREDIT_SPREAD, [95, 100], None, 60, 440))
        self.assertIsNone(cr.hold_to_expiry_pnl(
            BULLISH_PUT_CREDIT_SPREAD, [], 100, 60, 440))
        self.assertIsNone(cr.hold_to_expiry_pnl(NO_TRADE, [95, 100],
                                                100, 60, 440))
        self.assertIsNone(cr.hold_to_expiry_pnl(IRON_CONDOR, [95, 100],
                                                100, 60, 440))


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
class TestResolution(StoreCase):
    def seed(self, **extra_fields):
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        extras = {key: dict({"strikes": [95.0, 100.0],
                             "expiry": "2026-06-05"}, **extra_fields)}
        cr.record_candidates([cand("SPY")], selected_keys=[key],
                             extras=extras, config=self.cfg, now=NOW)
        return key

    def test_not_resolved_before_expiry(self):
        self.seed()
        n = cr.resolve_pending(lambda s: 105.0, today=date(2026, 6, 4),
                               config=self.cfg)
        self.assertEqual(n, 0)
        self.assertFalse(self.rows()[0]["resolved"])

    def test_resolves_at_expiry_with_payoff(self):
        self.seed()
        n = cr.resolve_pending(lambda s: 105.0, today=date(2026, 6, 5),
                               config=self.cfg)
        self.assertEqual(n, 1)
        row = self.rows()[0]
        self.assertTrue(row["resolved"])
        self.assertEqual(row["underlying_price_at_resolution"], 105.0)
        self.assertEqual(row["hypothetical_hold_to_expiry_pnl"], 60.0)
        # Already-resolved rows are not touched again.
        self.assertEqual(cr.resolve_pending(lambda s: 1.0,
                                            today=date(2026, 6, 6),
                                            config=self.cfg), 0)

    def test_failing_price_lookup_stays_pending(self):
        self.seed()

        def boom(symbol):
            raise RuntimeError("no data")

        n = cr.resolve_pending(boom, today=date(2026, 6, 6), config=self.cfg)
        self.assertEqual(n, 0)
        self.assertFalse(self.rows()[0]["resolved"])

    def test_expiry_falls_back_to_dte(self):
        # No stated expiry: entry 2026-06-01 + dte 4 -> 2026-06-05.
        cr.record_candidates([cand("QQQ")], config=self.cfg, now=NOW)
        self.assertEqual(cr.resolve_pending(lambda s: 1.0,
                                            today=date(2026, 6, 4),
                                            config=self.cfg), 0)
        self.assertEqual(cr.resolve_pending(lambda s: 1.0,
                                            today=date(2026, 6, 5),
                                            config=self.cfg), 1)

    def test_disabled_config_resolves_nothing(self):
        self.seed()
        off = CandidateResolutionConfig(enabled=False, file=self.cfg.file)
        self.assertEqual(cr.resolve_pending(lambda s: 105.0,
                                            today=date(2026, 6, 9),
                                            config=off), 0)


# --------------------------------------------------------------------------- #
# Paper outcome + summary
# --------------------------------------------------------------------------- #
class TestOutcomeAndSummary(StoreCase):
    def test_outcome_by_candidate_id(self):
        cr.record_candidates([cand("SPY")], config=self.cfg, now=NOW)
        cid = self.rows()[0]["candidate_id"]
        self.assertTrue(cr.record_paper_outcome(cid, 12.5, policy_pnl=8.0,
                                                config=self.cfg))
        row = self.rows()[0]
        self.assertEqual(row["actual_paper_pnl"], 12.5)
        self.assertEqual(row["hypothetical_policy_pnl"], 8.0)

    def test_outcome_by_paper_position_id(self):
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        cr.record_candidates([cand("SPY")], selected_keys=[key],
                             extras={key: {"paper_position_id": "pos9"}},
                             config=self.cfg, now=NOW)
        self.assertTrue(cr.record_paper_outcome("pos9", -5.0,
                                                config=self.cfg))
        self.assertEqual(self.rows()[0]["actual_paper_pnl"], -5.0)

    def test_outcome_unknown_id(self):
        cr.record_candidates([cand("SPY")], config=self.cfg, now=NOW)
        self.assertFalse(cr.record_paper_outcome("nope", 1.0,
                                                 config=self.cfg))

    def test_summarize_counts(self):
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        cr.record_candidates(
            [cand("SPY"), cand("QQQ"), cand("IWM")], selected_keys=[key],
            extras={key: {"strikes": [95.0, 100.0], "expiry": "2026-06-05"}},
            config=self.cfg, now=NOW)
        # SPY resolves via stated expiry; QQQ/IWM via the dte fallback
        # (entry 2026-06-01 + 4 days = same expiry day).
        cr.resolve_pending(lambda s: 105.0, today=date(2026, 6, 5),
                           config=self.cfg)
        summary = cr.summarize(self.cfg)
        self.assertEqual(summary["candidates"], 3)
        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["not_selected"], 2)
        self.assertEqual(summary["resolved"], 3)
        self.assertEqual(summary["pending"], 0)


# --------------------------------------------------------------------------- #
# Phase 11A — append-only JSONL stamp layer
# --------------------------------------------------------------------------- #
def gap_fields(symbol="SPY", strategy=BULLISH_PUT_CREDIT_SPREAD, **kw):
    """Candidate fields carrying the Triple-Gap signal inputs."""
    f = {"symbol": symbol, "strategy": strategy,
         "market_iv": 0.25, "forecast_vol": 0.20,
         "market_expected_move": 12.0, "oracle_expected_move": 10.0,
         "expected_value": 25.0, "probability_of_profit": 0.7,
         "oracle_score": 6.5, "volatility_edge": 0.12,
         "ev_per_dollar_risk": 0.06, "max_profit": 60.0, "max_loss": 440.0,
         "strikes": [95.0, 100.0], "expiry": "2026-06-05", "days": 4,
         "underlying_price_at_entry": 100.0, "recommendation": "TRADE"}
    f.update(kw)
    return f


class TestJsonlStamp(StoreCase):
    def lines(self):
        with open(self.jsonl, "r", encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    def test_stamp_candidate_appends_with_triple_gap(self):
        cid = cr.stamp_candidate(gap_fields(), jsonl_path=self.jsonl, now=NOW)
        self.assertTrue(cid)
        lines = self.lines()
        self.assertEqual(len(lines), 1)
        rec = lines[0]
        self.assertEqual(rec["record_type"], cr.RECORD_TYPE_CANDIDATE)
        self.assertEqual(rec["candidate_id"], cid)
        self.assertEqual(rec["symbol"], "SPY")
        # vol 50, move 20, ev 50 -> (.3*50+.3*20+.4*50)/1.0 = 41.0
        self.assertAlmostEqual(rec["vol_gap"], 0.05)
        self.assertAlmostEqual(rec["move_gap"], 2.0)
        self.assertAlmostEqual(rec["ev_gap"], 25.0)
        self.assertEqual(rec["triple_gap_score"], 41.0)
        self.assertEqual(rec["ev_gap_source"], "zero_baseline")

    def test_stamp_candidate_requires_symbol_and_strategy(self):
        self.assertIsNone(cr.stamp_candidate({"symbol": "SPY"},
                                             jsonl_path=self.jsonl))
        self.assertIsNone(cr.stamp_candidate(
            {"strategy": BULLISH_PUT_CREDIT_SPREAD}, jsonl_path=self.jsonl))
        self.assertFalse(os.path.exists(self.jsonl))

    def test_model_baseline_tag_when_market_neutral_present(self):
        cr.stamp_candidate(gap_fields(market_neutral_expected_value=5.0),
                           jsonl_path=self.jsonl, now=NOW)
        rec = self.lines()[0]
        self.assertEqual(rec["ev_gap"], 20.0)
        self.assertEqual(rec["ev_gap_source"], "model_baseline")

    def test_stamp_candidates_maps_and_flags_selected(self):
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        n = cr.stamp_candidates(
            [_Result(symbol="SPY"), _Result(symbol="QQQ")],
            source_command="BEST_EV_TRADES", selected_keys=[key],
            jsonl_path=self.jsonl, now=NOW)
        self.assertEqual(n, 2)
        recs = {r["symbol"]: r for r in self.lines()}
        self.assertTrue(recs["SPY"]["selected_for_paper_trade"])
        self.assertFalse(recs["QQQ"]["selected_for_paper_trade"])
        self.assertEqual(recs["SPY"]["source_command"], "BEST_EV_TRADES")

    def test_stamp_candidates_merges_extras(self):
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, DAY)
        cr.stamp_candidates(
            [_Result(symbol="SPY")], source_command="BEST_EV_PAPER_RUN",
            extras={key: {"strikes": [95.0, 100.0],
                          "underlying_price_at_entry": 101.0}},
            jsonl_path=self.jsonl, now=NOW)
        rec = self.lines()[0]
        self.assertEqual(rec["strikes"], [95.0, 100.0])
        self.assertEqual(rec["underlying_price_at_entry"], 101.0)

    def test_record_candidates_also_stamps_jsonl_fail_open(self):
        cr.record_candidates([cand("SPY"), cand("QQQ")],
                             source="best_ev_ranker", config=self.cfg, now=NOW)
        recs = cr.load_jsonl_records(self.jsonl)
        self.assertEqual({r["symbol"] for r in recs}, {"SPY", "QQQ"})


class TestJsonlLoad(StoreCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(cr.load_jsonl_records(self.jsonl), [])

    def test_malformed_and_partial_lines_tolerated(self):
        with open(self.jsonl, "w", encoding="utf-8") as fh:
            fh.write("not json\n")
            fh.write("[1,2,3]\n")                     # not a dict
            fh.write(json.dumps({"no": "id"}) + "\n")  # no candidate_id
            fh.write("\n")
            fh.write(json.dumps({"candidate_id": "abc",
                                 "symbol": "SPY"}) + "\n")
        recs = cr.load_jsonl_records(self.jsonl)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["candidate_id"], "abc")

    def test_fold_last_write_wins_non_null(self):
        cid = cr.stamp_candidate(gap_fields(), jsonl_path=self.jsonl, now=NOW)
        # A later snapshot overrides with non-null, keeps frozen stamp fields.
        cr._append_jsonl({"candidate_id": cid,
                          "record_type": cr.RECORD_TYPE_RESOLUTION,
                          "underlying_price_at_resolution": 105.0,
                          "triple_gap_score": None}, self.jsonl)
        recs = cr.load_jsonl_records(self.jsonl)
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["record_type"], cr.RECORD_TYPE_RESOLUTION)
        self.assertEqual(rec["underlying_price_at_resolution"], 105.0)
        self.assertEqual(rec["triple_gap_score"], 41.0)  # null did not clobber


class TestJsonlResolve(StoreCase):
    def seed_two(self):
        """One selected (SPY) and one rejected (QQQ) candidate."""
        sid = cr.stamp_candidate(gap_fields("SPY",
                                            selected_for_paper_trade=True),
                                 jsonl_path=self.jsonl, now=NOW)
        qid = cr.stamp_candidate(gap_fields("QQQ"),
                                 jsonl_path=self.jsonl, now=NOW)
        return sid, qid

    def folded(self):
        return {r["candidate_id"]: r
                for r in cr.load_jsonl_records(self.jsonl)}

    def test_resolved_expiry_tracks_selected_and_rejected(self):
        sid, qid = self.seed_two()
        n = cr.resolve_jsonl_candidates(lambda s: 105.0,
                                        today=date(2026, 6, 5),
                                        jsonl_path=self.jsonl, now=NOW)
        self.assertEqual(n, 2)
        f = self.folded()
        for cid in (sid, qid):
            self.assertEqual(f[cid]["resolution_status"],
                             cr.RESOLUTION_EXPIRY)
            self.assertEqual(f[cid]["underlying_price_at_resolution"], 105.0)
            self.assertAlmostEqual(f[cid]["actual_move"], 0.05)
            self.assertEqual(f[cid]["hypothetical_hold_to_expiry_pnl"], 60.0)
        self.assertTrue(f[sid]["selected_for_paper_trade"])
        self.assertFalse(f[qid]["selected_for_paper_trade"])

    def test_partial_when_price_but_not_due(self):
        sid, _ = self.seed_two()
        n = cr.resolve_jsonl_candidates(lambda s: 90.0,
                                        today=date(2026, 6, 3),
                                        jsonl_path=self.jsonl, now=NOW)
        self.assertEqual(n, 2)
        f = self.folded()
        self.assertEqual(f[sid]["resolution_status"], cr.RESOLUTION_PARTIAL)
        # 90 below short strike 100 -> full loss for the bull put credit.
        self.assertEqual(f[sid]["hypothetical_hold_to_expiry_pnl"], -440.0)
        self.assertAlmostEqual(f[sid]["actual_move"], -0.10)

    def test_missing_price_when_due(self):
        sid, _ = self.seed_two()
        n = cr.resolve_jsonl_candidates(lambda s: None,
                                        today=date(2026, 6, 5),
                                        jsonl_path=self.jsonl, now=NOW)
        self.assertEqual(n, 0)
        f = self.folded()
        self.assertEqual(f[sid]["resolution_status"],
                         cr.RESOLUTION_MISSING_PRICE)

    def test_unresolved_leaves_no_line(self):
        sid, _ = self.seed_two()
        n = cr.resolve_jsonl_candidates(lambda s: None,
                                        today=date(2026, 6, 3),
                                        jsonl_path=self.jsonl, now=NOW)
        self.assertEqual(n, 0)
        f = self.folded()
        self.assertEqual(f[sid]["record_type"], cr.RECORD_TYPE_CANDIDATE)
        self.assertIsNone(f[sid].get("resolution_status"))

    def test_resolved_expiry_not_re_resolved(self):
        self.seed_two()
        cr.resolve_jsonl_candidates(lambda s: 105.0, today=date(2026, 6, 5),
                                    jsonl_path=self.jsonl, now=NOW)
        again = cr.resolve_jsonl_candidates(lambda s: 1.0,
                                            today=date(2026, 6, 9),
                                            jsonl_path=self.jsonl, now=NOW)
        self.assertEqual(again, 0)


# --------------------------------------------------------------------------- #
# Phase 11B — candlestick fields frozen on the candidate stamp (additive)
# --------------------------------------------------------------------------- #
def _candle(o, h, l, c, v=100):
    return {"o": o, "h": h, "l": l, "c": c, "v": v}


def _hammer_candles():
    """Four down-trend candles + a hammer (lower shadow >> small body)."""
    prior = [_candle(p + 0.3, p + 0.6, p - 0.6, p)
             for p in (110, 108, 106, 104)]
    return prior + [_candle(100, 100.7, 98.5, 100.6)]


class TestJsonlCandlestick(StoreCase):
    def lines(self):
        with open(self.jsonl, "r", encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    def test_injected_candles_freeze_six_fields(self):
        cr.stamp_candidate(gap_fields(candles=_hammer_candles()),
                           jsonl_path=self.jsonl, now=NOW)
        rec = self.lines()[0]
        self.assertEqual(rec["candlestick_pattern"], "hammer")
        self.assertEqual(rec["candlestick_bias"], "bullish")
        self.assertEqual(rec["candlestick_strength"], "medium")
        self.assertIsInstance(rec["candlestick_confidence"], float)
        self.assertTrue(rec["candlestick_reason"])
        self.assertTrue(rec["candlestick_requires_confirmation"])
        # The raw candle list is NOT persisted to the record.
        self.assertNotIn("candles", rec)

    def test_precomputed_fields_pass_through(self):
        cr.stamp_candidate(
            gap_fields(candlestick_pattern="evening_star",
                       candlestick_bias="bearish",
                       candlestick_strength="strong",
                       candlestick_confidence=0.78,
                       candlestick_reason="precomputed upstream",
                       candlestick_requires_confirmation=False),
            jsonl_path=self.jsonl, now=NOW)
        rec = self.lines()[0]
        self.assertEqual(rec["candlestick_pattern"], "evening_star")
        self.assertEqual(rec["candlestick_bias"], "bearish")
        self.assertEqual(rec["candlestick_strength"], "strong")
        self.assertEqual(rec["candlestick_confidence"], 0.78)
        self.assertEqual(rec["candlestick_reason"], "precomputed upstream")
        self.assertFalse(rec["candlestick_requires_confirmation"])

    def test_absent_candles_leaves_fields_none(self):
        cr.stamp_candidate(gap_fields(), jsonl_path=self.jsonl, now=NOW)
        rec = self.lines()[0]
        for k in ("candlestick_pattern", "candlestick_bias",
                  "candlestick_strength", "candlestick_confidence",
                  "candlestick_reason", "candlestick_requires_confirmation"):
            self.assertIsNone(rec[k])

    def test_fold_carries_candlestick_to_resolution(self):
        cr.stamp_candidate(
            gap_fields("SPY", selected_for_paper_trade=True,
                       candles=_hammer_candles()),
            jsonl_path=self.jsonl, now=NOW)
        cr.resolve_jsonl_candidates(lambda s: 105.0, today=date(2026, 6, 5),
                                    jsonl_path=self.jsonl, now=NOW)
        folded = {r["candidate_id"]: r
                  for r in cr.load_jsonl_records(self.jsonl)}
        rec = next(iter(folded.values()))
        self.assertEqual(rec["record_type"], cr.RECORD_TYPE_RESOLUTION)
        # Frozen candlestick stamp survives the resolution fold.
        self.assertEqual(rec["candlestick_pattern"], "hammer")
        self.assertEqual(rec["candlestick_bias"], "bullish")


# --------------------------------------------------------------------------- #
# Hooks + no execution path touched
# --------------------------------------------------------------------------- #
class TestHooksPresent(unittest.TestCase):
    def test_ranker_records_candidates_fail_open(self):
        with open(os.path.join(HERE, "best_ev_ranker.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("import candidate_resolution", src)
        self.assertIn("record_candidates", src)

    def test_paper_runner_records_selection_fail_open(self):
        with open(os.path.join(HERE, "best_ev_paper_runner.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("import candidate_resolution", src)
        self.assertIn("selection_context", src)
        self.assertIn("record_candidates", src)

    def test_record_candidates_stamps_jsonl_layer(self):
        with open(os.path.join(HERE, "candidate_resolution.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        # record_candidates must also feed the additive JSONL stamp layer.
        body = src.split("def record_candidates", 1)[1]
        body = body.split("\ndef ", 1)[0]
        self.assertIn("stamp_candidates", body)

    def test_telegram_stamp_hooks_and_commands_present(self):
        with open(os.path.join(HERE, "telegram_bot.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("stamp_candidates", src)
        self.assertIn("SPREAD_PROPOSAL", src)
        self.assertIn("ADVISORY_CHECK", src)
        self.assertIn("TRIPLE_GAP_REPORT", src)
        self.assertIn("SIGNAL_SEPARATION", src)


class TestNoExecutionPathTouched(unittest.TestCase):
    def test_module_never_imports_live_trader_or_network(self):
        with open(os.path.join(HERE, "candidate_resolution.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
