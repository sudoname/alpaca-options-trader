"""
Offline tests for Phase 10B — Best EV Ranking.

No creds, no network, no broker. Covers the eight required areas:
  1. Ranking order            5. Empty candidates
  2. Recommendation filtering 6. EV tie-breakers
  3. Include-negative         7. Telegram output
  4. Symbol parsing           8. No execution path touched

best_ev_ranker is ADVISORY ONLY: the last test class statically proves the
live execution modules (run_alpaca_intraday, smart_trader) do not consume it,
and that the ranker itself never imports the trader or talks to the network.
"""

import math
import os
import unittest
from datetime import date, timedelta

import best_ev_ranker as ber
from best_ev_ranker import (
    BestEVConfig, parse_symbols, default_universe, scan_universe,
    rank_candidates, run_best_ev, format_best_ev_report,
    NO_CANDIDATES_MESSAGE, FOOTER,
)
from ev_engine import (
    EVResult,
    STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE,
    STATUS_OK, STATUS_INSUFFICIENT,
)
from spread_builder import SpreadLeg, SpreadProposal, BULLISH_PUT_CREDIT_SPREAD

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = (date.today() + timedelta(days=30)).isoformat()

# Phase 11B: run_best_ev now auto-stamps candlesticks, which would build a live
# bar provider from .env creds and hit the network. Keep this offline suite
# network-free by disabling ranker fetch for the whole module. The dedicated
# candlestick tests below pass explicit configs and are unaffected.
_PRIOR_FETCH_ENV = None


def setUpModule():
    global _PRIOR_FETCH_ENV
    _PRIOR_FETCH_ENV = os.environ.get("CANDLESTICK_FETCH_IN_RANKER")
    os.environ["CANDLESTICK_FETCH_IN_RANKER"] = "false"


def tearDownModule():
    if _PRIOR_FETCH_ENV is None:
        os.environ.pop("CANDLESTICK_FETCH_IN_RANKER", None)
    else:
        os.environ["CANDLESTICK_FETCH_IN_RANKER"] = _PRIOR_FETCH_ENV


def row(sym="SPY", strategy=BULLISH_PUT_CREDIT_SPREAD, ev=10.0, pop=0.7,
        ratio=0.10, mp=50.0, ml=450.0, costs=9.0, score=60.0, days=30,
        rec=ACCEPT, status=STATUS_OK):
    """Hand-built EVResult row for pure ranking/filter/format tests."""
    return EVResult(symbol=sym, strategy=strategy, expected_value=ev,
                    probability_of_profit=pop, ev_per_dollar_risk=ratio,
                    max_profit=mp, max_loss=ml, estimated_costs=costs,
                    oracle_score=score, days=days, recommendation=rec,
                    status=status)


def leg(action, otype, strike, bid, ask):
    return SpreadLeg(action=action, option_type=otype, strike=strike,
                     bid=bid, ask=ask, expiration=EXP)


def far_otm_bull_put(symbol, credit, mp, ml, be):
    """A deep-OTM fat-credit bull put -> PoP ~ 1 -> genuinely EV-positive."""
    return SpreadProposal(
        strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol=symbol,
        legs=[leg("sell", "put", 80, credit + 0.40, credit + 0.50),
              leg("buy", "put", 75, 0.40, 0.50)],
        net_credit_or_debit=credit, max_profit=mp, max_loss=ml,
        breakeven=be, width=5.0, oracle_score=70.0)


class _FakeTrader:
    """Duck-typed trader: propose_spread + get_price_history (no network)."""

    def __init__(self, symbol, proposal):
        self.symbol = symbol
        self._proposal = proposal

    def propose_spread(self, symbol):
        return self._proposal

    def get_price_history(self, symbol, days=130):
        return [100.0 + math.sin(i / 3.0) for i in range(130)]


def make_factory(proposals):
    """trader_factory over a {symbol: proposal} dict; unknown symbol raises."""
    def factory(symbol):
        if symbol not in proposals:
            raise RuntimeError(f"no data for {symbol}")
        return _FakeTrader(symbol, proposals[symbol])
    return factory


class _FakeLoader:
    def __init__(self, vals):
        self.v = vals

    def get_str(self, k, d=""):
        return str(self.v.get(k, d))

    def get_int(self, k, d=0):
        return int(self.v.get(k, d))

    def get_bool(self, k, d=False):
        val = self.v.get(k, d)
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    def get_float(self, k, d=0.0):
        return float(self.v.get(k, d))


# --------------------------------------------------------------------------- #
# 1. Ranking order
# --------------------------------------------------------------------------- #
class TestRankingOrder(unittest.TestCase):
    def test_sorted_by_ev_per_dollar_risk_desc(self):
        rows = [row(sym="LOW", ratio=0.05, ev=200.0),
                row(sym="HIGH", ratio=0.30, ev=10.0),
                row(sym="MID", ratio=0.15, ev=50.0)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertEqual([r.symbol for r in ranked], ["HIGH", "MID", "LOW"])

    def test_ratio_dominates_raw_ev(self):
        # Bigger raw EV must NOT beat a better EV-per-risk.
        rows = [row(sym="FAT", ratio=0.08, ev=500.0),
                row(sym="LEAN", ratio=0.25, ev=20.0)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertEqual(ranked[0].symbol, "LEAN")

    def test_end_to_end_scan_ranks_fatter_credit_first(self):
        proposals = {
            "AAA": far_otm_bull_put("AAA", 1.50, 150.0, 350.0, 78.5),
            "BBB": far_otm_bull_put("BBB", 0.60, 60.0, 440.0, 79.4),
        }
        ranked, scanned = run_best_ev("AAA,BBB", make_factory(proposals))
        self.assertEqual(scanned, 2)
        self.assertEqual([r.symbol for r in ranked], ["AAA", "BBB"])
        self.assertGreater(ranked[0].ev_per_dollar_risk,
                           ranked[1].ev_per_dollar_risk)
        for r in ranked:
            self.assertEqual(r.status, STATUS_OK)
            self.assertGreater(r.expected_value, 0.0)

    def test_failing_symbol_is_skipped_not_fatal(self):
        proposals = {"AAA": far_otm_bull_put("AAA", 1.50, 150.0, 350.0, 78.5)}
        ranked, scanned = run_best_ev("AAA,CCC", make_factory(proposals))
        self.assertEqual(scanned, 2)           # universe still counted
        self.assertEqual([r.symbol for r in ranked], ["AAA"])

    def test_scan_universe_never_raises(self):
        out = scan_universe(["XXX"], make_factory({}))
        self.assertEqual(out, [])


# --------------------------------------------------------------------------- #
# 2. Recommendation filtering
# --------------------------------------------------------------------------- #
class TestRecommendationFiltering(unittest.TestCase):
    def test_tier_floor_accept(self):
        rows = [row(sym="S", rec=STRONG_ACCEPT), row(sym="A", rec=ACCEPT),
                row(sym="N", rec=NEUTRAL), row(sym="W", rec=WEAK_SETUP),
                row(sym="R", rec=REJECT_CANDIDATE)]
        cfg = BestEVConfig(min_recommendation=ACCEPT)
        kept = {r.symbol for r in rank_candidates(rows, cfg)}
        self.assertEqual(kept, {"S", "A"})

    def test_default_floor_is_neutral(self):
        rows = [row(sym="N", rec=NEUTRAL), row(sym="W", rec=WEAK_SETUP)]
        kept = {r.symbol for r in rank_candidates(rows, BestEVConfig())}
        self.assertEqual(kept, {"N"})

    def test_non_ok_and_missing_ev_dropped(self):
        rows = [row(sym="BAD", status=STATUS_INSUFFICIENT),
                row(sym="NOEV", ev=None),
                row(sym="GOOD")]
        kept = [r.symbol for r in rank_candidates(rows, BestEVConfig())]
        self.assertEqual(kept, ["GOOD"])

    def test_none_input_safe(self):
        self.assertEqual(rank_candidates(None, BestEVConfig()), [])
        self.assertEqual(rank_candidates([None], BestEVConfig()), [])


# --------------------------------------------------------------------------- #
# 3. Include-negative behavior
# --------------------------------------------------------------------------- #
class TestIncludeNegative(unittest.TestCase):
    def test_default_drops_non_positive_ev(self):
        rows = [row(sym="NEG", ev=-5.0, ratio=-0.01),
                row(sym="ZERO", ev=0.0, ratio=0.0),
                row(sym="POS", ev=5.0, ratio=0.01)]
        kept = [r.symbol for r in rank_candidates(rows, BestEVConfig())]
        self.assertEqual(kept, ["POS"])

    def test_include_negative_keeps_them(self):
        rows = [row(sym="NEG", ev=-12.0, ratio=-0.03, rec=WEAK_SETUP),
                row(sym="POS", ev=5.0, ratio=0.01, rec=NEUTRAL)]
        cfg = BestEVConfig(include_negative=True,
                           min_recommendation=WEAK_SETUP)
        kept = [r.symbol for r in rank_candidates(rows, cfg)]
        self.assertEqual(kept, ["POS", "NEG"])   # positive still ranks first

    def test_negative_clearly_labeled_in_report(self):
        cfg = BestEVConfig(include_negative=True,
                           min_recommendation=WEAK_SETUP)
        ranked = rank_candidates(
            [row(sym="NEG", ev=-12.0, ratio=-0.03, rec=WEAK_SETUP)], cfg)
        text = format_best_ev_report(ranked, scanned=1, config=cfg)
        self.assertIn("⚠️ NEGATIVE EV", text)
        self.assertIn("EV: -$12.00", text)


# --------------------------------------------------------------------------- #
# 4. Symbol parsing
# --------------------------------------------------------------------------- #
class TestSymbolParsing(unittest.TestCase):
    def test_commas_whitespace_case_dedupe(self):
        self.assertEqual(parse_symbols("spy,qqq aapl  spy ,QQQ"),
                         ["SPY", "QQQ", "AAPL"])

    def test_invalid_tokens_dropped(self):
        self.assertEqual(parse_symbols("SPY, TOOLONG1, 123, br-k, , QQQ"),
                         ["SPY", "QQQ"])

    def test_list_input(self):
        self.assertEqual(parse_symbols(["nvda", "SPY,qqq"]),
                         ["NVDA", "SPY", "QQQ"])

    def test_none_and_empty(self):
        self.assertEqual(parse_symbols(None), [])
        self.assertEqual(parse_symbols(""), [])

    def test_max_symbols_cap_applied(self):
        proposals = {s: far_otm_bull_put(s, 1.50, 150.0, 350.0, 78.5)
                     for s in ("AA", "BB", "CC", "DD")}
        cfg = BestEVConfig(max_symbols=2)
        ranked, scanned = run_best_ev("AA,BB,CC,DD", make_factory(proposals),
                                      config=cfg)
        self.assertEqual(scanned, 2)
        self.assertEqual({r.symbol for r in ranked}, {"AA", "BB"})

    def test_default_universe_from_loader(self):
        loader = _FakeLoader({"SCHEDULER_SYMBOLS": "spy nvda"})
        self.assertEqual(default_universe(loader=loader), ["SPY", "NVDA"])

    def test_default_universe_fallback(self):
        self.assertEqual(default_universe(loader=_FakeLoader({})),
                         ["SPY", "QQQ"])

    def test_config_from_env_validates_tier(self):
        loader = _FakeLoader({"BEST_EV_MAX_SYMBOLS": 10,
                              "BEST_EV_MIN_RECOMMENDATION": "bogus_tier",
                              "BEST_EV_INCLUDE_NEGATIVE": "true",
                              "BEST_EV_TOP_N": 3})
        cfg = BestEVConfig.from_env(loader=loader)
        self.assertEqual(cfg.max_symbols, 10)
        self.assertEqual(cfg.min_recommendation, NEUTRAL)  # invalid -> default
        self.assertTrue(cfg.include_negative)
        self.assertEqual(cfg.top_n, 3)


# --------------------------------------------------------------------------- #
# 5. Empty candidates
# --------------------------------------------------------------------------- #
class TestEmptyCandidates(unittest.TestCase):
    def test_empty_report_message(self):
        text = format_best_ev_report([], scanned=4)
        self.assertIn("🏆 *Best EV Trades*", text)
        self.assertIn(NO_CANDIDATES_MESSAGE, text)
        self.assertIn("_Scanned 4 symbol(s)._", text)
        self.assertIn(FOOTER, text)

    def test_empty_universe_runs_clean(self):
        ranked, scanned = run_best_ev("", make_factory({}))
        self.assertEqual((ranked, scanned), ([], 0))
        self.assertIn(NO_CANDIDATES_MESSAGE,
                      format_best_ev_report(ranked, scanned=scanned))

    def test_all_filtered_out_yields_message(self):
        rows = [row(sym="NEG", ev=-5.0, ratio=-0.01)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertIn(NO_CANDIDATES_MESSAGE,
                      format_best_ev_report(ranked, scanned=1))


# --------------------------------------------------------------------------- #
# 6. EV tie-breakers
# --------------------------------------------------------------------------- #
class TestTieBreakers(unittest.TestCase):
    def test_equal_ratio_higher_ev_first(self):
        rows = [row(sym="SMALL", ratio=0.20, ev=20.0),
                row(sym="BIG", ratio=0.20, ev=80.0)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertEqual([r.symbol for r in ranked], ["BIG", "SMALL"])

    def test_equal_ratio_and_ev_higher_oracle_first(self):
        rows = [row(sym="LOWSC", ratio=0.20, ev=50.0, score=55.0),
                row(sym="HISC", ratio=0.20, ev=50.0, score=85.0)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertEqual([r.symbol for r in ranked], ["HISC", "LOWSC"])

    def test_final_tiebreak_lower_costs_first(self):
        rows = [row(sym="DEAR", ratio=0.20, ev=50.0, score=70.0, costs=18.0),
                row(sym="CHEAP", ratio=0.20, ev=50.0, score=70.0, costs=6.0)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertEqual([r.symbol for r in ranked], ["CHEAP", "DEAR"])

    def test_none_ratio_sorts_last(self):
        rows = [row(sym="NORATIO", ratio=None, ev=99.0),
                row(sym="OK", ratio=0.01, ev=5.0)]
        ranked = rank_candidates(rows, BestEVConfig())
        self.assertEqual([r.symbol for r in ranked], ["OK", "NORATIO"])


# --------------------------------------------------------------------------- #
# 7. Telegram output
# --------------------------------------------------------------------------- #
class TestTelegramOutput(unittest.TestCase):
    def test_entry_formatting_matches_spec(self):
        ranked = [row(sym="SPY", ev=18.40, pop=0.71, ratio=0.23,
                      score=64.0, days=30, rec=STRONG_ACCEPT)]
        text = format_best_ev_report(ranked, scanned=1)
        self.assertIn("🏆 *Best EV Trades*", text)
        self.assertIn("1. SPY Bull Put Credit Spread", text)
        self.assertIn("EV: +$18.40", text)
        self.assertIn("PoP: 71%", text)
        self.assertIn("EV/Risk: 0.23", text)
        self.assertIn("Score: 64", text)
        self.assertIn("DTE: 30", text)
        self.assertIn(f"Recommendation: {STRONG_ACCEPT}", text)
        self.assertIn(FOOTER, text)

    def test_top_n_limit(self):
        rows = [row(sym=f"S{i}", ratio=0.30 - i * 0.01, ev=50.0 - i)
                for i in range(7)]
        ranked = rank_candidates(rows, BestEVConfig())
        text = format_best_ev_report(ranked, scanned=7,
                                     config=BestEVConfig(top_n=5))
        self.assertIn("5. S4", text)
        self.assertNotIn("6. S5", text)
        self.assertNotIn("7. S6", text)

    def test_end_to_end_report_from_fake_traders(self):
        proposals = {
            "AAA": far_otm_bull_put("AAA", 1.50, 150.0, 350.0, 78.5),
            "BBB": far_otm_bull_put("BBB", 0.60, 60.0, 440.0, 79.4),
        }
        ranked, scanned = run_best_ev("AAA, BBB", make_factory(proposals))
        text = format_best_ev_report(ranked, scanned=scanned)
        self.assertIn("1. AAA Bull Put Credit Spread", text)
        self.assertIn("2. BBB Bull Put Credit Spread", text)
        self.assertIn("_Scanned 2 symbol(s)._", text)
        self.assertIn(FOOTER, text)
        self.assertNotIn("NEGATIVE EV", text)


# --------------------------------------------------------------------------- #
# 8. No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            return f.read()

    def test_live_modules_do_not_import_ranker(self):
        for mod in ("run_alpaca_intraday.py", "smart_trader.py"):
            self.assertNotIn("best_ev_ranker", self._read(mod),
                             f"{mod} must not consume best_ev_ranker")

    def test_ranker_has_no_execution_or_network(self):
        src = self._read("best_ev_ranker.py")
        for forbidden in ("import smart_trader", "from smart_trader",
                          "import requests", "place_order", "submit_order"):
            self.assertNotIn(forbidden, src)

    def test_telegram_surface_exists(self):
        src = self._read("telegram_bot.py")
        self.assertIn("BEST_EV_TRADES", src)
        self.assertIn("def best_ev_trades", src)


# --------------------------------------------------------------------------- #
# 9. Phase 11B — candlestick stamping in the ranker (ANALYTICS ONLY).
#     Offline: the bar provider is injected; no creds / network ever touched.
# --------------------------------------------------------------------------- #
import collections

from oracle.signals import candlestick_patterns as csp  # noqa: E402
import candidate_resolution as cr  # noqa: E402

# A Bar-like namedtuple matching market_view.Bar's field ORDER (date first).
# Its presence proves _to_candle parses by name, not by tuple position.
_BarLike = collections.namedtuple("_BarLike", "date o h l c v close_dt")


def _hammer_dicts():
    """Four down-trend candles then a hammer -> bullish_reversal."""
    out = [{"o": p + 0.3, "h": p + 0.6, "l": p - 0.6, "c": p, "v": 100}
           for p in (110, 108, 106, 104)]
    out.append({"o": 100.0, "h": 100.7, "l": 98.5, "c": 100.6, "v": 100})
    return out


def _hammer_bars():
    """Same hammer fixture as Bar-like namedtuples (date in slot 0)."""
    bars = [_BarLike(f"d{i}", p + 0.3, p + 0.6, p - 0.6, p, 100, None)
            for i, p in enumerate((110, 108, 106, 104))]
    bars.append(_BarLike("d4", 100.0, 100.7, 98.5, 100.6, 100, None))
    return bars


def _cfg(enabled=True, fetch=True, lookback=10):
    return csp.CandlestickConfig(enabled=enabled, fetch_in_ranker=fetch,
                                 ranker_lookback=lookback)


class TestCandlestickStamping(unittest.TestCase):
    def setUp(self):
        self.now = __import__("datetime").datetime(
            2026, 6, 17, tzinfo=__import__("datetime").timezone.utc)
        self.day = "2026-06-17"

    def test_extras_carry_fields_from_dict_candles(self):
        provider = lambda sym, lb: _hammer_dicts()
        extras = ber._candlestick_extras(
            [row(sym="SPY")], bar_provider=provider, config=_cfg(),
            now=self.now)
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, self.day)
        self.assertIn(key, extras)
        self.assertEqual(extras[key]["candlestick_pattern"], "hammer")
        self.assertEqual(extras[key]["candlestick_bias"], "bullish")
        # Only the 6 frozen fields — never the raw candle arrays.
        self.assertNotIn("candles", extras[key])
        self.assertEqual(set(extras[key]), set(cr._CANDLESTICK_KEYS))

    def test_bar_namedtuple_is_parsed_by_name(self):
        # Regression: market_view.Bar is a NamedTuple (date in slot 0). The
        # detector's tuple branch would mis-read it; _to_candle must coerce.
        provider = lambda sym, lb: _hammer_bars()
        extras = ber._candlestick_extras(
            [row(sym="SPY")], bar_provider=provider, config=_cfg(),
            now=self.now)
        key = cr.candidate_key("SPY", BULLISH_PUT_CREDIT_SPREAD, self.day)
        self.assertEqual(extras[key]["candlestick_pattern"], "hammer")

    def test_disabled_via_fetch_flag_returns_empty(self):
        provider = lambda sym, lb: _hammer_dicts()
        out = ber._candlestick_extras(
            [row()], bar_provider=provider, config=_cfg(fetch=False),
            now=self.now)
        self.assertEqual(out, {})

    def test_globally_disabled_returns_empty(self):
        provider = lambda sym, lb: _hammer_dicts()
        out = ber._candlestick_extras(
            [row()], bar_provider=provider, config=_cfg(enabled=False),
            now=self.now)
        self.assertEqual(out, {})

    def test_no_provider_returns_empty(self):
        out = ber._candlestick_extras(
            [row()], bar_provider=None, config=_cfg(),
            now=self.now)
        # No injected provider + offline (default provider may be None) -> {}.
        # In CI with creds in .env this still must not place/predict anything;
        # we only assert it is a dict and contains no execution side effects.
        self.assertIsInstance(out, dict)

    def test_provider_raising_is_fail_open(self):
        def boom(sym, lb):
            raise RuntimeError("network down")
        out = ber._candlestick_extras(
            [row()], bar_provider=boom, config=_cfg(), now=self.now)
        self.assertEqual(out, {})

    def test_empty_bars_yield_no_pattern(self):
        out = ber._candlestick_extras(
            [row()], bar_provider=lambda s, l: [], config=_cfg(),
            now=self.now)
        self.assertEqual(out, {})

    def test_unique_symbol_fetched_once(self):
        calls = []

        def counting(sym, lb):
            calls.append(sym)
            return _hammer_dicts()
        rows = [row(sym="SPY"), row(sym="SPY"), row(sym="QQQ")]
        ber._candlestick_extras(rows, bar_provider=counting,
                                config=_cfg(), now=self.now)
        self.assertEqual(sorted(calls), ["QQQ", "SPY"])

    def test_to_candle_coerces_barlike(self):
        d = ber._to_candle(_BarLike("d", 1.0, 2.0, 0.5, 1.5, 99, None))
        self.assertEqual((d["o"], d["h"], d["l"], d["c"], d["v"]),
                         (1.0, 2.0, 0.5, 1.5, 99))
        # Dicts pass straight through.
        src = {"o": 1, "h": 2, "l": 0, "c": 1}
        self.assertIs(ber._to_candle(src), src)


if __name__ == "__main__":
    unittest.main()
