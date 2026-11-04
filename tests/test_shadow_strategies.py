"""
Tests for the shadow strategy engine (shadow_strategies.py) and its
counterfactual resolver (research_outcomes.OutcomeReconciler.resolve_shadow).

Coverage:
  * Simons mean reversion  - oversold -> CALL, overbought -> PUT, flat -> None,
                             below-threshold logged rejected, top-K ranking
  * Soros momentum+volume  - accept, PUT direction, weak volume, exhaustion,
                             missing volume, no-signal -> None
  * Runner                 - rows are kind='shadow_candidate' with the right
                             strategy_version; data-fn errors never propagate
  * Wiring                 - flag off -> no runner, zero rows; flag on with a
                             ledger-bearing trader -> runner constructed
  * Resolver               - simulated outcome with correct pnl sign for CALL
                             and PUT; idempotent re-runs; baseline analytics
                             never see shadow rows

Offline by design: tmp_path SQLite, injected fakes, no creds, no network.
"""

from datetime import datetime, timezone

from research_ledger import ResearchLedger
from research_outcomes import OutcomeReconciler
from shadow_strategies import (
    ShadowStrategyRunner,
    SimonsMeanReversion,
    SorosMomentumVolume,
    build_strategies,
)

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
FLAT = [100.0] * 30
OVERSOLD = [100.0 + 0.05 * i for i in range(25)] + [101, 100, 97, 94, 90]
OVERBOUGHT = [100.0 - 0.05 * i for i in range(25)] + [99, 100, 103, 106, 110]
MOMO = [100.0 * (1.006 ** i) for i in range(30)]           # steady up-trend
MOMO_DOWN = [100.0 * (0.994 ** i) for i in range(30)]      # steady down-trend
MOMO_VOL = [1_000_000.0] * 29 + [2_000_000.0]              # volume spike
WEAK_VOL = [1_000_000.0] * 30
EXHAUSTED = [100.0] * 27 + [104, 109, 115]                 # 3-day blow-off


def ctx(sym, closes, volumes=None, price=None):
    return {"symbol": sym, "closes": list(closes),
            "volumes": list(volumes) if volumes else None,
            "underlying_price": price if price is not None else closes[-1],
            "regime": "test", "baseline_candidate": None}


def make_ledger(tmp_path, name="shadow.db"):
    return ResearchLedger(str(tmp_path / name))


def shadow_entry(led, did, underlying="SPY", price=100.0, direction="CALL",
                 variant="simons_meanrev",
                 event_at="2026-10-01T14:30:00+00:00", accepted=True):
    led.record_snapshot(
        decision_id=did, kind="shadow_candidate", accepted=accepted,
        symbol=underlying, underlying=underlying, underlying_price=price,
        direction=direction, score=2.0,
        strategy_version=f"shadow:{variant}",
        extra={"shadow_variant": variant}, event_at=event_at)


NOW = datetime(2026, 10, 3, 14, 30, tzinfo=timezone.utc)  # 2 days after entry


# --------------------------------------------------------------------------- #
# Simons mean reversion
# --------------------------------------------------------------------------- #
class TestSimons:
    def test_oversold_emits_accepted_call(self):
        c = SimonsMeanReversion({"window": 20, "z_entry": 1.5}).evaluate(
            ctx("OV", OVERSOLD))
        assert c is not None and c["accepted"] and c["direction"] == "CALL"
        assert c["underlying_price"] == OVERSOLD[-1]
        assert c["signals"]["z"] <= -1.5

    def test_overbought_emits_accepted_put(self):
        c = SimonsMeanReversion({"window": 20, "z_entry": 1.5}).evaluate(
            ctx("OB", OVERBOUGHT))
        assert c is not None and c["accepted"] and c["direction"] == "PUT"
        assert c["signals"]["z"] >= 1.5

    def test_flat_series_not_logged(self):
        assert SimonsMeanReversion({"window": 20}).evaluate(
            ctx("FLAT", FLAT)) is None

    def test_short_history_not_logged(self):
        assert SimonsMeanReversion({"window": 20}).evaluate(
            ctx("SHORT", OVERSOLD[:10])) is None

    def test_passing_half_z_but_below_entry_logged_rejected(self):
        # z_entry pushed high: the oversold fixture clears |z|>=0.5 but not
        # the entry bar -> logged as a rejected row, not dropped.
        c = SimonsMeanReversion({"window": 20, "z_entry": 10.0}).evaluate(
            ctx("OV", OVERSOLD))
        assert c is not None and not c["accepted"]
        assert c["reject_code"] == "shadow:below_z_entry"

    def test_top_k_downgrades_excess_passers(self):
        strat = SimonsMeanReversion({"window": 20, "z_entry": 1.5})
        cands = [strat.evaluate(ctx("OV", OVERSOLD)),
                 strat.evaluate(ctx("OB", OVERBOUGHT))]
        assert all(c and c["accepted"] for c in cands)
        out = ShadowStrategyRunner._apply_top_k(cands, top_k=1)
        accepted = [c for c in out if c["accepted"]]
        rejected = [c for c in out if not c["accepted"]]
        assert len(accepted) == 1 and len(rejected) == 1
        assert rejected[0]["reject_code"] == "shadow:not_top_k"
        # the keeper is the higher |z|
        assert accepted[0]["score"] >= rejected[0]["score"]


# --------------------------------------------------------------------------- #
# Soros momentum + volume
# --------------------------------------------------------------------------- #
class TestSoros:
    def test_momentum_with_volume_accepted_call(self):
        c = SorosMomentumVolume().evaluate(ctx("MOMO", MOMO, MOMO_VOL))
        assert c is not None and c["accepted"] and c["direction"] == "CALL"
        assert c["signals"]["vol_ratio"] > 1.2

    def test_down_momentum_is_put(self):
        c = SorosMomentumVolume().evaluate(ctx("DN", MOMO_DOWN, MOMO_VOL))
        assert c is not None and c["direction"] == "PUT"

    def test_weak_volume_rejected(self):
        c = SorosMomentumVolume().evaluate(ctx("WEAK", MOMO, WEAK_VOL))
        assert c is not None and not c["accepted"]
        assert c["reject_code"] == "shadow:weak_volume"

    def test_exhaustion_rejected(self):
        exh_vol = [1_000_000.0] * 29 + [3_000_000.0]
        c = SorosMomentumVolume().evaluate(ctx("EXH", EXHAUSTED, exh_vol))
        assert c is not None and not c["accepted"]
        assert c["reject_code"] == "shadow:exhausted"

    def test_missing_volume_rejected_not_dropped(self):
        c = SorosMomentumVolume().evaluate(ctx("NOVOL", MOMO, None))
        assert c is not None and not c["accepted"]
        assert c["reject_code"] == "shadow:no_volume_data"

    def test_no_momentum_not_logged(self):
        assert SorosMomentumVolume().evaluate(
            ctx("FLAT", FLAT, MOMO_VOL)) is None


# --------------------------------------------------------------------------- #
# Runner (real temp ledger)
# --------------------------------------------------------------------------- #
class TestRunner:
    def _data(self):
        return {"OV": {"closes": OVERSOLD, "volumes": WEAK_VOL},
                "OB": {"closes": OVERBOUGHT, "volumes": WEAK_VOL},
                "MOMO": {"closes": MOMO, "volumes": MOMO_VOL}}

    def test_rows_have_shadow_kind_and_variant(self, tmp_path):
        led = make_ledger(tmp_path)
        data = self._data()
        runner = ShadowStrategyRunner(
            led,
            [SimonsMeanReversion({"window": 20, "z_entry": 1.5, "top_k": 3}),
             SorosMomentumVolume()],
            fetch_bars=lambda s, d: data.get(s),
            fetch_price=lambda s: data[s]["closes"][-1] if s in data else None)
        s = runner.run_cycle(["OV", "OB", "MOMO"], [])
        runner.drain()
        runner.close()
        assert s["submitted"] >= 3 and s["errors"] == 0
        rows = led.snapshots(kind="shadow_candidate")
        assert rows
        versions = {r["strategy_version"] for r in rows}
        assert versions == {"shadow:simons_meanrev", "shadow:soros_momentum"}
        # every row carries an underlying price (resolver depends on it)
        assert all(r["underlying_price"] for r in rows)
        # nothing leaked into the baseline kind
        assert not led.snapshots(kind="candidate")
        led.close()

    def test_data_fn_errors_never_propagate(self, tmp_path):
        led = make_ledger(tmp_path)
        data = self._data()

        def boom_bars(sym, days):
            if sym == "BOOM":
                raise RuntimeError("bars blew up")
            return data.get(sym)

        def boom_price(sym):
            raise RuntimeError("price blew up")   # every price call fails

        runner = ShadowStrategyRunner(
            led, [SimonsMeanReversion({"window": 20, "z_entry": 1.5})],
            fetch_bars=boom_bars, fetch_price=boom_price)
        s = runner.run_cycle(["OV", "BOOM", "MISSING"], [])
        runner.drain()
        runner.close()
        assert s["errors"] >= 1            # counted, never raised
        assert runner.errors >= 1
        acct = runner.accounting()
        assert acct["cycles"] == 1
        led.close()

    def test_build_strategies_selection(self):
        both = build_strategies(["simons_meanrev", "soros_momentum"])
        assert {s.variant for s in both} == {"simons_meanrev", "soros_momentum"}
        one = build_strategies(["soros_momentum", "unknown_thing"])
        assert [s.variant for s in one] == ["soros_momentum"]
        default = build_strategies(None)
        assert {s.variant for s in default} == {"simons_meanrev",
                                                "soros_momentum"}


# --------------------------------------------------------------------------- #
# Scheduler wiring (flag gate)
# --------------------------------------------------------------------------- #
class _FakeCapture:
    def __init__(self, ledger):
        self._ledger = ledger


class _FakeTrader:
    def __init__(self, ledger=None):
        self.research_capture = _FakeCapture(ledger) if ledger else None

    def get_daily_bars(self, ticker=None, days=40):
        return {"closes": [], "volumes": []}

    def get_current_price(self, ticker=None):
        return None


class TestWiring:
    def _config(self, monkeypatch, env):
        monkeypatch.delenv("ENABLE_SHADOW_STRATEGIES", raising=False)
        monkeypatch.delenv("SHADOW_STRATEGY_LIST", raising=False)
        from run_alpaca_intraday import Config
        return Config(env=env, armed_override=False)

    def test_flag_off_no_runner_no_rows(self, tmp_path, monkeypatch):
        from run_alpaca_intraday import IntradayScheduler
        cfg = self._config(monkeypatch, {})
        assert cfg.enable_shadow_strategies is False
        led = make_ledger(tmp_path)
        sched = IntradayScheduler(cfg, _FakeTrader(led), pdt=None)
        assert sched.shadow_runner is None
        assert not led.snapshots(kind="shadow_candidate")
        led.close()

    def test_flag_on_constructs_runner(self, tmp_path, monkeypatch):
        from run_alpaca_intraday import IntradayScheduler
        cfg = self._config(monkeypatch, {"ENABLE_SHADOW_STRATEGIES": "1"})
        assert cfg.enable_shadow_strategies is True
        led = make_ledger(tmp_path)
        sched = IntradayScheduler(cfg, _FakeTrader(led), pdt=None)
        assert sched.shadow_runner is not None
        variants = set(sched.shadow_runner._comparators)
        assert variants == {"simons_meanrev", "soros_momentum"}
        sched.shadow_runner.close()
        led.close()

    def test_flag_on_without_ledger_stays_off(self, tmp_path, monkeypatch):
        from run_alpaca_intraday import IntradayScheduler
        cfg = self._config(monkeypatch, {"ENABLE_SHADOW_STRATEGIES": "1"})
        sched = IntradayScheduler(cfg, _FakeTrader(ledger=None), pdt=None)
        assert sched.shadow_runner is None


# --------------------------------------------------------------------------- #
# Counterfactual resolver
# --------------------------------------------------------------------------- #
class TestResolveShadow:
    def test_call_and_put_pnl_signs(self, tmp_path):
        led = make_ledger(tmp_path)
        shadow_entry(led, "sh-call", underlying="UP", price=100.0,
                     direction="CALL")
        shadow_entry(led, "sh-put", underlying="UP", price=100.0,
                     direction="PUT")
        rec = OutcomeReconciler(led, price_fn=lambda s: 110.0,
                                apply=True, now=NOW)
        s = rec.resolve_shadow(horizon_min=390.0)
        assert s["resolved"] == 2
        import json
        by_did = {}
        for r in led._rows("SELECT decision_id, payload_json FROM events "
                           "WHERE event_type='outcome'"):
            by_did[r["decision_id"]] = json.loads(r["payload_json"])
        assert by_did["sh-call"]["class"] == "simulated"
        assert by_did["sh-call"]["counterfactual_pnl_pct"] == 10.0
        assert by_did["sh-put"]["counterfactual_pnl_pct"] == -10.0
        assert by_did["sh-call"]["shadow_variant"] == "simons_meanrev"
        led.close()

    def test_idempotent_rerun_writes_nothing(self, tmp_path):
        led = make_ledger(tmp_path)
        shadow_entry(led, "sh-1", price=100.0)
        rec = OutcomeReconciler(led, price_fn=lambda s: 105.0,
                                apply=True, now=NOW)
        assert rec.resolve_shadow(horizon_min=390.0)["resolved"] == 1
        n_events = led.stats()["events"]
        s2 = OutcomeReconciler(led, price_fn=lambda s: 105.0,
                               apply=True, now=NOW).resolve_shadow(
                                   horizon_min=390.0)
        assert s2["resolved"] == 0
        assert led.stats()["events"] == n_events
        led.close()

    def test_not_due_until_horizon(self, tmp_path):
        led = make_ledger(tmp_path)
        shadow_entry(led, "sh-young", price=100.0,
                     event_at=NOW.isoformat())      # just logged
        rec = OutcomeReconciler(led, price_fn=lambda s: 105.0,
                                apply=True, now=NOW)
        s = rec.resolve_shadow(horizon_min=390.0)
        assert s["resolved"] == 0 and s["not_due"] == 1
        led.close()

    def test_rejected_shadow_rows_not_resolved(self, tmp_path):
        led = make_ledger(tmp_path)
        shadow_entry(led, "sh-rej", price=100.0, accepted=False)
        rec = OutcomeReconciler(led, price_fn=lambda s: 105.0,
                                apply=True, now=NOW)
        s = rec.resolve_shadow(horizon_min=390.0)
        assert s["resolved"] == 0 and s["checked"] == 0
        led.close()

    def test_baseline_analytics_never_see_shadow(self, tmp_path):
        from research_validation import observations_from_ledger
        led = make_ledger(tmp_path)
        shadow_entry(led, "sh-iso", price=100.0)
        OutcomeReconciler(led, price_fn=lambda s: 110.0, apply=True,
                          now=NOW).resolve_shadow(horizon_min=390.0)
        # even asking for simulated outcomes, baseline obs exclude shadow kind
        obs = observations_from_ledger(led, include_simulated=True)
        assert all(o.get("decision_id") != "sh-iso" for o in obs)
        assert obs == []
        led.close()
