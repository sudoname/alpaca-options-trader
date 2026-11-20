"""Oracle 4.0.1 Stage B acceptance tests (verified model inference).

Master-prompt battery for the shared ForecastProvider interface:
  * registered model loaded and invoked in fixture (real logits/probs,
    ordered quantiles, RV, CIF, latency, input digest);
  * two intentionally different models yield different forecasts;
  * schema mismatch, missing weights, wrong hash, stale model, train
    overlap and missing features all produce structured abstention;
  * live/replay forecast parity on identical frozen input;
  * cache identity + versioned refresh; failures never cached;
  * shadow integration: bars-mode invocation, legacy-mode refusal
    (HISTORY_UNTIMESTAMPED), fail-open identity of challenger/baseline
    decisions when the model is misconfigured;
  * wall-clock benchmark recorded (not just successful exceptions).

Forecast-conditioned scenario distributions / challenger-rank alteration
are Stage C acceptance (valuation layer) and are intentionally absent here.
All hermetic: tiny untrained MLPs saved via the real save_model path.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

import pytest

from oracle.ml import mlp

pytestmark = pytest.mark.skipif(
    not mlp.TORCH_AVAILABLE, reason="torch unavailable")

if mlp.TORCH_AVAILABLE:
    import torch

from oracle.lab.purged_validation import Standardizer
from oracle.ml import bars as barsmod
from oracle.ml import forecast as fcmod
from oracle.ml import registry as regmod
from oracle.ml import shadow as shmod
from oracle.ml.bars import Bar, BarStore
from oracle.ml.schema import canonical_json

UTC = timezone.utc

# 2026-01-05 is a Monday under EST: RTH = 14:30-21:00 UTC.
S1_OPEN = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
AS_OF = datetime(2026, 1, 5, 15, 20, tzinfo=UTC)

# fixture models train on early-December sessions so AS_OF (Jan 5) sits
# inside the valid window: train_end 2025-12-04 .. stale after 2026-01-18
TRAIN_SESSIONS = ("2025-12-01", "2025-12-02")
VAL_SESSION = "2025-12-03"

FEAT_NAMES = ["dte_days", "quote_mid", "underlying_price"]
FEATS = {"dte_days": 1.0, "quote_mid": 1.05, "underlying_price": 100.5}


# --------------------------------------------------------------------------- #
# fixtures: tiny real MLP artifacts + registry
# --------------------------------------------------------------------------- #

def _fit_standardizer() -> Standardizer:
    std = Standardizer()
    std.fit([
        {"dte_days": 1.0, "quote_mid": 1.0, "underlying_price": 99.0},
        {"dte_days": 2.0, "quote_mid": 1.2, "underlying_price": 101.0},
        {"dte_days": 3.0, "quote_mid": 0.9, "underlying_price": 100.0},
    ], feature_names=FEAT_NAMES)
    return std


def _make_artifact(tmp_path, stem: str, seed: int,
                   train_sessions=TRAIN_SESSIONS,
                   val_session: str = VAL_SESSION,
                   temperature: float = 2.0):
    """Build + save one tiny untrained MLP through the real save path."""
    torch.manual_seed(seed)
    model = mlp.MultitaskMLP(2 * len(FEAT_NAMES), mlp.MLPConfig(hidden=(8, 8, 8)))
    out_stem = str(tmp_path / stem)
    mlp.save_model(
        model, out_stem,
        feature_columns=FEAT_NAMES + [f"{n}_missing" for n in FEAT_NAMES],
        standardizer_state=mlp.standardizer_state(_fit_standardizer()),
        temperature=temperature, seed=seed, git_sha=None,
        dataset_hash="fixture", train_sessions=list(train_sessions),
        val_session=val_session, extra={"horizon": "60m"})
    return out_stem + "_model.pt", out_stem + "_manifest.json"


def _register(regpath: str, model_id: str, pt: str, man: str,
              kind: str = "neural"):
    return regmod.register_model(regpath, model_id, kind=kind,
                                 artifact_path=pt, manifest_path=man,
                                 notes="stage-b fixture")


@pytest.fixture(autouse=True)
def _fresh_cache():
    fcmod.clear_cache()
    yield
    fcmod.clear_cache()


@pytest.fixture()
def rig(tmp_path):
    """One registered fixture model (seed 1) -> (registry_path, pt, man)."""
    pt, man = _make_artifact(tmp_path, "fix_a", seed=1)
    regpath = str(tmp_path / "registry.json")
    _register(regpath, "fix_a", pt, man)
    return regpath, pt, man


def _edit_registry(regpath: str, model_id: str, **updates):
    with open(regpath, encoding="utf-8") as f:
        data = json.load(f)
    data["models"][model_id].update(updates)
    with open(regpath, "w", encoding="utf-8") as f:
        json.dump(data, f)


# --------------------------------------------------------------------------- #
# load + invoke: real logits, calibrated probs, quantiles, RV, CIF, digest
# --------------------------------------------------------------------------- #

class TestLoadAndInvoke:
    def test_registered_model_loaded_and_invoked(self, rig):
        regpath, _pt, man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        assert p.feature_names == FEAT_NAMES
        assert p.temperature == 2.0
        assert p.horizon == "60m"
        fc = p.forecast(FEATS, as_of=AS_OF)
        d = fc.to_dict()
        assert d["available"] is True
        # real calibrated probabilities from actual logits
        assert len(fc.direction_logits) == 3
        s = fc.p_down + fc.p_neutral + fc.p_up
        assert abs(s - 1.0) < 1e-6
        assert all(0.0 < v < 1.0 for v in (fc.p_down, fc.p_neutral, fc.p_up))
        # ordered log-return quantiles over the ladder
        qvals = [fc.quantiles[f"{q:g}"] for q in p.quantile_ladder]
        assert qvals == sorted(qvals)
        assert fc.rv_forecast > 0.0
        # CIF per hazard cause, 7 bins each
        assert set(fc.cif) == {"stop", "target", "other_exit"}
        assert all(len(v) == 7 for v in fc.cif.values())
        # provenance + accounting
        assert fc.checksums["manifest"] == regmod.sha256_file(man)
        assert fc.n_features == 3 and fc.n_missing == 0
        assert len(fc.input_digest) == 64
        assert fc.latency_ms >= 0.0

    def test_missing_feature_accounting(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        fc = p.forecast({"dte_days": 1.0, "underlying_price": 100.5},
                        as_of=AS_OF)
        assert fc.n_missing == 1
        assert fc.missing_features == ["quote_mid"]

    def test_two_models_yield_different_forecasts(self, tmp_path):
        pt_a, man_a = _make_artifact(tmp_path, "m_a", seed=1)
        pt_b, man_b = _make_artifact(tmp_path, "m_b", seed=2)
        regpath = str(tmp_path / "registry.json")
        _register(regpath, "m_a", pt_a, man_a)
        _register(regpath, "m_b", pt_b, man_b)
        fa = fcmod.get_provider(regpath, "m_a").forecast(FEATS, as_of=AS_OF)
        fb = fcmod.get_provider(regpath, "m_b").forecast(FEATS, as_of=AS_OF)
        assert fa.direction_logits != fb.direction_logits
        assert (fa.p_up, fa.p_down) != (fb.p_up, fb.p_down)

    def test_input_digest_canonical_and_stable(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        fc1 = p.forecast(FEATS, as_of=AS_OF)
        fc2 = p.forecast(dict(FEATS), as_of=AS_OF)
        want = hashlib.sha256(
            canonical_json(dict(FEATS)).encode("utf-8")).hexdigest()
        assert fc1.input_digest == want == fc2.input_digest

    def test_parity_across_cache_clear_on_frozen_input(self, rig):
        """Live/replay parity: identical frozen input -> identical forecast
        after a full provider reload (fresh process simulation)."""
        regpath, _pt, _man = rig
        fc1 = fcmod.get_provider(regpath, "fix_a").forecast(FEATS, as_of=AS_OF)
        fcmod.clear_cache()
        fc2 = fcmod.get_provider(regpath, "fix_a").forecast(FEATS, as_of=AS_OF)
        assert fc1.direction_logits == fc2.direction_logits
        assert (fc1.p_down, fc1.p_neutral, fc1.p_up) == \
            (fc2.p_down, fc2.p_neutral, fc2.p_up)
        assert fc1.quantiles == fc2.quantiles
        assert fc1.rv_forecast == fc2.rv_forecast
        assert fc1.cif == fc2.cif
        assert fc1.input_digest == fc2.input_digest


# --------------------------------------------------------------------------- #
# verification chain: wrong hash / missing weights / schema mismatch refuse
# --------------------------------------------------------------------------- #

def _reason(regpath, model_id):
    with pytest.raises(fcmod.ForecastUnavailable) as ei:
        fcmod.get_provider(regpath, model_id)
    return ei.value


class TestVerificationChain:
    def test_tampered_weights_refused(self, rig):
        regpath, pt, _man = rig
        with open(pt, "ab") as f:
            f.write(b"\x00tamper")
        assert _reason(regpath, "fix_a").reason == "MODEL_INTEGRITY"

    def test_missing_weights_refused(self, rig):
        regpath, pt, _man = rig
        os.remove(pt)
        assert _reason(regpath, "fix_a").reason == "MODEL_INTEGRITY"

    def test_tampered_manifest_refused(self, rig):
        regpath, _pt, man = rig
        with open(man, "ab") as f:
            f.write(b"\n")
        assert _reason(regpath, "fix_a").reason == \
            "REGISTRY_CHECKSUM_MISMATCH"

    def test_manifest_missing_refused(self, rig):
        regpath, _pt, man = rig
        os.remove(man)
        assert _reason(regpath, "fix_a").reason == "MANIFEST_MISSING"

    def test_schema_mismatch_refused(self, rig, tmp_path):
        # same length (so d_in passes) but wrong missing-flag order
        regpath, _pt, man = rig
        with open(man, encoding="utf-8") as f:
            m = json.load(f)
        cols = m["feature_columns"]
        cols[4], cols[5] = cols[5], cols[4]
        bad = str(tmp_path / "fix_a_bad_manifest.json")  # same dir as .pt
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(m, f)
        _register(regpath, "fix_bad_schema",
                  os.path.join(str(tmp_path), m["model_file"]), bad)
        assert _reason(regpath, "fix_bad_schema").reason == "SCHEMA_INVALID"

    def test_artifact_checksum_cross_check(self, rig):
        regpath, _pt, man = rig
        _edit_registry(regpath, "fix_a", checksums={
            "manifest": regmod.sha256_file(man),
            "artifact": "0" * 64})
        assert _reason(regpath, "fix_a").reason == \
            "ARTIFACT_CHECKSUM_MISMATCH"

    def test_manifest_not_pinned_refused(self, rig):
        regpath, _pt, _man = rig
        _edit_registry(regpath, "fix_a", checksums={})
        assert _reason(regpath, "fix_a").reason == "MANIFEST_NOT_PINNED"

    def test_non_neural_kind_refused_not_mislabeled(self, rig, tmp_path):
        # a classical entry must refuse, never be served as neural inference
        regpath, pt, man = rig
        _register(regpath, "classical_x", pt, man, kind="classical")
        assert _reason(regpath, "classical_x").reason == "KIND_UNSUPPORTED"

    def test_registry_missing(self, tmp_path):
        assert _reason(str(tmp_path / "none.json"), "x").reason == \
            "REGISTRY_MISSING"

    def test_registry_entry_missing(self, rig):
        regpath, _pt, _man = rig
        assert _reason(regpath, "ghost").reason == "REGISTRY_ENTRY_MISSING"

    def test_model_id_unset(self, rig):
        regpath, _pt, _man = rig
        assert _reason(regpath, "").reason == "MODEL_ID_UNSET"


# --------------------------------------------------------------------------- #
# temporal guards + feature availability
# --------------------------------------------------------------------------- #

class TestTemporalGuards:
    def test_train_overlap_refused(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        with pytest.raises(fcmod.ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=datetime(2025, 12, 2, 16, 0, tzinfo=UTC))
        assert ei.value.reason == "TRAIN_OVERLAP"

    def test_stale_model_refused(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        with pytest.raises(fcmod.ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=datetime(2026, 3, 1, 16, 0, tzinfo=UTC))
        assert ei.value.reason == "MODEL_STALE"

    def test_naive_as_of_refused(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        with pytest.raises(fcmod.ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=datetime(2026, 1, 5, 15, 20))
        assert ei.value.reason == "INFERENCE_ERROR"

    def test_all_features_missing_refused(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        with pytest.raises(fcmod.ForecastUnavailable) as ei:
            p.forecast({}, as_of=AS_OF)
        assert ei.value.reason == "FEATURES_ALL_MISSING"
        d = ei.value.to_dict()
        assert d["available"] is False and d["model_id"] == "fix_a"


# --------------------------------------------------------------------------- #
# cache: loaded once outside scan loop, versioned refresh, failures uncached
# --------------------------------------------------------------------------- #

class TestCacheAndRefresh:
    def test_cache_identity(self, rig):
        regpath, _pt, _man = rig
        assert fcmod.get_provider(regpath, "fix_a") is \
            fcmod.get_provider(regpath, "fix_a")

    def test_versioned_refresh_on_registry_change(self, rig, tmp_path):
        regpath, _pt, _man = rig
        p1 = fcmod.get_provider(regpath, "fix_a")
        # a new artifact version lands under the same model id
        pt_b, man_b = _make_artifact(tmp_path, "fix_a_v2", seed=2)
        with open(man_b, encoding="utf-8") as f:
            art_b = json.load(f)["checksum_sha256"]
        _edit_registry(regpath, "fix_a", manifest_path=man_b, checksums={
            "manifest": regmod.sha256_file(man_b), "artifact": art_b})
        p2 = fcmod.get_provider(regpath, "fix_a")
        assert p2 is not p1
        assert p2.checksums["manifest"] == regmod.sha256_file(man_b)
        fa = p1.forecast(FEATS, as_of=AS_OF)
        fb = p2.forecast(FEATS, as_of=AS_OF)
        assert fa.direction_logits != fb.direction_logits

    def test_failures_never_cached(self, rig):
        regpath, _pt, man = rig
        with open(man, "rb") as f:
            orig = f.read()
        with open(man, "ab") as f:
            f.write(b"\n")
        with pytest.raises(fcmod.ForecastUnavailable):
            fcmod.get_provider(regpath, "fix_a")
        with open(man, "wb") as f:
            f.write(orig)
        p = fcmod.get_provider(regpath, "fix_a")  # recovers, no stale failure
        assert p.forecast(FEATS, as_of=AS_OF).rv_forecast > 0.0


# --------------------------------------------------------------------------- #
# shadow integration: actual invocation, legacy refusal, fail-open identity
# --------------------------------------------------------------------------- #

def _bar(ts: datetime, close: float) -> Bar:
    return Bar(symbol="SPY", resolution_s=60, ts_utc=ts, open=close,
               high=close, low=close, close=close, volume=1.0,
               session_date=barsmod.session_date_of(ts),
               rth=barsmod.is_rth(ts), source="test")


def _write_bars(store_path: str, n: int = 41):
    st = BarStore(store_path, readonly=False)
    st.insert_bars([_bar(S1_OPEN + timedelta(minutes=i),
                         100.0 + (i % 5) * 0.2) for i in range(n)])
    st.close()


def _cand():
    return {"sym": "SPY", "score": 1.0, "underlying_price": 100.5,
            "opt": {"symbol": "SPY260105C00100000", "bid": 1.0, "ask": 1.1,
                    "dte": 1}}


class TestShadowIntegration:
    def test_bars_mode_actually_invokes_registered_model(self, rig, tmp_path):
        regpath, _pt, _man = rig
        store = str(tmp_path / "bars.sqlite")
        _write_bars(store)
        cfg = shmod.ShadowConfig(enabled=True, model_id="fix_a",
                                 registry_path=regpath, max_scenarios=64,
                                 bars_db=store)
        rec = shmod.shadow_scan(
            [_cand()], scan_id="sb1", step_minutes=1.0, now=AS_OF, cfg=cfg,
            out_path=str(tmp_path / "out.jsonl"))
        inf = rec["model"]["inference"]
        assert inf["attempted"] is True and inf["loaded"] is True
        assert inf["n_forecasts"] == 1 and inf["n_abstained"] == 0
        assert inf["d_in"] == 6 and inf["forecast_ms_total"] > 0.0
        fc = rec["forecasts"]["sb1:SPY:0"]
        assert fc["available"] is True
        assert abs(fc["p_down"] + fc["p_neutral"] + fc["p_up"] - 1.0) < 1e-6
        assert len(fc["direction_logits"]) == 3
        assert len(fc["input_digest"]) == 64
        assert fc["latency_ms"] >= 0.0
        # written record round-trips as JSON
        with open(tmp_path / "out.jsonl", encoding="utf-8") as f:
            on_disk = json.loads(f.readline())
        assert on_disk["forecasts"]["sb1:SPY:0"]["available"] is True

    def test_legacy_grid_refuses_untimestamped_features(self, rig, tmp_path):
        regpath, _pt, _man = rig
        cfg = shmod.ShadowConfig(enabled=True, model_id="fix_a",
                                 registry_path=regpath, max_scenarios=32)
        rec = shmod.shadow_scan(
            [_cand()], scan_id="sb2",
            price_history_fn=lambda u, days=90: [100.0 + (i % 7) * 0.1
                                                 for i in range(80)],
            now=AS_OF, cfg=cfg, out_path=str(tmp_path / "out.jsonl"))
        inf = rec["model"]["inference"]
        assert inf["loaded"] is True
        assert inf["n_forecasts"] == 0 and inf["n_abstained"] == 1
        fc = rec["forecasts"]["sb2:SPY:0"]
        assert fc["available"] is False
        assert fc["reason"] == "HISTORY_UNTIMESTAMPED"

    def test_identity_metadata_alone_never_counts_as_inference(
            self, rig, tmp_path):
        # model id set but weights deleted: identity may resolve, but the
        # record must carry a structured abstention, not model evidence
        regpath, pt, _man = rig
        os.remove(pt)
        store = str(tmp_path / "bars.sqlite")
        _write_bars(store)
        cfg = shmod.ShadowConfig(enabled=True, model_id="fix_a",
                                 registry_path=regpath, max_scenarios=64,
                                 bars_db=store)
        rec = shmod.shadow_scan(
            [_cand()], scan_id="sb3", step_minutes=1.0, now=AS_OF, cfg=cfg,
            out_path=str(tmp_path / "out.jsonl"))
        inf = rec["model"]["inference"]
        assert inf["loaded"] is False and inf["n_forecasts"] == 0
        assert inf["forecast_unavailable"]["reason"] == "MODEL_INTEGRITY"
        assert rec["forecasts"]["sb3:SPY:0"]["available"] is False

    def test_fail_open_identical_decisions(self, rig, tmp_path):
        """A misconfigured/failing model cannot change challenger or
        baseline decisions: records differ only in forecast/inference."""
        regpath, _pt, _man = rig
        store = str(tmp_path / "bars.sqlite")
        _write_bars(store)
        common = dict(enabled=True, registry_path=regpath,
                      max_scenarios=64, bars_db=store)
        rec_off = shmod.shadow_scan(
            [_cand()], scan_id="same", step_minutes=1.0, now=AS_OF,
            cfg=shmod.ShadowConfig(model_id="", **common),
            out_path=str(tmp_path / "a.jsonl"))
        rec_bad = shmod.shadow_scan(
            [_cand()], scan_id="same", step_minutes=1.0, now=AS_OF,
            cfg=shmod.ShadowConfig(model_id="ghost_model", **common),
            out_path=str(tmp_path / "b.jsonl"))
        assert rec_off["challenger"] == rec_bad["challenger"]
        assert rec_off["baseline"] == rec_bad["baseline"]
        # Scenario numbers are byte-identical (CRN fail-open); only the
        # "forecast" provenance sub-key may differ (None vs unavailable).
        strip = lambda scen: {
            u: {k: v for k, v in m.items() if k != "forecast"}
            for u, m in scen.items()
        }
        assert strip(rec_off["scenarios"]) == strip(rec_bad["scenarios"])
        assert rec_off["scenarios"]["SPY"]["forecast"] is None
        assert rec_bad["scenarios"]["SPY"]["forecast"] == {
            "fallback_reason": "forecast_unavailable",
            "detail": "REGISTRY_ENTRY_MISSING",
        }
        assert rec_off["model"]["inference"]["reason"] == "MODEL_ID_UNSET"
        assert rec_bad["forecasts"]["same:SPY:0"]["reason"] == \
            "REGISTRY_ENTRY_MISSING"

    def test_disabled_hook_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ENABLE_ORACLE4_SHADOW", raising=False)
        assert shmod.maybe_shadow_scan([_cand()], scan_id="off") is None


# --------------------------------------------------------------------------- #
# wall-clock benchmark (recorded, generous bound -- not just no-exception)
# --------------------------------------------------------------------------- #

class TestBenchmark:
    def test_inference_wall_clock_recorded(self, rig):
        regpath, _pt, _man = rig
        p = fcmod.get_provider(regpath, "fix_a")
        n = 50
        t0 = time.perf_counter()
        lat = [p.forecast(FEATS, as_of=AS_OF).latency_ms for _ in range(n)]
        wall_ms = (time.perf_counter() - t0) * 1000.0
        mean_ms = sum(lat) / n
        print(f"\n[stage-b benchmark] {n} forecasts: wall {wall_ms:.1f} ms "
              f"total, mean model latency {mean_ms:.3f} ms")
        assert all(v >= 0.0 for v in lat)
        assert wall_ms < 30_000.0  # generous CI bound; typical ~1-10ms each
