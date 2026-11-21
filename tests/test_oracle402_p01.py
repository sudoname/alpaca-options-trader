"""Oracle 4.0.2 P0-1: trained-model -> valuation wiring tests.

Covers the new pipeline plumbing only; the heavy acceptance criteria
(forecast changes distributions / can flip rankings, checksum verification,
TRAIN_OVERLAP / MODEL_STALE refusals) are already pinned by
tests/test_oracle401_valuation.py and tests/test_oracle401_forecast.py.

Here we verify, with an empty candidate set (run_valuation is safe on it):
  * --forecast-max-staleness-days reaches get_provider as
    ``max_staleness_days`` and is echoed in the report's forecast section;
  * when no override is given, the provider's own default is untouched
    (kwarg absent) and the override is recorded as None;
  * a provider load failure is never silent: it lands in
    forecast.provider_note with an explicit reason, and the run still
    completes as labeled baseline.
"""
from __future__ import annotations

import oracle.ml.forecast as fc_mod
import oracle.ml.valuation as val
from oracle.ml.valuation import ValuationConfig, run_valuation


class _FakeProvider:
    """Minimal stand-in; never asked to forecast (zero candidates)."""


def _patch_empty_dataset(monkeypatch):
    monkeypatch.setattr(val, "load_candidates",
                        lambda path, sessions=None: [])
    monkeypatch.setattr(val, "load_underlying_obs", lambda path: {})


def test_staleness_override_reaches_provider_and_report(monkeypatch):
    _patch_empty_dataset(monkeypatch)
    calls = {}

    def fake_get_provider(registry_path, model_id, **kw):
        calls["registry_path"] = registry_path
        calls["model_id"] = model_id
        calls.update(kw)
        return _FakeProvider()

    monkeypatch.setattr(fc_mod, "get_provider", fake_get_provider)
    out = run_valuation(
        "unused.sqlite", ["2026-01-05"], ValuationConfig(), seed=7,
        registry_path="reg.json", model_id="mlp_fc_x",
        forecast_max_staleness_days=120)

    assert calls["registry_path"] == "reg.json"
    assert calls["model_id"] == "mlp_fc_x"
    assert calls["max_staleness_days"] == 120
    fc = out["forecast"]
    assert fc["registry"] == "reg.json"
    assert fc["model_id"] == "mlp_fc_x"
    assert fc["max_staleness_days_override"] == 120
    assert fc["provider_note"] is None


def test_no_override_leaves_provider_default(monkeypatch):
    _patch_empty_dataset(monkeypatch)
    calls = {}

    def fake_get_provider(registry_path, model_id, **kw):
        calls.update(kw)
        return _FakeProvider()

    monkeypatch.setattr(fc_mod, "get_provider", fake_get_provider)
    out = run_valuation(
        "unused.sqlite", ["2026-01-05"], ValuationConfig(), seed=7,
        registry_path="reg.json", model_id="mlp_fc_x")

    assert "max_staleness_days" not in calls
    assert out["forecast"]["max_staleness_days_override"] is None


def test_provider_failure_is_labeled_not_silent(monkeypatch):
    _patch_empty_dataset(monkeypatch)

    def boom(registry_path, model_id, **kw):
        raise RuntimeError("registry pin mismatch")

    monkeypatch.setattr(fc_mod, "get_provider", boom)
    out = run_valuation(
        "unused.sqlite", ["2026-01-05"], ValuationConfig(), seed=7,
        registry_path="reg.json", model_id="mlp_fc_x")

    note = out["forecast"]["provider_note"]
    assert note["available"] is False
    assert note["reason"] == "PROVIDER_ERROR"
    assert "registry pin mismatch" in note["detail"]
    # run completed as labeled baseline, nothing crashed
    assert out["counts"]["candidates"] == 0


def test_structured_unavailable_uses_to_dict(monkeypatch):
    _patch_empty_dataset(monkeypatch)

    class Unavailable(Exception):
        def to_dict(self):
            return {"available": False, "reason": "MODEL_STALE",
                    "detail": "train_end too old"}

    def boom(registry_path, model_id, **kw):
        raise Unavailable()

    monkeypatch.setattr(fc_mod, "get_provider", boom)
    out = run_valuation(
        "unused.sqlite", ["2026-01-05"], ValuationConfig(), seed=7,
        registry_path="reg.json", model_id="mlp_fc_x")

    assert out["forecast"]["provider_note"]["reason"] == "MODEL_STALE"


def test_no_model_requested_means_no_provider_lookup(monkeypatch):
    _patch_empty_dataset(monkeypatch)

    def boom(*a, **kw):  # would fail the test if ever called
        raise AssertionError("get_provider must not be called")

    monkeypatch.setattr(fc_mod, "get_provider", boom)
    out = run_valuation(
        "unused.sqlite", ["2026-01-05"], ValuationConfig(), seed=7)

    fc = out["forecast"]
    assert fc["model_id"] is None
    assert fc["provider_note"] is None
    assert fc["scan_modes"] == {}
