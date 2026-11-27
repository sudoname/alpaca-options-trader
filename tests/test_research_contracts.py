"""Contract-level regression tests; no broker, torch or API dependencies."""
from datetime import datetime, timedelta, timezone

import pytest

from oracle.research_contracts import (
    CandidatePolicy, FeatureValue, ResearchCandidate, ResearchDecision,
    ResearchForecast, ResearchModel, ResearchSnapshot, SnapshotSource,
    validate_prediction,
)

T = datetime(2026, 11, 22, 15, 0, tzinfo=timezone.utc)


def snapshot(asset="future"):
    return ResearchSnapshot(
        "snap-1", asset, "MES-202612", T, T + timedelta(seconds=1),
        (FeatureValue("vwap_distance", 0.25, T), FeatureValue("volume", None, T)),
    )


def forecast():
    return ResearchForecast(
        "snap-1", "model-1", "sha256:abc", T + timedelta(seconds=2),
        0.2, 0.3, 0.5, "5m",
    )


def candidate():
    return ResearchCandidate("c1", "snap-1", "opening-range", "long", 1)


def test_contracts_are_instrument_agnostic():
    for asset in ("future", "option", "equity"):
        s = snapshot(asset)
        assert s.asset_class == asset
        assert s.feature_dict() == {"vwap_distance": 0.25, "volume": None}
        validate_prediction(s, candidate(), forecast())


def test_rejects_naive_time_and_leaked_feature():
    with pytest.raises(ValueError, match="timezone"):
        ResearchSnapshot("s", "future", "MES", T.replace(tzinfo=None), T, ())
    with pytest.raises(ValueError, match="unavailable"):
        ResearchSnapshot(
            "s", "future", "MES", T, T,
            (FeatureValue("future_price", 100, T + timedelta(seconds=1)),),
        )


def test_rejects_duplicate_and_nonfinite_features():
    f = FeatureValue("x", 1, T)
    with pytest.raises(ValueError, match="duplicate"):
        ResearchSnapshot("s", "future", "MES", T, T, (f, f))
    with pytest.raises(ValueError, match="finite"):
        FeatureValue("x", float("nan"), T)


def test_probability_sanity():
    with pytest.raises(ValueError, match="sum to one"):
        ResearchForecast("s", "m", "sha", T, 0.8, 0.5, 0.2, "5m")


def test_mismatched_ids_fail_closed():
    bad = ResearchCandidate("c", "wrong", "strategy", "short", 1)
    with pytest.raises(ValueError, match="identity"):
        validate_prediction(snapshot(), bad, forecast())


def test_pre_input_forecast_is_invalid():
    f = ResearchForecast(
        "snap-1", "model-1", "sha", T, 0.2, 0.3, 0.5, "5m",
    )
    with pytest.raises(ValueError, match="predates"):
        validate_prediction(snapshot(), candidate(), f)


def test_decision_fingerprint_and_enter_requirements():
    a = ResearchDecision("c1", "snap-1", "abstain", "missing quote")
    assert a.fingerprint() == ResearchDecision(
        "c1", "snap-1", "abstain", "missing quote"
    ).fingerprint()
    with pytest.raises(ValueError, match="requires"):
        ResearchDecision("c1", "snap-1", "enter", "edge detected")


def test_no_broker_or_model_framework_imports():
    import inspect
    import oracle.research_contracts as module

    source = inspect.getsource(module)
    for forbidden in ("import alpaca", "import ninja", "import torch", "import lightgbm",
                      "from oracle.execution", "from oracle.ml"):
        assert forbidden not in source


def test_protocols_are_runtime_checkable():
    class FakeModel:
        def predict(self, s):
            return forecast()

    assert isinstance(FakeModel(), ResearchModel)
