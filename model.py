"""
Supervised expected-net-return model (scaffold).

A deliberately simple, stdlib-only baseline: the predicted expected NET return
for a state is the historical mean NET P/L of completed episodes that landed in
the same discrete state bucket (the same `state_key` the RL layer uses). No
sklearn, no numpy. This exists to give an honest, leakage-resistant yardstick
for walk-forward evaluation; it does NOT feed the decision policy in this phase.

Design rules that keep it honest on thin data:
  * `train` returns `insufficient_data` below a sample floor instead of fitting
    noise, and stamps the feature_version it trained on.
  * `predict_expected_net_return` ABSTAINS (returns 0.0) on an unseen state, a
    too-thin bucket, or a feature_version mismatch. Abstention is explicit so
    coverage can be measured separately from accuracy.

An "episode" is a dict carrying at least: `state_key`, `net_pnl_pct`, and
(optionally) `feature_version` and `as_of`. These are exactly the columns the
episode store already records.
"""

import json
from typing import Dict, List, Optional

FEATURE_VERSION = "1.0.0"


def _net_pnl(ep: Dict) -> Optional[float]:
    val = ep.get("net_pnl_pct")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class ExpectedReturnModel:
    def __init__(
        self,
        feature_version: str = FEATURE_VERSION,
        min_state_samples: int = 2,
        min_total: int = 10,
    ):
        self.feature_version = feature_version
        self.min_state_samples = min_state_samples
        self.min_total = min_total
        self.buckets: Dict[str, float] = {}      # state_key -> mean net pnl
        self.counts: Dict[str, int] = {}         # state_key -> n
        self.global_mean: float = 0.0
        self.trained: bool = False
        self.n_train: int = 0

    # ------------------------------------------------------------------ train
    def train(self, episodes: List[Dict]) -> Dict:
        """Fit bucket means from completed episodes. Returns a status dict."""
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        all_pnls: List[float] = []

        for ep in episodes:
            # Respect feature_version when present; mismatched rows are skipped.
            ev = ep.get("feature_version")
            if ev is not None and ev != self.feature_version:
                continue
            skey = ep.get("state_key")
            pnl = _net_pnl(ep)
            if not skey or pnl is None:
                continue
            sums[skey] = sums.get(skey, 0.0) + pnl
            counts[skey] = counts.get(skey, 0) + 1
            all_pnls.append(pnl)

        n = len(all_pnls)
        if n < self.min_total:
            self.trained = False
            self.n_train = n
            return {"status": "insufficient_data", "n": n, "min_total": self.min_total}

        self.buckets = {k: sums[k] / counts[k] for k in sums}
        self.counts = counts
        self.global_mean = sum(all_pnls) / n
        self.trained = True
        self.n_train = n
        return {
            "status": "trained",
            "n": n,
            "states": len(self.buckets),
            "feature_version": self.feature_version,
            "global_mean": self.global_mean,
        }

    # ---------------------------------------------------------------- predict
    def predict_expected_net_return(self, features: Dict) -> float:
        """Expected NET return % for a state, or 0.0 (ABSTAIN) when unsure.

        Abstains on: model not trained, feature_version mismatch, unseen state,
        or a bucket thinner than `min_state_samples`.
        """
        if not self.trained:
            return 0.0
        if not isinstance(features, dict):
            return 0.0
        fv = features.get("feature_version")
        if fv is not None and fv != self.feature_version:
            return 0.0
        skey = features.get("state_key")
        if not skey or skey not in self.buckets:
            return 0.0
        if self.counts.get(skey, 0) < self.min_state_samples:
            return 0.0
        return self.buckets[skey]

    def covers(self, features: Dict) -> bool:
        """True when the model would make a non-abstaining prediction."""
        if not self.trained or not isinstance(features, dict):
            return False
        fv = features.get("feature_version")
        if fv is not None and fv != self.feature_version:
            return False
        skey = features.get("state_key")
        return bool(skey) and self.counts.get(skey, 0) >= self.min_state_samples

    # ------------------------------------------------------------- persistence
    def report(self) -> Dict:
        ranked = sorted(self.buckets.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "trained": self.trained,
            "feature_version": self.feature_version,
            "n_train": self.n_train,
            "states": len(self.buckets),
            "global_mean": self.global_mean,
            "best": ranked[:5],
            "worst": ranked[-5:],
        }

    def save(self, path: str = "model.json") -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "feature_version": self.feature_version,
                    "min_state_samples": self.min_state_samples,
                    "min_total": self.min_total,
                    "buckets": self.buckets,
                    "counts": self.counts,
                    "global_mean": self.global_mean,
                    "trained": self.trained,
                    "n_train": self.n_train,
                },
                f,
                indent=2,
            )

    def load(self, path: str = "model.json") -> bool:
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        self.feature_version = d.get("feature_version", self.feature_version)
        self.min_state_samples = d.get("min_state_samples", self.min_state_samples)
        self.min_total = d.get("min_total", self.min_total)
        self.buckets = d.get("buckets", {})
        self.counts = d.get("counts", {})
        self.global_mean = d.get("global_mean", 0.0)
        self.trained = d.get("trained", False)
        self.n_train = d.get("n_train", 0)
        return True


def default_model_factory() -> ExpectedReturnModel:
    return ExpectedReturnModel()


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Two states: "good" averages +20, "bad" averages -15.
    episodes = []
    for i in range(8):
        episodes.append({"as_of": f"2026-01-{i+1:02d}", "state_key": "good",
                         "net_pnl_pct": 18.0 + (i % 3), "feature_version": FEATURE_VERSION})
    for i in range(8):
        episodes.append({"as_of": f"2026-02-{i+1:02d}", "state_key": "bad",
                         "net_pnl_pct": -14.0 - (i % 3), "feature_version": FEATURE_VERSION})

    m = ExpectedReturnModel(min_total=10, min_state_samples=2)
    status = m.train(episodes)
    if status["status"] != "trained":
        print("FAIL: should have trained", status); ok = False

    pg = m.predict_expected_net_return({"state_key": "good", "feature_version": FEATURE_VERSION})
    pb = m.predict_expected_net_return({"state_key": "bad", "feature_version": FEATURE_VERSION})
    if not (pg > 0 > pb):
        print("FAIL: ranking wrong", pg, pb); ok = False

    # Unseen state -> abstain 0.0.
    pu = m.predict_expected_net_return({"state_key": "unseen", "feature_version": FEATURE_VERSION})
    if pu != 0.0:
        print("FAIL: unseen state should abstain", pu); ok = False

    # Version mismatch -> abstain 0.0.
    pm = m.predict_expected_net_return({"state_key": "good", "feature_version": "9.9.9"})
    if pm != 0.0:
        print("FAIL: version mismatch should abstain", pm); ok = False

    # Insufficient data -> insufficient_data status, predictions abstain.
    m2 = ExpectedReturnModel(min_total=10)
    s2 = m2.train(episodes[:4])
    if s2["status"] != "insufficient_data":
        print("FAIL: thin data should be insufficient_data", s2); ok = False
    if m2.predict_expected_net_return({"state_key": "good"}) != 0.0:
        print("FAIL: untrained model must abstain"); ok = False

    # Thin bucket abstains even when overall trained.
    thin = list(episodes) + [{"as_of": "2026-03-01", "state_key": "thin",
                              "net_pnl_pct": 50.0, "feature_version": FEATURE_VERSION}]
    m3 = ExpectedReturnModel(min_total=10, min_state_samples=2)
    m3.train(thin)
    if m3.predict_expected_net_return({"state_key": "thin", "feature_version": FEATURE_VERSION}) != 0.0:
        print("FAIL: single-sample bucket should abstain"); ok = False

    # Save/load round-trip.
    import os
    import tempfile
    import uuid
    p = os.path.join(tempfile.gettempdir(), f"model_{uuid.uuid4().hex}.json")
    m.save(p)
    m4 = ExpectedReturnModel()
    if not m4.load(p) or abs(m4.predict_expected_net_return(
            {"state_key": "good", "feature_version": FEATURE_VERSION}) - pg) > 1e-9:
        print("FAIL: save/load round-trip"); ok = False
    try:
        os.remove(p)
    except OSError:
        pass

    print("model self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
