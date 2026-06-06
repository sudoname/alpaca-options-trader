"""
Walk-forward evaluation harness (honest, leakage-resistant).

Given a list of completed `episodes` (time-stamped, net-of-cost), this trains a
model on the past and measures it ONLY on the future:

  * episodes are sorted by `as_of`; NOTHING is shuffled;
  * the most recent `holdout_frac` is set aside as an UNTOUCHED holdout the
    folds never see;
  * the remaining pool is evaluated with expanding-window folds (train on
    [0..k], test on the next chunk), with an `embargo_days` gap between the end
    of training and the start of testing to prevent adjacency leakage.

The evaluation policy is intentionally trivial and for measurement only: "act
when the model predicts a positive expected NET return". Metrics are computed on
the acted trades: expectancy, win_rate, max_drawdown, effective_sample_size, and
coverage (how often the model made a non-abstaining prediction). This harness
NEVER feeds the live decision policy.
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from model import ExpectedReturnModel, default_model_factory


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_of_date(ep: Dict) -> Optional[datetime]:
    raw = ep.get("as_of") or ep.get("closed_at") or ""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        try:
            return datetime.strptime(str(raw)[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _features_of(ep: Dict) -> Dict:
    return {
        "state_key": ep.get("state_key"),
        "feature_version": ep.get("feature_version"),
    }


def _max_drawdown(pnls: List[float]) -> float:
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return mdd


def _evaluate(model: ExpectedReturnModel, test: List[Dict]) -> Dict:
    """Apply the act-if-positive policy and summarize realized NET outcomes."""
    taken: List[float] = []
    covered = 0
    for ep in test:
        feats = _features_of(ep)
        if model.covers(feats):
            covered += 1
        pred = model.predict_expected_net_return(feats)
        if pred > 0:
            pnl = ep.get("net_pnl_pct")
            if pnl is not None:
                taken.append(float(pnl))
    n = len(taken)
    wins = sum(1 for p in taken if p > 0)
    total = sum(taken)
    return {
        "test_n": len(test),
        "taken": n,
        "coverage": (covered / len(test)) if test else 0.0,
        "expectancy_pct": (total / n) if n else 0.0,
        "win_rate": (wins / n) if n else 0.0,
        "total_pnl_pct": total,
        "max_drawdown": _max_drawdown(taken),
        "effective_sample_size": float(n),
    }


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward_eval(
    episodes: List[Dict],
    *,
    model_factory: Callable[[], ExpectedReturnModel] = default_model_factory,
    n_folds: int = 4,
    holdout_frac: float = 0.2,
    embargo_days: int = 1,
) -> Dict:
    """Expanding-window folds + an untouched final holdout. Returns a report."""
    usable = [e for e in episodes if e.get("net_pnl_pct") is not None and e.get("state_key")]
    usable.sort(key=lambda e: (e.get("as_of") or e.get("closed_at") or ""))
    n = len(usable)
    if n < 4:
        return {"status": "insufficient_data", "n": n}

    holdout_n = max(1, int(n * holdout_frac))
    pool = usable[: n - holdout_n]
    holdout = usable[n - holdout_n:]

    folds: List[Dict] = []
    oos_taken: List[float] = []

    if len(pool) >= n_folds + 1:
        chunk = len(pool) // (n_folds + 1)
        for k in range(1, n_folds + 1):
            train = pool[: chunk * k]
            test = pool[chunk * k: chunk * (k + 1)]
            if not train or not test:
                continue
            # Embargo: drop test rows too close in time to the train cutoff.
            cutoff = _as_of_date(train[-1])
            if cutoff is not None and embargo_days > 0:
                limit = cutoff + timedelta(days=embargo_days)
                test = [e for e in test if (_as_of_date(e) is None or _as_of_date(e) > limit)]
            if not test:
                continue
            model = model_factory()
            status = model.train(train)
            res = _evaluate(model, test)
            res["fold"] = k
            res["train_n"] = len(train)
            res["train_status"] = status.get("status")
            folds.append(res)
            # collect realized OOS taken pnls for an aggregate
            for ep in test:
                feats = _features_of(ep)
                if model.predict_expected_net_return(feats) > 0 and ep.get("net_pnl_pct") is not None:
                    oos_taken.append(float(ep["net_pnl_pct"]))

    # Final model trained on the whole pool, evaluated on the untouched holdout.
    final_model = model_factory()
    final_status = final_model.train(pool)
    holdout_res = _evaluate(final_model, holdout)
    holdout_res["train_n"] = len(pool)
    holdout_res["train_status"] = final_status.get("status")

    oos_wins = sum(1 for p in oos_taken if p > 0)
    oos = {
        "taken": len(oos_taken),
        "expectancy_pct": (sum(oos_taken) / len(oos_taken)) if oos_taken else 0.0,
        "win_rate": (oos_wins / len(oos_taken)) if oos_taken else 0.0,
        "max_drawdown": _max_drawdown(oos_taken),
        "effective_sample_size": float(len(oos_taken)),
    }

    return {
        "status": "ok",
        "n": n,
        "pool_n": len(pool),
        "holdout_n": len(holdout),
        "folds": folds,
        "oos_aggregate": oos,
        "holdout": holdout_res,
    }


def shift_labels_forward(episodes: List[Dict]) -> List[Dict]:
    """Time-order episodes and reassign each one the NEXT episode's net P/L.

    Used by the leakage test: a model that truly learns state->return should do
    materially worse when the labels are misaligned by one step.
    """
    ordered = sorted(episodes, key=lambda e: (e.get("as_of") or e.get("closed_at") or ""))
    out = []
    for i in range(len(ordered) - 1):
        ep = dict(ordered[i])
        ep["net_pnl_pct"] = ordered[i + 1].get("net_pnl_pct")
        out.append(ep)
    return out


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    FV = "1.0.0"

    # Build a time series where state_key deterministically sets the NET return:
    # "good" -> +20, "bad" -> -15. States are assigned by a SEEDED RNG (not a
    # fixed alternation) so the order is non-periodic. This matters for the
    # future-shift test below: on periodic data, shifting labels merely swaps
    # which bucket wins and the model stays perfectly predictive; on a
    # non-periodic sequence, shifting genuinely scrambles state->return.
    import random as _random
    rng = _random.Random(1234)
    episodes = []
    day = datetime(2026, 1, 1)
    for i in range(60):
        skey = "good" if rng.random() < 0.5 else "bad"
        pnl = 20.0 if skey == "good" else -15.0
        episodes.append({
            "as_of": (day + timedelta(days=i)).isoformat(),
            "state_key": skey,
            "net_pnl_pct": pnl,
            "feature_version": FV,
        })

    factory = lambda: ExpectedReturnModel(min_total=5, min_state_samples=2)

    rep = walk_forward_eval(episodes, model_factory=factory, n_folds=4,
                            holdout_frac=0.2, embargo_days=1)
    if rep.get("status") != "ok":
        print("FAIL: walk_forward_eval did not run", rep); return 1

    # On aligned data the model should only take the "good" state -> positive
    # holdout expectancy and a perfect win rate.
    ho = rep["holdout"]
    if ho["expectancy_pct"] <= 0:
        print("FAIL: aligned holdout expectancy should be positive", ho); ok = False
    if ho["taken"] == 0:
        print("FAIL: model should have taken some holdout trades", ho); ok = False
    aligned_oos = rep["oos_aggregate"]["expectancy_pct"]

    # Untouched holdout must not overlap the pool.
    if rep["pool_n"] + rep["holdout_n"] != rep["n"]:
        print("FAIL: holdout/pool partition is inconsistent", rep); ok = False

    # Future-shift: misaligning labels by one step should degrade OOS expectancy.
    shifted = shift_labels_forward(episodes)
    rep_s = walk_forward_eval(shifted, model_factory=factory, n_folds=4,
                              holdout_frac=0.2, embargo_days=1)
    shifted_oos = rep_s["oos_aggregate"]["expectancy_pct"]
    if not (shifted_oos < aligned_oos):
        print("FAIL: shifted-label OOS should be worse than aligned",
              shifted_oos, aligned_oos); ok = False

    # Insufficient data path.
    if walk_forward_eval(episodes[:2], model_factory=factory).get("status") != "insufficient_data":
        print("FAIL: tiny input should be insufficient_data"); ok = False

    print("walk_forward self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
