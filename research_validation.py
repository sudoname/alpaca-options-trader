"""
Chronological validation over ledger evidence: honest splits, honest metrics.

Everything operates on plain "observation" dicts so the math is testable with
no database and no network:

    {
      "decision_id": str,
      "t_decision": ISO str,          # when the decision was made
      "t_outcome":  ISO str | None,   # when the outcome window CLOSED
      "net_pnl_pct": float | None,    # realized net return %
      "net_pnl_dollars": float | None,
      "gross_pnl_pct": float | None,
      "p_win": float | None,          # predicted P(net>0) at decision time
      "ev": float | None,             # predicted EV at decision time
      "regime": str | None,
      "direction": str | None,
      "filled": bool,                 # did the order actually fill
      "outlay": float | None,         # entry cash at risk ($)
      "shortfall": float | None,      # entry execution shortfall ($/contract)
      "simulated": bool,              # counterfactual (skip policy) outcome
    }

Guarantees the spec demands:
  * splits are strictly chronological; the FINAL TEST slice is returned
    untouched and marked so callers fit nothing on it;
  * purging removes any training observation whose OUTCOME window overlaps a
    later split, and an embargo gap is enforced on top;
  * walk-forward reports per-fold sample counts and coverage so thin folds
    cannot masquerade as evidence;
  * Brier/calibration are computed ONLY over observations with a defined
    binary outcome and a defined prediction;
  * correlated observations are handled by day-block bootstrap CIs, and
    multiple-experiment testing by Bonferroni adjustment;
  * simulated (counterfactual) outcomes are excluded from realized metrics
    unless explicitly requested, and are labeled when included.
"""

import json
import math
import random
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _day(ts: Optional[str]) -> Optional[str]:
    dt = _parse_iso(ts)
    return dt.date().isoformat() if dt else None


# --------------------------------------------------------------------------- #
# Ledger -> observations
# --------------------------------------------------------------------------- #
def observations_from_ledger(ledger, since=None, until=None,
                             include_simulated: bool = False) -> List[Dict]:
    """Join accepted snapshots with their outcome events. Decisions without a
    terminal outcome are EXCLUDED (unresolved is not evidence of anything).
    Only baseline ``kind='candidate'`` snapshots count — shadow variants must
    never leak into baseline validation."""
    out = []
    for s in ledger.snapshots(since=since, until=until, accepted=1,
                              kind="candidate"):
        rec = ledger.get_decision(s["decision_id"])
        events = rec["events"]
        outcome = next((e for e in events if e["event_type"] == "outcome"), None)
        if outcome is None:
            continue
        try:
            payload = json.loads(outcome.get("payload_json") or "{}")
        except (TypeError, ValueError):
            payload = {}
        simulated = (payload.get("class") == "simulated")
        if simulated and not include_simulated:
            continue
        filled = any(e["event_type"] in ("fill", "partial_fill") for e in events)
        qty = None
        try:
            qty = (json.loads(s.get("sizing_json") or "{}") or {}).get("qty")
        except (TypeError, ValueError):
            pass
        fill_ev = next((e for e in events
                        if e["event_type"] in ("fill", "partial_fill")), None)
        outlay = None
        if fill_ev and fill_ev.get("price") is not None:
            mult = float(s.get("multiplier") or 100.0)
            outlay = float(fill_ev["price"]) * mult * float(fill_ev.get("qty") or qty or 1)
        out.append({
            "decision_id": s["decision_id"],
            "t_decision": s.get("event_at_utc") or s.get("received_at_utc"),
            "t_outcome": outcome.get("event_at_utc") or outcome.get("received_at_utc"),
            "net_pnl_pct": payload.get("net_pnl_pct",
                                       payload.get("counterfactual_pnl_pct")),
            "net_pnl_dollars": payload.get("net_pnl_dollars"),
            "gross_pnl_pct": payload.get("gross_pnl_pct"),
            "p_win": s.get("p_win"),
            "ev": s.get("ev"),
            "regime": s.get("regime"),
            "direction": s.get("direction"),
            "filled": filled,
            "outlay": outlay,
            "shortfall": None,   # filled in by callers that ran replay
            "simulated": simulated,
        })
    out.sort(key=lambda o: o.get("t_decision") or "")
    return out


# --------------------------------------------------------------------------- #
# Splits: chronological, purged, embargoed
# --------------------------------------------------------------------------- #
def purge_overlapping(train: List[Dict], boundary_iso: str,
                      embargo_minutes: float = 0.0) -> Tuple[List[Dict], int]:
    """Drop training observations whose OUTCOME window reaches past
    (boundary - embargo). An observation with no t_outcome is treated as
    open-ended and purged (we cannot prove it closed before the boundary)."""
    boundary = _parse_iso(boundary_iso)
    if boundary is None:
        return list(train), 0
    cutoff_s = boundary.timestamp() - embargo_minutes * 60.0
    kept, purged = [], 0
    for o in train:
        end = _parse_iso(o.get("t_outcome"))
        if end is None or end.timestamp() > cutoff_s:
            purged += 1
            continue
        kept.append(o)
    return kept, purged


def chronological_split(obs: Sequence[Dict], train_frac: float = 0.6,
                        val_frac: float = 0.2,
                        embargo_minutes: float = 0.0) -> Dict:
    """Sort by decision time; slice into train/val/final-test; purge train
    against val, and train+val against test. The test slice is returned
    exactly as recorded and flagged untouched: fit NOTHING on it."""
    rows = sorted(obs, key=lambda o: _parse_iso(o.get("t_decision"))
                  or datetime.min.replace(tzinfo=timezone.utc))
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train, val, test = rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]

    purged_train = purged_val = 0
    if val:
        train, purged_train = purge_overlapping(
            train, val[0]["t_decision"], embargo_minutes)
    if test:
        val, purged_val = purge_overlapping(
            val, test[0]["t_decision"], embargo_minutes)
        train, more = purge_overlapping(
            train, test[0]["t_decision"], embargo_minutes)
        purged_train += more
    return {
        "train": train, "val": val, "test": test,
        "test_untouched": True,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "purged": {"train": purged_train, "val": purged_val},
        "embargo_minutes": embargo_minutes,
    }


def walk_forward(obs: Sequence[Dict], n_folds: int = 5,
                 embargo_minutes: float = 0.0,
                 metric_fn: Optional[Callable[[List[Dict]], Dict]] = None) -> Dict:
    """Expanding-window walk-forward. Fold k trains on everything before the
    fold (purged + embargoed) and evaluates on the fold slice. Per-fold sample
    counts and date coverage are always reported; folds too small to mean
    anything are visible as such rather than hidden in an average."""
    metric_fn = metric_fn or performance_metrics
    rows = sorted(obs, key=lambda o: _parse_iso(o.get("t_decision"))
                  or datetime.min.replace(tzinfo=timezone.utc))
    n = len(rows)
    if n_folds < 1 or n == 0:
        return {"folds": [], "n": n}
    fold_size = max(1, n // (n_folds + 1))
    folds = []
    for k in range(1, n_folds + 1):
        start, end = k * fold_size, min((k + 1) * fold_size, n)
        if start >= n:
            break
        test = rows[start:end]
        train = rows[:start]
        train, purged = purge_overlapping(
            train, test[0]["t_decision"], embargo_minutes)
        folds.append({
            "fold": k,
            "train_n": len(train),
            "purged": purged,
            "test_n": len(test),
            "test_start": test[0]["t_decision"],
            "test_end": test[-1]["t_decision"],
            "coverage_days": len({_day(o["t_decision"]) for o in test}),
            "metrics": metric_fn(test),
        })
    return {"folds": folds, "n": n, "embargo_minutes": embargo_minutes}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def performance_metrics(obs: Sequence[Dict]) -> Dict:
    """Trade metrics over observations with defined net P/L. Each figure
    carries its own sample count; undefined -> None, never 0-pretending."""
    rows = [o for o in obs if o.get("net_pnl_pct") is not None]
    n = len(rows)
    pcts = [float(o["net_pnl_pct"]) for o in rows]
    dollars = [float(o["net_pnl_dollars"]) for o in rows
               if o.get("net_pnl_dollars") is not None]
    wins = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p < 0]

    # Max drawdown on the cumulative net-% equity curve (chronological).
    cum = peak = max_dd = 0.0
    for o in sorted(rows, key=lambda r: r.get("t_decision") or ""):
        cum += float(o["net_pnl_pct"])
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    days = {_day(o.get("t_decision")) for o in obs if _day(o.get("t_decision"))}
    submitted = list(obs)
    filled = [o for o in obs if o.get("filled")]
    shortfalls = [float(o["shortfall"]) for o in obs
                  if o.get("shortfall") is not None]
    outlays = [float(o["outlay"]) for o in obs if o.get("outlay") is not None]
    n_sim = sum(1 for o in obs if o.get("simulated"))

    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    return {
        "n": n,
        "n_observations": len(submitted),
        "n_simulated_included": n_sim,
        "total_net_pnl_pct": sum(pcts) if pcts else None,
        "avg_net_trade_pct": sum(pcts) / n if n else None,
        "total_net_pnl_dollars": sum(dollars) if dollars else None,
        "win_rate": len(wins) / n if n else None,
        "payoff_ratio": (avg_win / abs(avg_loss))
                        if (avg_win is not None and avg_loss) else None,
        "max_drawdown_pct": max_dd if n else None,
        "turnover_per_day": (len(submitted) / len(days)) if days else None,
        "exposure_dollars": sum(outlays) if outlays else None,
        "fill_rate": (len(filled) / len(submitted)) if submitted else None,
        "avg_entry_shortfall": (sum(shortfalls) / len(shortfalls))
                               if shortfalls else None,
        "shortfall_n": len(shortfalls),
    }


def brier_score(obs: Sequence[Dict]) -> Dict:
    """Brier score ONLY over observations with both a defined prediction
    (p_win) and a defined binary outcome (net_pnl_pct)."""
    pairs = [(float(o["p_win"]), 1.0 if float(o["net_pnl_pct"]) > 0 else 0.0)
             for o in obs
             if o.get("p_win") is not None and o.get("net_pnl_pct") is not None]
    if not pairs:
        return {"n": 0, "brier": None}
    return {"n": len(pairs),
            "brier": sum((p - y) ** 2 for p, y in pairs) / len(pairs)}


def calibration_table(obs: Sequence[Dict], bins: int = 5) -> List[Dict]:
    """Predicted-vs-observed win frequency per probability bin (defined
    predictions + defined binary outcomes only)."""
    pairs = [(float(o["p_win"]), 1.0 if float(o["net_pnl_pct"]) > 0 else 0.0)
             for o in obs
             if o.get("p_win") is not None and o.get("net_pnl_pct") is not None]
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(p, y) for p, y in pairs
               if (lo <= p < hi) or (b == bins - 1 and p == hi)]
        table.append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": len(inb),
            "avg_predicted": sum(p for p, _ in inb) / len(inb) if inb else None,
            "observed_win_rate": sum(y for _, y in inb) / len(inb) if inb else None,
        })
    return table


def ev_vs_realized(obs: Sequence[Dict], group_by: str = "regime") -> Dict:
    """Average predicted EV vs average realized net return, grouped (by regime
    or direction), using only observations that carry both numbers."""
    groups: Dict[str, Dict] = {}
    for o in obs:
        if o.get("ev") is None or o.get("net_pnl_pct") is None:
            continue
        key = str(o.get(group_by) or "unknown")
        g = groups.setdefault(key, {"n": 0, "sum_ev": 0.0, "sum_real": 0.0})
        g["n"] += 1
        g["sum_ev"] += float(o["ev"])
        g["sum_real"] += float(o["net_pnl_pct"])
    for g in groups.values():
        g["avg_predicted_ev"] = g.pop("sum_ev") / g["n"]
        g["avg_realized_net_pct"] = g.pop("sum_real") / g["n"]
        g["gap"] = g["avg_predicted_ev"] - g["avg_realized_net_pct"]
    return groups


# --------------------------------------------------------------------------- #
# Uncertainty: day-block bootstrap + multiple comparisons
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(obs: Sequence[Dict],
                       stat_fn: Callable[[List[Dict]], Optional[float]],
                       n_boot: int = 500, seed: int = 7,
                       alpha: float = 0.05) -> Dict:
    """Bootstrap CI resampling whole DAYS (not trades), so same-day correlated
    observations move together. Deterministic for a given seed."""
    by_day: Dict[str, List[Dict]] = {}
    for o in obs:
        d = _day(o.get("t_decision"))
        if d:
            by_day.setdefault(d, []).append(o)
    days = sorted(by_day)
    if not days:
        return {"n_days": 0, "point": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample: List[Dict] = []
        for _ in range(len(days)):
            sample.extend(by_day[rng.choice(days)])
        v = stat_fn(sample)
        if v is not None:
            stats.append(float(v))
    stats.sort()
    if not stats:
        return {"n_days": len(days), "point": stat_fn(list(obs)),
                "lo": None, "hi": None}
    lo_i = int((alpha / 2) * len(stats))
    hi_i = min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))
    return {"n_days": len(days), "n_boot": len(stats),
            "point": stat_fn(list(obs)),
            "lo": stats[lo_i], "hi": stats[hi_i], "alpha": alpha}


def bonferroni(alpha: float, n_experiments: int) -> float:
    """Per-experiment significance threshold after multiple comparisons."""
    return alpha / max(1, int(n_experiments))


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #
def validation_report(obs: Sequence[Dict], *, embargo_minutes: float = 390.0,
                      n_folds: int = 5, n_experiments: int = 1,
                      seed: int = 7) -> Dict:
    """One honest report: splits, walk-forward, metrics, calibration,
    EV-vs-realized, and a day-block CI on the average net trade."""
    obs = list(obs)
    split = chronological_split(obs, embargo_minutes=embargo_minutes)
    wf = walk_forward(obs, n_folds=n_folds, embargo_minutes=embargo_minutes)

    def _avg_net(rows):
        m = performance_metrics(rows)
        return m["avg_net_trade_pct"]

    return {
        "n_observations": len(obs),
        "split_counts": split["counts"],
        "purged": split["purged"],
        "embargo_minutes": embargo_minutes,
        "test_untouched": split["test_untouched"],
        "train_metrics": performance_metrics(split["train"]),
        "val_metrics": performance_metrics(split["val"]),
        "walk_forward": wf,
        "brier": brier_score(obs),
        "calibration": calibration_table(obs),
        "ev_vs_realized_by_regime": ev_vs_realized(obs, "regime"),
        "ev_vs_realized_by_direction": ev_vs_realized(obs, "direction"),
        "avg_net_trade_ci": block_bootstrap_ci(obs, _avg_net, seed=seed),
        "bonferroni_alpha_0p05": bonferroni(0.05, n_experiments),
        "note": ("final test slice intentionally not evaluated here; "
                 "evaluate it once, at the very end, by explicit choice"),
    }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _mk(i: int, day: int, net: Optional[float], p: Optional[float] = None,
        ev: Optional[float] = None, regime: str = "trend",
        hold_min: int = 60, filled: bool = True) -> Dict:
    t0 = datetime(2026, 11, 1 + day, 10, 0, tzinfo=timezone.utc)
    t1 = t0.timestamp() + hold_min * 60
    return {
        "decision_id": f"d{i}", "t_decision": t0.isoformat(),
        "t_outcome": datetime.fromtimestamp(t1, tz=timezone.utc).isoformat(),
        "net_pnl_pct": net, "net_pnl_dollars": net, "gross_pnl_pct": net,
        "p_win": p, "ev": ev, "regime": regime, "direction": "CALL",
        "filled": filled, "outlay": 100.0, "shortfall": 0.02,
        "simulated": False,
    }


def _self_test() -> int:
    ok = True

    # ---------------- splits: chronological + untouched test --------------- #
    obs = [_mk(i, day=i, net=(1.0 if i % 2 == 0 else -1.0),
               p=0.6 if i % 2 == 0 else 0.4, ev=0.5) for i in range(10)]
    split = chronological_split(obs, train_frac=0.6, val_frac=0.2)
    if split["counts"]["test"] != 2 or not split["test_untouched"]:
        print("FAIL: split counts", split["counts"]); ok = False
    last_train = max(_parse_iso(o["t_decision"]) for o in split["train"])
    first_val = min(_parse_iso(o["t_decision"]) for o in split["val"])
    if last_train >= first_val:
        print("FAIL: chronological leakage train->val"); ok = False

    # ---------------- purge: overlapping outcome windows are removed ------- #
    # Trade on day 5 whose outcome closes AFTER the val boundary (day 6 10:00).
    long_hold = _mk(99, day=5, net=2.0, hold_min=60 * 48)  # closes day 7
    obs2 = obs + [long_hold]
    split2 = chronological_split(obs2, train_frac=0.7, val_frac=0.2)
    train_ids = {o["decision_id"] for o in split2["train"]}
    if "d99" in train_ids:
        print("FAIL: outcome overlapping val must be purged from train"); ok = False
    if split2["purged"]["train"] < 1:
        print("FAIL: purge must be counted", split2["purged"]); ok = False

    # Embargo widens the exclusion: a trade closing 30 min before the
    # boundary survives embargo=0 but is purged with embargo=60.
    near = _mk(98, day=4, net=1.0, hold_min=60 * 24 - 30)  # closes day5 09:30
    base = [_mk(i, day=i, net=1.0) for i in range(5)] + [near]
    kept0, _ = purge_overlapping(base, _mk(0, day=5, net=0.0)["t_decision"], 0)
    kept60, _ = purge_overlapping(base, _mk(0, day=5, net=0.0)["t_decision"], 60)
    ids0 = {o["decision_id"] for o in kept0}
    ids60 = {o["decision_id"] for o in kept60}
    if "d98" not in ids0 or "d98" in ids60:
        print("FAIL: embargo semantics", ids0, ids60); ok = False

    # Open-ended (no t_outcome) training rows are purged, not trusted.
    openrow = dict(_mk(97, day=0, net=None), t_outcome=None)
    keptx, purgedx = purge_overlapping([openrow],
                                       _mk(0, day=5, net=0.0)["t_decision"], 0)
    if keptx or purgedx != 1:
        print("FAIL: open outcome must purge"); ok = False

    # ---------------- walk-forward ----------------------------------------- #
    wf = walk_forward(obs, n_folds=4)
    if len(wf["folds"]) != 4:
        print("FAIL: fold count", len(wf["folds"])); ok = False
    for f in wf["folds"]:
        if f["test_n"] <= 0 or f["coverage_days"] <= 0:
            print("FAIL: fold coverage", f); ok = False
        if f["train_n"] > 0 and f["test_start"] is not None:
            pass  # purge already asserted above
    # Later folds train on more data (expanding window).
    if not wf["folds"][0]["train_n"] < wf["folds"][-1]["train_n"]:
        print("FAIL: expanding window"); ok = False

    # ---------------- metrics ---------------------------------------------- #
    m = performance_metrics(obs)
    if m["n"] != 10 or abs(m["win_rate"] - 0.5) > 1e-9:
        print("FAIL: win rate", m); ok = False
    if abs(m["payoff_ratio"] - 1.0) > 1e-9:
        print("FAIL: payoff ratio", m["payoff_ratio"]); ok = False
    # Equity path +1,-1,... -> max drawdown 1.
    if abs(m["max_drawdown_pct"] - 1.0) > 1e-9:
        print("FAIL: drawdown", m["max_drawdown_pct"]); ok = False
    if m["fill_rate"] != 1.0 or m["turnover_per_day"] != 1.0:
        print("FAIL: fill/turnover", m); ok = False
    if m["exposure_dollars"] != 1000.0 or m["shortfall_n"] != 10:
        print("FAIL: exposure/shortfall", m); ok = False
    if performance_metrics([])["win_rate"] is not None:
        print("FAIL: empty metrics must be None"); ok = False

    # ---------------- Brier / calibration ---------------------------------- #
    bs = brier_score(obs)
    # p=0.6 on winners, p=0.4 on losers -> (0.4^2)*10/10 = 0.16
    if bs["n"] != 10 or abs(bs["brier"] - 0.16) > 1e-9:
        print("FAIL: brier", bs); ok = False
    if brier_score([_mk(1, 1, None, p=0.7)])["n"] != 0:
        print("FAIL: undefined outcome must be excluded from Brier"); ok = False
    if brier_score([_mk(1, 1, 1.0, p=None)])["n"] != 0:
        print("FAIL: undefined prediction must be excluded from Brier"); ok = False
    cal = calibration_table(obs, bins=5)
    row = next(r for r in cal if r["bin"] == "[0.60,0.80)")
    if row["n"] != 5 or row["observed_win_rate"] != 1.0:
        print("FAIL: calibration bin", row); ok = False

    # ---------------- EV vs realized --------------------------------------- #
    evr = ev_vs_realized(obs, "regime")
    if abs(evr["trend"]["avg_predicted_ev"] - 0.5) > 1e-9 or \
       abs(evr["trend"]["avg_realized_net_pct"] - 0.0) > 1e-9:
        print("FAIL: ev vs realized", evr); ok = False

    # ---------------- bootstrap: day blocks, deterministic ----------------- #
    def _avg(rows):
        m2 = performance_metrics(rows)
        return m2["avg_net_trade_pct"]

    ci1 = block_bootstrap_ci(obs, _avg, n_boot=200, seed=42)
    ci2 = block_bootstrap_ci(obs, _avg, n_boot=200, seed=42)
    if ci1 != ci2:
        print("FAIL: bootstrap must be deterministic per seed"); ok = False
    if ci1["n_days"] != 10 or not (ci1["lo"] <= ci1["point"] <= ci1["hi"]):
        print("FAIL: CI sanity", ci1); ok = False

    if abs(bonferroni(0.05, 10) - 0.005) > 1e-12:
        print("FAIL: bonferroni"); ok = False

    # ---------------- full report + ledger adapter ------------------------- #
    rep = validation_report(obs, embargo_minutes=0, n_folds=3, n_experiments=4)
    if rep["bonferroni_alpha_0p05"] != 0.0125 or not rep["test_untouched"]:
        print("FAIL: report fields"); ok = False
    if "test_metrics" in rep:
        print("FAIL: report must NOT evaluate the final test slice"); ok = False

    from research_ledger import ResearchLedger
    led = ResearchLedger(":memory:")
    d1 = led.record_snapshot(decision_id="v1", accepted=True, direction="CALL",
                             symbol="SPY261219C00500000", p_win=0.6, ev=1.0,
                             quote={"bid": 1.0, "ask": 1.1}, sizing={"qty": 2})
    led.record_event(decision_id=d1, event_type="fill", broker_order_id="o1",
                     qty=2, price=1.12)
    led.record_event(decision_id=d1, event_type="outcome",
                     payload={"class": "realized", "net_pnl_pct": 10.0,
                              "gross_pnl_pct": 14.0})
    d2 = led.record_snapshot(decision_id="v2", accepted=True, direction="PUT",
                             symbol="QQQ261219P00400000")
    led.record_event(decision_id=d2, event_type="outcome",
                     payload={"class": "simulated",
                              "policy": "underlying_counterfactual_v1",
                              "counterfactual_pnl_pct": -3.0})
    led.record_snapshot(decision_id="v3", accepted=True, direction="CALL")

    real_only = observations_from_ledger(led)
    if len(real_only) != 1 or real_only[0]["decision_id"] != "v1":
        print("FAIL: simulated/unresolved excluded by default", real_only); ok = False
    if abs(real_only[0]["outlay"] - 1.12 * 100 * 2) > 1e-9:
        print("FAIL: outlay from fill", real_only[0]["outlay"]); ok = False
    with_sim = observations_from_ledger(led, include_simulated=True)
    if len(with_sim) != 2 or not any(o["simulated"] and
                                     o["net_pnl_pct"] == -3.0 for o in with_sim):
        print("FAIL: simulated included only on request + labeled", with_sim); ok = False

    led.close()
    print("research_validation self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
