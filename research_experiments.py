"""
Experiment registry, component comparison / ablation, and the shadow harness.

Three concerns, one research-side module (never imported by the live path):

1. ExperimentRegistry — a durable record of every comparison we ATTEMPT,
   including failures. Each experiment names its hypothesis, the component
   under test, the baseline and candidate configurations, the dataset and
   evaluation window, and explicit rejection criteria frozen at registration.
   One controlled change at a time is enforced: a candidate differing from
   its baseline in more than one key is refused unless the caller explicitly
   opts out (and the changed keys are stored either way).

2. Component comparison — measurable decomposition of performance into the
   bot's separable components, computed only from recorded evidence:
     direction        win rate / avg net P/L by CALL vs PUT (realized outcomes)
     contract_select  entry spread % and cost drag of the chosen contracts
     entry_execution  modeled-vs-actual entry shortfall (replay vs fills)
     exit_rules       outcome mix and avg net P/L per exit reason
   Plus single-feature ablation generation (disable exactly one feature).

3. ShadowComparator — lets a candidate scorer log decisions BESIDE the
   baseline without placing orders or touching baseline state: bounded queue,
   non-blocking enqueue, dropped-event accounting, writes immutable
   `kind='shadow_candidate'` snapshots to the research ledger. Nothing here
   can submit an order, and nothing here promotes a candidate: promotion is
   only ever a separate, explicit config change made by a human.
"""

import json
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DEFAULT_DB = "research_experiments.db"
STATUSES = ("registered", "running", "completed", "failed", "rejected",
            "inconclusive")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def diff_keys(baseline: Dict, candidate: Dict) -> List[str]:
    """Keys whose values differ between two flat config dicts (either side)."""
    keys = set(baseline) | set(candidate)
    return sorted(k for k in keys if baseline.get(k) != candidate.get(k))


def evaluate_rejection(criteria: Optional[Dict], result: Optional[Dict]
                       ) -> Tuple[bool, List[str]]:
    """Check a result against frozen rejection criteria.

    Supported criterion forms (all optional):
      min_<metric>: result[metric] must be >= value
      max_<metric>: result[metric] must be <= value
    A metric missing from the result FAILS its criterion (absence of evidence
    is not acceptance). Returns (passed, list_of_violations).
    """
    if not criteria:
        return True, []
    result = result or {}
    violations = []
    for key, bound in criteria.items():
        if key.startswith("min_"):
            metric, op = key[4:], "min"
        elif key.startswith("max_"):
            metric, op = key[4:], "max"
        else:
            continue
        val = result.get(metric)
        if val is None:
            violations.append(f"{key}: metric '{metric}' missing from result")
            continue
        try:
            v, b = float(val), float(bound)
        except (TypeError, ValueError):
            violations.append(f"{key}: non-numeric ({val!r} vs {bound!r})")
            continue
        if op == "min" and v < b:
            violations.append(f"{key}: {v} < {b}")
        if op == "max" and v > b:
            violations.append(f"{key}: {v} > {b}")
    return (not violations), violations


# --------------------------------------------------------------------------- #
# Experiment registry
# --------------------------------------------------------------------------- #
class ExperimentRegistry:
    """Durable registry of attempted experiments (including failures)."""

    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                hypothesis TEXT,
                component TEXT,
                baseline_json TEXT,
                candidate_json TEXT,
                changed_keys_json TEXT,
                dataset_id TEXT,
                eval_window_start TEXT,
                eval_window_end TEXT,
                rejection_criteria_json TEXT,
                status TEXT NOT NULL,
                result_json TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiment_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                at TEXT NOT NULL,
                event TEXT NOT NULL,
                detail_json TEXT
            );
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------- lifecycle
    def register(self, *, name: str, hypothesis: str, component: str,
                 baseline: Dict, candidate: Dict, dataset_id: str,
                 eval_window: Tuple[Optional[str], Optional[str]] = (None, None),
                 rejection_criteria: Optional[Dict] = None,
                 allow_multi_change: bool = False,
                 notes: str = "") -> str:
        """Register one experiment. Enforces one-controlled-change-at-a-time:
        more than one differing config key is refused unless the caller sets
        allow_multi_change=True (the changed keys are recorded regardless)."""
        changed = diff_keys(baseline or {}, candidate or {})
        if len(changed) > 1 and not allow_multi_change:
            raise ValueError(
                f"candidate changes {len(changed)} keys {changed}; register "
                "one controlled change at a time (or allow_multi_change=True)")
        eid = str(uuid.uuid4())
        now = _now()
        self.conn.execute(
            """INSERT INTO experiments (experiment_id, name, hypothesis,
               component, baseline_json, candidate_json, changed_keys_json,
               dataset_id, eval_window_start, eval_window_end,
               rejection_criteria_json, status, result_json, notes,
               created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (eid, name, hypothesis, component,
             json.dumps(baseline, sort_keys=True, default=str),
             json.dumps(candidate, sort_keys=True, default=str),
             json.dumps(changed),
             dataset_id, eval_window[0], eval_window[1],
             json.dumps(rejection_criteria or {}, sort_keys=True),
             "registered", None, notes, now, now))
        self.conn.commit()
        self.log(eid, "registered", {"changed_keys": changed})
        return eid

    def log(self, experiment_id: str, event: str, detail: Optional[Dict] = None):
        """Append to the attempt log. EVERY attempt — including failures —
        belongs here; a crashed evaluation is still evidence."""
        self.conn.execute(
            "INSERT INTO experiment_log (experiment_id, at, event, detail_json)"
            " VALUES (?,?,?,?)",
            (experiment_id, _now(), event,
             json.dumps(detail, default=str) if detail is not None else None))
        self.conn.commit()

    def record_result(self, experiment_id: str, result: Optional[Dict],
                      status: Optional[str] = None, notes: str = "") -> str:
        """Store an evaluation result. When no status is given, the frozen
        rejection criteria decide: violated -> 'rejected', satisfied ->
        'completed'. Never promotes anything anywhere."""
        row = self.get(experiment_id)
        if row is None:
            raise KeyError(f"unknown experiment {experiment_id}")
        if status is None:
            criteria = json.loads(row["rejection_criteria_json"] or "{}")
            passed, violations = evaluate_rejection(criteria, result)
            status = "completed" if passed else "rejected"
            self.log(experiment_id, "criteria_evaluated",
                     {"passed": passed, "violations": violations})
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        self.conn.execute(
            "UPDATE experiments SET result_json=?, status=?, notes=?, "
            "updated_at=? WHERE experiment_id=?",
            (json.dumps(result, default=str) if result is not None else None,
             status, notes or row["notes"], _now(), experiment_id))
        self.conn.commit()
        self.log(experiment_id, f"result_{status}", result)
        return status

    def record_failure(self, experiment_id: str, error: str):
        """A failed attempt is recorded, not discarded."""
        self.conn.execute(
            "UPDATE experiments SET status='failed', updated_at=? "
            "WHERE experiment_id=?", (_now(), experiment_id))
        self.conn.commit()
        self.log(experiment_id, "failed", {"error": error})

    # ---------------------------------------------------------------- reads
    def get(self, experiment_id: str) -> Optional[Dict]:
        cur = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list(self, status: Optional[str] = None) -> List[Dict]:
        if status:
            cur = self.conn.execute(
                "SELECT * FROM experiments WHERE status=? ORDER BY created_at",
                (status,))
        else:
            cur = self.conn.execute(
                "SELECT * FROM experiments ORDER BY created_at")
        return [dict(r) for r in cur.fetchall()]

    def history(self, experiment_id: str) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT * FROM experiment_log WHERE experiment_id=? ORDER BY id",
            (experiment_id,))
        return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #
def single_feature_ablations(base_config: Dict, features: List[str],
                             off_value=False) -> List[Dict]:
    """Candidate configs that each disable EXACTLY one feature. Unknown
    feature names are refused (an ablation of nothing proves nothing)."""
    missing = [f for f in features if f not in base_config]
    if missing:
        raise KeyError(f"features not in base config: {missing}")
    out = []
    for f in features:
        cand = dict(base_config)
        cand[f] = off_value
        out.append({"ablated_feature": f, "config": cand})
    return out


def register_ablation_suite(registry: ExperimentRegistry, *, base_config: Dict,
                            features: List[str], dataset_id: str,
                            rejection_criteria: Optional[Dict] = None,
                            name_prefix: str = "ablation") -> List[str]:
    """One experiment per single-feature ablation."""
    ids = []
    for ab in single_feature_ablations(base_config, features):
        eid = registry.register(
            name=f"{name_prefix}:{ab['ablated_feature']}",
            hypothesis=(f"disabling '{ab['ablated_feature']}' does not "
                        "degrade net performance"),
            component="ablation",
            baseline=base_config, candidate=ab["config"],
            dataset_id=dataset_id, rejection_criteria=rejection_criteria)
        ids.append(eid)
    return ids


# --------------------------------------------------------------------------- #
# Component comparison (from recorded evidence only)
# --------------------------------------------------------------------------- #
def _terminal_outcomes(ledger, since=None, until=None) -> List[Dict]:
    """(snapshot, outcome_payload) pairs for decisions with an outcome event.
    Baseline candidates only; shadow snapshots are excluded."""
    out = []
    for s in ledger.snapshots(since=since, until=until, accepted=1,
                              kind="candidate"):
        rec = ledger.get_decision(s["decision_id"])
        for e in rec["events"]:
            if e["event_type"] == "outcome":
                try:
                    payload = json.loads(e.get("payload_json") or "{}")
                except (TypeError, ValueError):
                    payload = {}
                out.append({"snapshot": s, "payload": payload, "event": e})
                break
    return out


def component_report(ledger, since=None, until=None,
                     replay_model=None) -> Dict:
    """Decompose recorded performance by component. Only uses evidence that
    exists; every section reports its own sample count."""
    from research_replay import ExecutionModel, replay_cohort

    outcomes = _terminal_outcomes(ledger, since, until)
    realized = [o for o in outcomes
                if (o["payload"].get("class") or "realized") == "realized"]

    # --- direction: win rate / avg net by CALL vs PUT ---------------------- #
    direction: Dict[str, Dict] = {}
    for o in realized:
        d = (o["snapshot"].get("direction") or "?").upper()
        net = o["payload"].get("net_pnl_pct")
        if net is None:
            continue
        g = direction.setdefault(d, {"n": 0, "wins": 0, "sum_net": 0.0})
        g["n"] += 1
        g["wins"] += 1 if float(net) > 0 else 0
        g["sum_net"] += float(net)
    for g in direction.values():
        g["win_rate"] = g["wins"] / g["n"] if g["n"] else None
        g["avg_net_pnl_pct"] = g["sum_net"] / g["n"] if g["n"] else None

    # --- contract selection: entry spread % and modeled cost drag ---------- #
    spreads, drags = [], []
    for s in ledger.snapshots(since=since, until=until, accepted=1):
        bid, ask = s.get("quote_bid"), s.get("quote_ask")
        if bid is None or ask is None or (bid + ask) <= 0:
            continue
        mid = (float(bid) + float(ask)) / 2.0
        if mid > 0:
            spreads.append((float(ask) - float(bid)) / mid * 100.0)
    for o in realized:
        g, n = o["payload"].get("gross_pnl_pct"), o["payload"].get("net_pnl_pct")
        if g is not None and n is not None:
            drags.append(float(g) - float(n))
    contract = {
        "n_quoted": len(spreads),
        "avg_entry_spread_pct": sum(spreads) / len(spreads) if spreads else None,
        "n_with_cost": len(drags),
        "avg_cost_drag_pct": sum(drags) / len(drags) if drags else None,
    }

    # --- entry execution: modeled vs actual shortfall (via replay) --------- #
    model = replay_model or ExecutionModel()
    cohort = replay_cohort(ledger, model, since=since, until=until, accepted=1)
    execution = {
        "n_shortfall": cohort["summary"]["shortfall_samples"],
        "avg_entry_shortfall_per_contract":
            cohort["summary"]["avg_entry_shortfall"],
        "modeled_fill_rate": cohort["summary"]["fill_rate"],
    }

    # --- exit rules: outcome mix per recorded exit reason ------------------ #
    exits: Dict[str, Dict] = {}
    for o in realized:
        reason = str(o["payload"].get("reason") or "unknown")
        net = o["payload"].get("net_pnl_pct")
        g = exits.setdefault(reason, {"n": 0, "sum_net": 0.0, "with_net": 0})
        g["n"] += 1
        if net is not None:
            g["sum_net"] += float(net)
            g["with_net"] += 1
    for g in exits.values():
        g["avg_net_pnl_pct"] = g["sum_net"] / g["with_net"] if g["with_net"] else None
        g.pop("sum_net")

    return {
        "window": {"since": since, "until": until},
        "n_outcomes": len(outcomes),
        "n_realized": len(realized),
        "direction": direction,
        "contract_selection": contract,
        "entry_execution": execution,
        "exit_rules": exits,
    }


# --------------------------------------------------------------------------- #
# Shadow comparison harness
# --------------------------------------------------------------------------- #
class ShadowComparator:
    """Log candidate decisions beside the baseline — orderless, non-blocking.

    * ``submit()`` NEVER blocks and NEVER raises: a full queue increments
      ``dropped`` and returns False. A research backlog cannot stall risk
      checks or position management.
    * Snapshots are written with kind='shadow_candidate', reusing the same
      immutable snapshot schema as the baseline so comparisons share one
      source of truth.
    * The harness never mutates what it is given (inputs are copied) and has
      no broker, no order method, and no way to change live config: promotion
      is a separate explicit human config change, by design.
    """

    def __init__(self, ledger, *, variant: str, maxsize: int = 256,
                 start_worker: bool = True, worker_timeout_s: float = 1.0):
        self._ledger = ledger
        self.variant = variant
        self.dropped = 0
        self.written = 0
        self.write_errors = 0
        self._q: "queue.Queue[Dict]" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._timeout = worker_timeout_s
        self._worker = None
        if start_worker:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    # ------------------------------------------------------------ enqueue
    def submit(self, candidate: Dict) -> bool:
        """Non-blocking, exception-free enqueue of one candidate decision."""
        try:
            self._q.put_nowait(dict(candidate or {}))  # copy: never share state
            return True
        except queue.Full:
            self.dropped += 1
            return False
        except Exception:
            self.dropped += 1
            return False

    # ------------------------------------------------------------- drain
    def _write_one(self, item: Dict):
        extra = dict(item.get("extra") or {})
        extra["shadow_variant"] = self.variant
        try:
            self._ledger.record_snapshot(
                decision_id=item.get("decision_id"),
                kind="shadow_candidate",
                accepted=bool(item.get("accepted", True)),
                reject_code=item.get("reject_code"),
                symbol=item.get("symbol"),
                underlying=item.get("underlying"),
                quote=item.get("quote"),
                underlying_price=item.get("underlying_price"),
                greeks=item.get("greeks"),
                direction=item.get("direction"),
                score=item.get("score"), p_win=item.get("p_win"),
                ev=item.get("ev"), signals=item.get("signals"),
                regime=item.get("regime"), sizing=item.get("sizing"),
                prediction=item.get("prediction"),
                strategy_version=f"shadow:{self.variant}",
                extra=extra)
            self.written += 1
        except Exception:
            self.write_errors += 1

    def drain(self) -> int:
        """Synchronously flush the queue (used by tests / shutdown)."""
        n = 0
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return n
            self._write_one(item)
            n += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=self._timeout)
            except queue.Empty:
                continue
            self._write_one(item)

    def close(self, timeout_s: float = 2.0):
        """Bounded shutdown: flush what we can, then stop; stragglers are
        counted as dropped rather than waited on forever."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout_s)
        self.drain()

    def accounting(self) -> Dict:
        return {"variant": self.variant, "written": self.written,
                "dropped": self.dropped, "write_errors": self.write_errors,
                "queued": self._q.qsize()}


def compare_shadow_to_baseline(ledger, variant: str,
                               since=None, until=None) -> Dict:
    """Side-by-side counts/scores of a shadow variant vs baseline snapshots.
    Comparison only — promotion requires a human config change elsewhere."""
    base = ledger.snapshots(since=since, until=until, kind="candidate")
    shadow = [s for s in ledger.snapshots(since=since, until=until,
                                          kind="shadow_candidate")
              if f'"shadow_variant": "{variant}"' in (s.get("extra_json") or "")]

    def _agg(snaps):
        scores = [s["score"] for s in snaps if s.get("score") is not None]
        return {
            "n": len(snaps),
            "n_accepted": sum(1 for s in snaps if s.get("accepted")),
            "avg_score": sum(scores) / len(scores) if scores else None,
        }

    base_a, shad_a = _agg(base), _agg(shadow)
    overlap = ({s["decision_id"] for s in base}
               & {s["decision_id"] for s in shadow})
    return {"variant": variant, "baseline": base_a, "shadow": shad_a,
            "shared_decision_ids": len(overlap),
            "note": "comparison only; promotion is a separate explicit "
                    "config change"}


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from research_ledger import ResearchLedger

    ok = True

    # ---------------- registry: one controlled change ---------------------- #
    reg = ExperimentRegistry(":memory:")
    base = {"EV_GATE": 0.5, "MOMENTUM_WINDOW": 14, "USE_IV_RANK": True}

    eid = reg.register(
        name="ev-gate-0.6", hypothesis="raising the EV gate improves net P/L",
        component="direction", baseline=base,
        candidate=dict(base, EV_GATE=0.6), dataset_id="ledger:2026-11",
        eval_window=("2026-11-01", "2026-11-30"),
        rejection_criteria={"min_decisions": 50, "min_avg_net_pnl_pct": 0.0})
    if reg.get(eid)["status"] != "registered":
        print("FAIL: registration status"); ok = False
    if json.loads(reg.get(eid)["changed_keys_json"]) != ["EV_GATE"]:
        print("FAIL: changed keys"); ok = False

    try:
        reg.register(name="two-changes", hypothesis="h", component="x",
                     baseline=base,
                     candidate=dict(base, EV_GATE=0.6, MOMENTUM_WINDOW=20),
                     dataset_id="d")
        print("FAIL: multi-change must be refused"); ok = False
    except ValueError:
        pass
    eid2 = reg.register(name="two-ok", hypothesis="h", component="x",
                        baseline=base,
                        candidate=dict(base, EV_GATE=0.6, MOMENTUM_WINDOW=20),
                        dataset_id="d", allow_multi_change=True,
                        rejection_criteria={"min_decisions": 50,
                                            "min_avg_net_pnl_pct": 0.0})
    if len(json.loads(reg.get(eid2)["changed_keys_json"])) != 2:
        print("FAIL: multi-change keys recorded"); ok = False

    # Rejection criteria drive the status; missing metrics fail.
    st = reg.record_result(eid, {"decisions": 80, "avg_net_pnl_pct": 1.2})
    if st != "completed":
        print("FAIL: satisfying result should complete", st); ok = False
    st2 = reg.record_result(eid2, {"decisions": 10})  # too few + missing metric
    if st2 != "rejected":
        print("FAIL: violating result should reject", st2); ok = False
    passed, viol = evaluate_rejection({"min_x": 1}, {})
    if passed or "missing" not in viol[0]:
        print("FAIL: missing metric must violate", viol); ok = False

    # Failures are logged, not lost.
    eid3 = reg.register(name="crashy", hypothesis="h", component="x",
                        baseline=base, candidate=dict(base, USE_IV_RANK=False),
                        dataset_id="d")
    reg.record_failure(eid3, "ZeroDivisionError: empty cohort")
    if reg.get(eid3)["status"] != "failed":
        print("FAIL: failure status"); ok = False
    if not any(h["event"] == "failed" for h in reg.history(eid3)):
        print("FAIL: failure must be in history"); ok = False
    if len(reg.list()) != 3 or len(reg.list(status="failed")) != 1:
        print("FAIL: listing"); ok = False

    # ---------------- ablations ------------------------------------------- #
    abls = single_feature_ablations(base, ["USE_IV_RANK", "EV_GATE"])
    if len(abls) != 2 or abls[0]["config"]["USE_IV_RANK"] is not False:
        print("FAIL: ablation generation"); ok = False
    for ab in abls:
        if len(diff_keys(base, ab["config"])) != 1:
            print("FAIL: ablation must change exactly one key"); ok = False
    try:
        single_feature_ablations(base, ["NOT_A_FEATURE"])
        print("FAIL: unknown feature must be refused"); ok = False
    except KeyError:
        pass
    suite = register_ablation_suite(reg, base_config=base,
                                    features=["USE_IV_RANK"],
                                    dataset_id="d")
    if len(suite) != 1 or reg.get(suite[0])["component"] != "ablation":
        print("FAIL: ablation suite registration"); ok = False

    # ---------------- component report ------------------------------------ #
    led = ResearchLedger(":memory:")
    d1 = led.record_snapshot(decision_id="c1", accepted=True, direction="CALL",
                             symbol="SPY261219C00500000",
                             quote={"bid": 1.00, "ask": 1.10}, sizing={"qty": 1})
    led.record_event(decision_id=d1, event_type="fill", broker_order_id="o1",
                     qty=1, price=1.15)
    led.record_event(decision_id=d1, event_type="outcome",
                     payload={"class": "realized", "reason": "take_profit",
                              "gross_pnl_pct": 20.0, "net_pnl_pct": 15.0})
    d2 = led.record_snapshot(decision_id="c2", accepted=True, direction="PUT",
                             symbol="QQQ261219P00400000",
                             quote={"bid": 2.00, "ask": 2.20}, sizing={"qty": 1})
    led.record_event(decision_id=d2, event_type="outcome",
                     payload={"class": "realized", "reason": "stop_loss",
                              "gross_pnl_pct": -10.0, "net_pnl_pct": -14.0})
    rep = component_report(led)
    if rep["n_realized"] != 2:
        print("FAIL: realized count", rep["n_realized"]); ok = False
    if rep["direction"]["CALL"]["win_rate"] != 1.0 or \
       rep["direction"]["PUT"]["win_rate"] != 0.0:
        print("FAIL: direction decomposition", rep["direction"]); ok = False
    if abs(rep["contract_selection"]["avg_cost_drag_pct"] - 4.5) > 1e-9:
        print("FAIL: cost drag (5+4)/2", rep["contract_selection"]); ok = False
    if rep["exit_rules"]["take_profit"]["avg_net_pnl_pct"] != 15.0:
        print("FAIL: exit decomposition", rep["exit_rules"]); ok = False
    if rep["entry_execution"]["n_shortfall"] != 1:
        print("FAIL: shortfall sample count", rep["entry_execution"]); ok = False

    # ---------------- shadow harness --------------------------------------- #
    # Bounded queue: no worker -> third submit drops, nothing raises/blocks.
    sh = ShadowComparator(led, variant="v2-scorer", maxsize=2,
                          start_worker=False)
    cand = {"decision_id": "c1", "symbol": "SPY261219C00500000",
            "direction": "CALL", "score": 8.5, "accepted": True,
            "quote": {"bid": 1.00, "ask": 1.10}}
    frozen = json.dumps(cand, sort_keys=True)
    if not (sh.submit(cand) and sh.submit(dict(cand, decision_id="c2"))):
        print("FAIL: first two submits should queue"); ok = False
    if sh.submit(dict(cand, decision_id="c3")) is not False or sh.dropped != 1:
        print("FAIL: full queue must drop and count", sh.accounting()); ok = False
    if json.dumps(cand, sort_keys=True) != frozen:
        print("FAIL: harness must not mutate the input"); ok = False
    n = sh.drain()
    if n != 2 or sh.written != 2:
        print("FAIL: drain should write queued items", sh.accounting()); ok = False

    # Shadow rows are kind='shadow_candidate', sharing decision ids.
    shadow_rows = led.snapshots(kind="shadow_candidate")
    if len(shadow_rows) != 2:
        print("FAIL: shadow snapshot count", len(shadow_rows)); ok = False
    if shadow_rows and "shadow_variant" not in (shadow_rows[0]["extra_json"] or ""):
        print("FAIL: variant tag missing"); ok = False

    cmpres = compare_shadow_to_baseline(led, "v2-scorer")
    if cmpres["shadow"]["n"] != 2 or cmpres["baseline"]["n"] != 2:
        print("FAIL: comparison counts", cmpres); ok = False
    if cmpres["shared_decision_ids"] != 2:
        print("FAIL: shadow should share decision ids", cmpres); ok = False

    # Write errors are counted, never raised.
    class _Boom:
        def record_snapshot(self, **kw):
            raise RuntimeError("disk full")

    sh2 = ShadowComparator(_Boom(), variant="v3", maxsize=4, start_worker=False)
    sh2.submit(cand)
    sh2.drain()
    if sh2.write_errors != 1 or sh2.written != 0:
        print("FAIL: write error accounting", sh2.accounting()); ok = False

    # Threaded worker drains on close() without hanging.
    sh3 = ShadowComparator(led, variant="v4", maxsize=16, start_worker=True,
                           worker_timeout_s=0.05)
    for i in range(5):
        sh3.submit(dict(cand, decision_id=f"w{i}"))
    sh3.close(timeout_s=2.0)
    acct = sh3.accounting()
    if acct["written"] + acct["dropped"] != 5 or acct["queued"] != 0:
        print("FAIL: close must account for every submit", acct); ok = False

    led.close()
    reg.close()
    print("research_experiments self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
