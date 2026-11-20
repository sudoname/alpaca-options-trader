# Oracle 4.0 Architecture

How the Oracle4 research pipeline (`oracle/ml/`) is layered, how data
flows through it, and where it touches production (one place, env-gated,
observe-only). Companion to `docs/oracle4_runbook.md` (operations) and
`docs/oracle4_experiments.md` (results per phase).

---

## 1. Design stance

Oracle4 is a **challenger research stack** that runs beside the
production single-leg bot without being able to affect it:

- every module is deterministic given `--seed` and produces a JSON/sqlite
  artifact with a content fingerprint;
- every consumer re-validates its inputs (holdout refusal, parity
  checks, negative controls) rather than trusting the producer;
- abstention is a first-class output — missing quotes, unresolvable IV
  or thin history become counted reasons, never silent drops or invented
  values;
- promotion to production is impossible from code alone: the best
  machine verdict is `ELIGIBLE_PENDING_HUMAN_REVIEW`, and the registry
  requires a named human for approval and champion changes.

## 2. Module map (`oracle/ml/`)

```
                         research_ledger.db  (production snapshots, read-only)
                                  │
   ┌──────────────── data layer ──┴─────────────────────────────┐
   │ ingest.py    point-in-time snapshot loading, PIT guards    │
   │ schema.py    sqlite dataset schema + manifest fingerprint  │
   │ features.py  leakage-safe feature construction             │
   │ labels.py    horizon labels (direction/returns), neutral band
   │ dataset.py   CLI: build dataset_v1.sqlite + manifest       │
   └──────────────────────────────┬─────────────────────────────┘
                                  │ dataset_v1.sqlite
   ┌───────────── validation / models ─────────────────────────┐
   │ validation (in train.py/fold logic): purged session        │
   │   walk-forward, embargoed folds, negative controls         │
   │ baselines.py  naive comparators (majority, persistence …)  │
   │ train.py      CLI: classical (ridge logistic etc.) and     │
   │ mlp.py        multitask MLP (torch, pure-NumPy fallback)   │
   │               → OOF predictions + calibration diagnostics  │
   └──────────────────────────────┬─────────────────────────────┘
                                  │ classical.json / mlp.json
   ┌───────────── decision layer ─┴─────────────────────────────┐
   │ valuation.py  scenario Monte Carlo per candidate,          │
   │               all-candidate ranking, decision trade/abstain│
   │ execution.py  fill/slippage calibration from PAPER logs    │
   │ replay.py     paired portfolio replay (challenger vs       │
   │               frozen baseline), friction mults, stress,    │
   │               bootstrap CIs, accounting parity, holdout    │
   └──────────────────────────────┬─────────────────────────────┘
                                  │ replay.json (+ execution_calibration.json)
   ┌───────────── governance layer ┴────────────────────────────┐
   │ promotion.py  14 pre-registered gates → three-tier verdict │
   │ registry.py   champion/challenger registry, SHA-256 pinned │
   │ shadow.py     observe-only forward shadow (JSONL per scan) │
   └────────────────────────────────────────────────────────────┘
```

All artifacts live in `artifacts/oracle4/` (git-ignored, regenerable
from the ledger + seeds).

## 3. Artifact flow

```
research_ledger.db
  └─ dataset.py ──► dataset_v1.sqlite + dataset_v1.manifest.json
       ├─ train.py ──► classical.json, mlp.json  (OOF, per-fold, controls)
       ├─ valuation.py ──► valuation.json        (per-candidate EV, abstentions)
       └─ execution.py ◄─ server_data/<date>/execution_calibration.jsonl
                        ──► execution_calibration.json (PAPER-labeled)
  replay.py(dataset, valuation, execution) ──► replay.json
  promotion.py(replay, execution [, forward-evidence]) ──► promotion_report.json
  registry.py ──► registry.json (append-only history, atomic writes)
  shadow.py  ──► oracle4_shadow.jsonl (one line per production scan)
```

Each JSON artifact embeds a `fingerprint`; `promotion_report.json`
records the replay fingerprint it judged, so a report can never be
mistaken for a verdict on different data.

## 4. The single production touchpoint

`run_alpaca_intraday.py::_rank_and_enter` contains a 16-line hook,
placed after candidates are scored and before the entry loop:

- gated on `os.getenv("ENABLE_ORACLE4_SHADOW") == "true"` — with the
  flag off (default) the cost is one `getenv` and the import never runs;
- wrapped in `try/except Exception` that only logs
  `[ORACLE4-SHADOW] failed (ignored)` — a research bug can never block,
  delay or reorder baseline entries;
- passes the exact candidate dicts production just scored plus
  `trader.get_price_history`, so baseline and challenger see the same
  frozen input;
- `oracle/ml/shadow.py` contains **no broker imports** — enforced by
  `test_no_broker_imports_in_module_source`, which scans the module
  source for `alpaca`, `submit_order`, `TradingClient`, etc.

The shadow writes one JSON record per scan (baseline score order,
challenger scenario-EV ranking, model identity from the registry,
missingness, inference ms, and an explicit "no order was submitted"
note). Because `get_price_history` returns timestamp-less daily closes,
scenario history is mapped onto a synthetic uniform grid and every
record carries `history_grid_assumed: true` — a known, labeled
approximation, not hidden.

## 5. Governance invariants (enforced by code + tests)

| invariant | where enforced |
|---|---|
| challenger never submits/modifies/cancels orders | shadow.py has no broker imports (test-enforced); hook is fail-open |
| shadow defaults OFF | `ShadowConfig.from_env()` requires explicit truthy flag; default-off test |
| artifact immutability | registry pins SHA-256; re-register same id with different bytes → `ValueError: immutable` |
| no automatic approval/promotion | `approve_model`/`set_champion` reject empty `approved_by`; no code path flips roles without a human string |
| rollback is cheap and auditable | rollback = `set_champion` to prior id; checksums prove no files moved; append-only history |
| thin evidence never relaxes gates | promotion verdict forces `INSUFFICIENT_EVIDENCE` when any volume/forward gate lacks data |
| mark-derived outcomes can't qualify | `evidence_basis` gate requires `live_fill_based: true` forward evidence |
| no PROMOTE output exists | verdict lattice: INSUFFICIENT_EVIDENCE → DO_NOT_PROMOTE → ELIGIBLE_PENDING_HUMAN_REVIEW |
| replay accounting honest | `parity.ok` invariant + holdout-date refusal checked inside the artifact |
| validation not leaking | purged/embargoed session folds; shuffled-label negative controls must sit at chance |

## 6. Promotion gates (pre-registered)

Defined once in `promotion.GATES_PREREGISTERED` and copied verbatim
into every report: ≥60 forward sessions, ≥200 resolved opportunities,
≥100 actual fills, expectancy bootstrap LB > 0, paired uplift bootstrap
LB > 0, net positive at 1.5× friction, 2× friction stress examined,
Sharpe (daily, annualized) > 1, profit factor > 1.3, max drawdown ≤ 20%
and no worse than baseline, calibration improved, coverage disparity
≤ 10%, evidence basis live-fill-based, mandatory invariants (parity,
holdout refusal). Statuses are PASS / FAIL / INSUFFICIENT; thresholds
are constants, not parameters — there is deliberately no CLI flag to
change them.

Current real-artifact verdict: **INSUFFICIENT_EVIDENCE** (no forward
sessions yet; outcomes mark-derived). That is the honest expected state
until the forward shadow accumulates live-scan evidence.

## 7. Testing

248 tests across 10 files (`tests/test_oracle4_*.py`) cover dataset
PIT-correctness, fold purging + negative controls, model training and
fallbacks, valuation abstention, execution calibration provenance,
replay parity/holdout, registry immutability and human-only approval,
shadow default-off/fail-open/no-broker-imports, and the promotion
verdict lattice. See runbook §4 for the exact command.
