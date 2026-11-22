"""Oracle4 Phase 7b: observe-only forward shadow for the challenger stack.

When ``ENABLE_ORACLE4_SHADOW=true`` the intraday scheduler hands each scan's
frozen candidate list to :func:`maybe_shadow_scan`.  The challenger (the
Phase 5 scenario-valuation ranker) then values and ranks exactly what the
baseline saw, and a single JSONL record is appended containing, side by
side: the model identity (registry id / checksum / git SHA), the flags in
effect, inference duration, input missingness, scenario drift proxies, the
challenger's ranking + abstentions, and the baseline's own ordering.

Hard guarantees:

* **No order submission.**  This module never imports a broker client and
  never calls anything that places, replaces, or cancels orders.  Its only
  side effect is appending one JSON line per scan.
* **Default OFF.**  All flags default to disabled; the production path pays
  one ``os.getenv`` check when the feature is off.
* **Fail-open.**  Any exception inside the shadow is swallowed (and
  printed) — production ranking and entry are byte-identical with the flag
  on or off.

History source (4.0.2 P0-2: real bars only):

* Per-underlying history is read exclusively from the read-only bars
  cache (:mod:`oracle.ml.bars`) configured via ``ORACLE4_BARS_DB``:
  actual timestamped completed 1-minute bars under the completed-bar
  PIT rule, bounded by a freshness requirement
  (``ORACLE4_BARS_MAX_STALENESS_MIN``, default 15 minutes).  When the
  store cannot serve an underlying — store missing, symbol unknown, no
  coverage, stale, insufficient history — the scan **abstains for that
  underlying with the structured ``BarsUnavailable`` reason**.  When
  ``ORACLE4_BARS_DB`` is unset every underlying abstains with
  ``BARS_DB_UNSET``.  Daily closes are never placed on an assumed
  intraday grid: the legacy synthetic-grid fallback was removed in
  4.0.2 (P0-2) and untimestamped close lists are no longer accepted.

Flags (all optional):

* ``ENABLE_ORACLE4_SHADOW``    — master switch (default ``false``)
* ``ORACLE4_MODEL_ID``         — registry id whose pinned artifact is
  hash-verified, loaded and actually invoked per candidate (4.0.1-B via
  :mod:`oracle.ml.forecast`); records carry real logits/probabilities,
  quantiles, RV, CIF, latency and input digest, or a structured
  abstention reason — identity metadata alone never counts as inference
* ``ORACLE4_MAX_CANDIDATES``   — truncate each scan's input (default 16)
* ``ORACLE4_MAX_SCENARIOS``    — MC paths per underlying (default 512)
* ``ORACLE4_SHADOW_JSONL``     — output path (default oracle4_shadow.jsonl)
* ``ORACLE4_REGISTRY``         — registry path for model identity lookup
* ``ORACLE4_BARS_DB``          — read-only bars cache path (default unset:
  every underlying abstains with ``BARS_DB_UNSET``)
* ``ORACLE4_BARS_MAX_STALENESS_MIN`` — max age in minutes of the newest
  completed bar before the underlying abstains as ``STALE`` (default 15)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from oracle.ml import books as bk
from oracle.ml import valuation as val

SCHEMA_VERSION = 1
RECORD_TYPE = "oracle4_shadow_scan"
DEFAULT_JSONL = "oracle4_shadow.jsonl"


def _truthy(s: Optional[str]) -> bool:
    return (s or "").strip().lower() in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class ShadowConfig:
    enabled: bool = False
    model_id: str = ""
    max_candidates: int = 16
    max_scenarios: int = 512
    jsonl_path: str = DEFAULT_JSONL
    registry_path: str = os.path.join("artifacts", "oracle4", "registry.json")
    bars_db: str = ""   # read-only bars cache; unset = abstain (BARS_DB_UNSET)
    bars_max_staleness_min: float = 15.0  # newest completed bar max age

    @classmethod
    def from_env(cls) -> "ShadowConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, "") or default)
            except (TypeError, ValueError):
                return default

        def _pos_float(name: str, default: float) -> float:
            try:
                v = float(os.getenv(name, "") or default)
            except (TypeError, ValueError):
                return default
            return v if v > 0 else default
        return cls(
            enabled=_truthy(os.getenv("ENABLE_ORACLE4_SHADOW")),
            model_id=(os.getenv("ORACLE4_MODEL_ID") or "").strip(),
            max_candidates=max(1, _int("ORACLE4_MAX_CANDIDATES", 16)),
            max_scenarios=_int("ORACLE4_MAX_SCENARIOS", 512),
            jsonl_path=(os.getenv("ORACLE4_SHADOW_JSONL") or DEFAULT_JSONL),
            registry_path=(os.getenv("ORACLE4_REGISTRY")
                           or cls.registry_path),
            bars_db=(os.getenv("ORACLE4_BARS_DB") or "").strip(),
            bars_max_staleness_min=_pos_float(
                "ORACLE4_BARS_MAX_STALENESS_MIN", 15.0),
        )


# --------------------------------------------------------------------------- #
# freezing production candidates into valuation inputs
# --------------------------------------------------------------------------- #

def _flt(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN -> None


def freeze_candidate(cand: Dict[str, Any], i: int, scan_id: str,
                     now: datetime) -> val.Candidate:
    """Map one scheduler candidate (``_evaluate`` output) to a frozen
    valuation Candidate.  Missing fields stay None so the valuation layer's
    abstention reasons account for them honestly."""
    opt = cand.get("opt") or {}
    symbol = opt.get("symbol")
    bid, ask = _flt(opt.get("bid")), _flt(opt.get("ask"))
    quote_mid = spread_abs = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        quote_mid = (bid + ask) / 2.0
        spread_abs = ask - bid

    contract = val.parse_occ_symbol(symbol)
    dte = _flt(opt.get("dte"))
    if dte is None:
        exp = (contract or {}).get("expiration") or opt.get("expiration")
        try:
            exp_d = datetime.strptime(str(exp), "%Y-%m-%d").date()
            dte = float((exp_d - now.date()).days)
        except (TypeError, ValueError):
            dte = None

    underlying = (cand.get("sym") or opt.get("underlying")
                  or (contract or {}).get("underlying") or "")
    return val.Candidate(
        decision_id=f"{scan_id}:{underlying}:{i}",
        symbol=symbol,
        underlying=str(underlying),
        session=now.date().isoformat(),
        as_of=now,
        kind="shadow",
        features={
            "quote_mid": quote_mid,
            "spread_abs": spread_abs,
            "dte_days": dte,
            "underlying_price": _flt(cand.get("underlying_price")),
            "baseline_score": _flt(cand.get("score")),
        },
        contract=contract,
    )


# --------------------------------------------------------------------------- #
# model identity (registry lookup, fail-open)
# --------------------------------------------------------------------------- #

def _model_identity(cfg: ShadowConfig) -> Dict[str, Any]:
    ident: Dict[str, Any] = {
        "model_id": cfg.model_id or None,
        "registry_entry_found": False,
        "checksums": None,
        "git_sha_registered": None,
        "approval_state": None,
        "role": None,
    }
    try:
        from oracle.ml import registry as regmod
        ident["git_sha_runtime"] = regmod.current_git_sha()
        if cfg.model_id:
            entry = regmod.get_model(cfg.registry_path, cfg.model_id)
            if entry:
                ident.update({
                    "registry_entry_found": True,
                    "checksums": entry.get("checksums"),
                    "git_sha_registered": entry.get("git_sha"),
                    "approval_state": entry.get("approval_state"),
                    "role": entry.get("role"),
                })
    except Exception as e:  # registry problems never break the shadow
        ident["registry_error"] = str(e)
    return ident


# --------------------------------------------------------------------------- #
# model inference (4.0.1-B: verified load + actual invocation, fail-open)
# --------------------------------------------------------------------------- #

def _decision_record(cand: Dict[str, Any], frozen: val.Candidate,
                     now: datetime) -> Any:
    """Map a scheduler candidate + its frozen valuation twin to a
    :class:`oracle.ml.schema.DecisionRecord` so the **exact** point-in-time
    feature builder (:func:`oracle.ml.features.build_feature_row`) runs on
    live-shadow inputs — no separate shadow approximation."""
    from oracle.ml import schema as sch
    opt = cand.get("opt") or {}
    contract = frozen.contract or {}
    expiration = contract.get("expiration") or opt.get("expiration")
    return sch.DecisionRecord(
        decision_id=frozen.decision_id,
        snapshot_id=frozen.decision_id,
        kind="shadow_candidate",
        accepted=False,
        reject_code=None,
        as_of=now,
        available_at=now,
        underlying=frozen.underlying,
        symbol=frozen.symbol,
        expiration=str(expiration) if expiration else None,
        strike=_flt(contract.get("strike")),
        option_type=contract.get("option_type"),
        multiplier=_flt(opt.get("multiplier")),
        quote_bid=_flt(opt.get("bid")),
        quote_ask=_flt(opt.get("ask")),
        underlying_price=_flt(cand.get("underlying_price")),
        iv=_flt(opt.get("iv")),
        delta=_flt(opt.get("delta")),
        direction=opt.get("direction") or cand.get("direction"),
        score=_flt(cand.get("score")),
        p_win=_flt(cand.get("p_win") or opt.get("p_win")),
        ev=_flt(cand.get("ev") or opt.get("ev")),
        signals=dict(cand.get("signals") or {}),
    )


def _run_forecasts(cfg: ShadowConfig,
                   frozen_src: Sequence[Dict[str, Any]],
                   cands: Sequence[val.Candidate],
                   bar_rows_by_u: Dict[str, List[Any]],
                   bars_mode: bool,
                   now: datetime) -> tuple:
    """Load the configured registry model (verified) and invoke it once per
    frozen candidate.  Returns ``(inference_info, forecasts)`` where
    ``forecasts`` maps decision_id -> Forecast dict or structured
    abstention.  Never raises.  4.0.1-C: the caller feeds these into
    scenario generation; any refusal falls open to the labeled baseline.
    """
    info: Dict[str, Any] = {
        "attempted": bool(cfg.model_id),
        "loaded": False,
        "n_forecasts": 0,
        "n_abstained": 0,
        "forecast_ms_total": 0.0,
    }
    forecasts: Dict[str, Any] = {}
    if not cfg.model_id:
        info["reason"] = "MODEL_ID_UNSET"
        return info, forecasts

    tf0 = time.perf_counter()
    try:
        from oracle.ml import forecast as fcmod
        from oracle.ml import features as featmod
        from oracle.ml import schema as sch
    except Exception as e:
        info["reason"] = "IMPORT_ERROR"
        info["detail"] = str(e)
        info["forecast_ms_total"] = (time.perf_counter() - tf0) * 1000.0
        return info, forecasts

    try:
        provider = fcmod.get_provider(cfg.registry_path, cfg.model_id)
    except fcmod.ForecastUnavailable as e:
        info["forecast_unavailable"] = e.to_dict()
        info["n_abstained"] = len(cands)
        info["forecast_ms_total"] = (time.perf_counter() - tf0) * 1000.0
        for c in cands:
            forecasts[c.decision_id] = e.to_dict()
        return info, forecasts
    except Exception as e:  # defensive: provider layer must stay fail-open
        info["reason"] = "PROVIDER_ERROR"
        info["detail"] = str(e)
        info["forecast_ms_total"] = (time.perf_counter() - tf0) * 1000.0
        return info, forecasts

    info.update({
        "loaded": True,
        "d_in": 2 * len(provider.feature_names),
        "temperature": provider.temperature,
        "horizon": provider.horizon,
        "checksums": provider.checksums,
    })

    if not bars_mode:
        # 4.0.2 P0-2: no bars store configured means no genuine timestamped
        # history exists — abstain per candidate rather than fabricate.
        e = fcmod.ForecastUnavailable(
            "BARS_DB_UNSET",
            "ORACLE4_BARS_DB unset: no timestamped bar history to build "
            "features on", model_id=cfg.model_id)
        for c in cands:
            forecasts[c.decision_id] = e.to_dict()
        info["n_abstained"] = len(cands)
        info["forecast_ms_total"] = (time.perf_counter() - tf0) * 1000.0
        return info, forecasts

    obs_by_u: Dict[str, List[Any]] = {}
    for u, bar_rows in bar_rows_by_u.items():
        obs_by_u[u] = [
            sch.UnderlyingObs(
                underlying=u, event_at=b.end_utc, available_at=b.end_utc,
                price=float(b.close), source_snapshot_id=f"bars:{u}")
            for b in bar_rows]
    index = featmod.ObsIndex(obs_by_u)

    for cand_dict, frozen in zip(frozen_src, cands):
        try:
            dec = _decision_record(cand_dict, frozen, now)
            frow = featmod.build_feature_row(dec, index)
            fc = provider.forecast(frow.features, as_of=now)
            forecasts[frozen.decision_id] = fc.to_dict()
            info["n_forecasts"] += 1
        except fcmod.ForecastUnavailable as e:
            forecasts[frozen.decision_id] = e.to_dict()
            info["n_abstained"] += 1
        except Exception as e:
            forecasts[frozen.decision_id] = {
                "available": False, "reason": "INFERENCE_ERROR",
                "detail": str(e), "model_id": cfg.model_id}
            info["n_abstained"] += 1

    info["forecast_ms_total"] = (time.perf_counter() - tf0) * 1000.0
    return info, forecasts


# --------------------------------------------------------------------------- #
# the shadow scan itself
# --------------------------------------------------------------------------- #

def shadow_scan(candidates: Sequence[Dict[str, Any]], *,
                scan_id: Optional[str] = None,
                step_minutes: float = 5.0,
                now: Optional[datetime] = None,
                cfg: Optional[ShadowConfig] = None,
                out_path: Optional[str] = None,
                base_seed: int = 7) -> Dict[str, Any]:
    """Value + rank one scan's frozen candidates; append one JSONL record.

    Observe-only: returns the record it wrote.  Never submits orders.
    """
    t0 = time.perf_counter()
    cfg = cfg or ShadowConfig.from_env()
    now = now or datetime.now(timezone.utc)
    scan_id = scan_id or str(uuid.uuid4())

    n_in = len(candidates)
    frozen_src = list(candidates)[:cfg.max_candidates]
    truncated = n_in - len(frozen_src)

    # baseline view: what production's score ordering considered, pre-freeze
    baseline_order = sorted(
        ({"sym": c.get("sym"),
          "symbol": (c.get("opt") or {}).get("symbol"),
          "score": _flt(c.get("score"))} for c in frozen_src),
        key=lambda r: -(r["score"] or float("-inf")))

    cands = [freeze_candidate(c, i, scan_id, now)
             for i, c in enumerate(frozen_src)]

    vcfg = val.ValuationConfig(n_paths=int(cfg.max_scenarios),
                               step_minutes=float(step_minutes))

    # one scenario set per underlying, shared across its contracts
    bars_mode = bool(cfg.bars_db)
    bar_store = None
    bars_note: Optional[Dict[str, Any]] = None
    if bars_mode:
        try:
            from oracle.ml import bars as barsmod
            bar_store = barsmod.BarStore(cfg.bars_db, readonly=True)
        except Exception as e:  # store missing/unreadable: abstain per-underlying
            bars_note = (e.to_dict() if hasattr(e, "to_dict")
                         else {"reason": "STORE_ERROR", "detail": str(e)})
    else:
        # 4.0.2 P0-2: real bars are the ONLY history source.  With no store
        # configured every underlying abstains -- the legacy synthetic-grid
        # fallback (daily closes on an assumed intraday grid) was removed.
        bars_note = {
            "reason": "BARS_DB_UNSET",
            "detail": ("ORACLE4_BARS_DB unset: no timestamped bar history; "
                       "synthetic-grid fallback was removed (4.0.2 P0-2)")}

    scen_by_u: Dict[str, Any] = {}
    scen_meta: Dict[str, Dict[str, Any]] = {}
    bar_rows_by_u: Dict[str, List[Any]] = {}  # 4.0.1-B: retained for features
    # ---- pass 1: history per underlying (4.0.1-C: loaded before forecasts
    # so the verified model can condition the shared scenario set) ----
    hist_by_u: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        u = c.underlying
        if not u or u in hist_by_u:
            continue
        s0 = c.features.get("underlying_price")
        obs: List[Any] = []
        bars_unavailable: Optional[Dict[str, Any]] = None
        # 4.0.2 P0-2: real completed bars are the only history source.  On
        # unavailability (store unset/missing, stale, thin coverage) abstain
        # with the structured reason -- never fabricate a synthetic grid.
        if bar_store is None:
            bars_unavailable = bars_note or {"reason": "STORE_MISSING"}
        else:
            try:
                from oracle.ml import bars as barsmod
                bar_rows = bar_store.load_bars(
                    u, 60,
                    start=now - timedelta(days=7), end=now, as_of=now,
                    rth_only=True, min_bars=vcfg.min_history_returns,
                    max_staleness=timedelta(
                        minutes=cfg.bars_max_staleness_min))
                obs = [(b.ts_utc, b.end_utc, b.close) for b in bar_rows]
                bar_rows_by_u[u] = list(bar_rows)
            except barsmod.BarsUnavailable as e:
                bars_unavailable = e.to_dict()
        hist_by_u[u] = {"obs": obs, "s0": s0,
                        "history_source": "bars",
                        "bars_unavailable": bars_unavailable}

    if bar_store is not None:
        try:
            bar_store.close()
        except Exception:
            pass

    # 4.0.1-B/C: verified model load + actual per-candidate invocation,
    # run BEFORE scenario generation so forecasts condition the shared
    # scenario sets.  Fail-open: any refusal is a structured abstention
    # and scenarios fall back to the explicitly labeled baseline.
    inference_info, forecasts = _run_forecasts(
        cfg, frozen_src, cands, bar_rows_by_u, bars_mode, now)
    fc_by_u: Dict[str, Any] = {}
    for c in cands:
        u = c.underlying
        if not u:
            continue
        fc = forecasts.get(c.decision_id)
        if fc is None:
            continue
        cur = fc_by_u.get(u)
        if cur is None or (not cur.get("available") and fc.get("available")):
            fc_by_u[u] = fc

    # ---- pass 2: one scenario set per underlying, shared across its
    # contracts (common random numbers), forecast-conditioned when the
    # underlying has a usable forecast ----
    for u, h in hist_by_u.items():
        bars_unavailable = h["bars_unavailable"]
        if bars_unavailable is not None:
            scen_by_u[u] = None
            scen_meta[u] = {
                "status": "bars_unavailable:"
                          + str(bars_unavailable.get("reason")),
                "bars_unavailable": bars_unavailable,
                "n_history_points": 0,
                "history_error": None,
                "history_source": h["history_source"],
            }
            continue
        s0 = h["s0"]
        scen, reason = val.generate_scenarios(
            h["obs"], now, s0 if s0 else 0.0, vcfg,
            val.scenario_seed(base_seed, u, scan_id),
            forecast=fc_by_u.get(u))
        scen_by_u[u] = scen
        meta = dict(scen.meta) if scen is not None else {}
        meta.update({"status": reason, "n_history_points": len(h["obs"]),
                     "history_error": None,
                     "history_source": h["history_source"]})
        scen_meta[u] = meta

    valuations: List[Dict[str, Any]] = []
    for c in cands:
        scen = scen_by_u.get(c.underlying)
        if scen is None:
            reason = scen_meta.get(c.underlying, {}).get(
                "status", "no_scenarios")
            valuations.append(val._abstain(c, f"no_scenarios:{reason}"))
            continue
        priced, abstain = val.reprice_candidate(c, scen, vcfg)
        if abstain is not None:
            valuations.append(abstain)
            continue
        valuations.append(val.value_candidate(
            c, priced, vcfg, friction=1.0,
            forecast=forecasts.get(c.decision_id)))

    ranked = val.rank_candidates(valuations, vcfg)

    missingness: Dict[str, int] = {}
    for v in valuations:
        if v.get("status") != "ok":
            r = str(v.get("reason"))
            missingness[r] = missingness.get(r, 0) + 1

    model_ident = _model_identity(cfg)
    model_ident["inference"] = inference_info

    # Strategy-book assignment (Stage G): observe-only annotation, never a
    # gate.  Fail-open so a books error can never break the shadow record.
    try:
        book_assignments = bk.assign_books(
            cands, now=now, scan_id=scan_id,
            bars_mode="bars" if bars_mode else "bars_db_unset")
        books_summary = bk.summarize_books(book_assignments)
    except Exception:  # pragma: no cover - defensive fail-open
        book_assignments, books_summary = {}, {}

    inference_ms = (time.perf_counter() - t0) * 1000.0
    record = {
        "schema_version": SCHEMA_VERSION,
        "type": RECORD_TYPE,
        "at": now.isoformat(),
        "scan_id": scan_id,
        "model": model_ident,
        "forecasts": forecasts,
        "flags": {
            "ENABLE_ORACLE4_SHADOW": cfg.enabled,
            "ORACLE4_MODEL_ID": cfg.model_id,
            "ORACLE4_MAX_CANDIDATES": cfg.max_candidates,
            "ORACLE4_MAX_SCENARIOS": cfg.max_scenarios,
            "ORACLE4_BARS_DB": cfg.bars_db or None,
            "ORACLE4_BARS_MAX_STALENESS_MIN": cfg.bars_max_staleness_min,
        },
        "inference_ms": inference_ms,
        "bars_db": cfg.bars_db or None,
        "input": {
            "n_candidates_in": n_in,
            "n_candidates_truncated": truncated,
            "n_valued": len(valuations),
            "n_eligible": ranked["n_eligible"],
            "missingness": missingness,
        },
        "scenarios": scen_meta,
        "challenger": {
            "decision": ranked["decision"],
            "selected_symbol": ranked["selected_symbol"],
            "selected_ev": ranked["selected_ev"],
            "ranking": ranked["ranking"],
            "excluded": ranked["excluded"],
        },
        "baseline": {
            "top_sym": baseline_order[0]["sym"] if baseline_order else None,
            "top_symbol": (baseline_order[0]["symbol"]
                           if baseline_order else None),
            "top_score": (baseline_order[0]["score"]
                          if baseline_order else None),
            "order": baseline_order,
        },
        "books": {
            "by_decision": book_assignments,
            "summary": books_summary,
        },
        "outcome": {
            "state": "unresolved",
            "resolved_at": None,
            "resolver": None,
        },
        "note": ("observe-only shadow: no order was submitted; paper/live "
                 "entries remain the baseline's alone"),
    }

    path = out_path or cfg.jsonl_path
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def maybe_shadow_scan(candidates: Sequence[Dict[str, Any]],
                      **kw: Any) -> Optional[Dict[str, Any]]:
    """Flag-gated, fail-open entry point for the scheduler hook.

    Returns None (and does nothing) when ``ENABLE_ORACLE4_SHADOW`` is not
    truthy; otherwise runs :func:`shadow_scan` and swallows any error.
    """
    cfg = ShadowConfig.from_env()
    if not cfg.enabled:
        return None
    try:
        return shadow_scan(candidates, cfg=cfg, **kw)
    except Exception as e:  # never disturb production
        try:
            print(f"[ORACLE4-SHADOW] scan failed (ignored): {e}")
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------- #
# read-only report over the shadow log
# --------------------------------------------------------------------------- #

def _quantiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    def q(p: float) -> float:
        k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[k]
    return {"min": s[0], "p50": q(0.5), "p90": q(0.9), "max": s[-1],
            "mean": sum(s) / len(s)}


def summarize_log(path: str) -> Dict[str, Any]:
    """Aggregate the shadow JSONL into a read-only report dict."""
    n = 0
    bad = 0
    inference: List[float] = []
    model_ids: Dict[str, int] = {}
    abstain: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    agree = disagree = 0
    clamped = unclamped = 0
    truncated_scans = 0
    books_candidates: Dict[str, int] = {}
    books_paired_groups = 0
    books_annotations: Dict[str, int] = {}
    outcome_states: Dict[str, int] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if rec.get("type") != RECORD_TYPE:
                    continue
                n += 1
                if isinstance(rec.get("inference_ms"), (int, float)):
                    inference.append(float(rec["inference_ms"]))
                mid = (rec.get("model") or {}).get("model_id") or "(none)"
                model_ids[mid] = model_ids.get(mid, 0) + 1
                for r, c in ((rec.get("input") or {}).get(
                        "missingness") or {}).items():
                    abstain[r] = abstain.get(r, 0) + int(c)
                ch = rec.get("challenger") or {}
                decisions[ch.get("decision") or "?"] = \
                    decisions.get(ch.get("decision") or "?", 0) + 1
                base_top = (rec.get("baseline") or {}).get("top_symbol")
                if ch.get("decision") == "trade" and base_top is not None:
                    if ch.get("selected_symbol") == base_top:
                        agree += 1
                    else:
                        disagree += 1
                for meta in (rec.get("scenarios") or {}).values():
                    if meta.get("vol_scale_clamped_ood") is True:
                        clamped += 1
                    elif meta.get("vol_scale_clamped_ood") is False:
                        unclamped += 1
                if ((rec.get("input") or {}).get(
                        "n_candidates_truncated") or 0) > 0:
                    truncated_scans += 1
                bks = rec.get("books") or {}
                summ = bks.get("summary") or {}
                for b, c in (summ.get("candidates_by_book") or {}).items():
                    books_candidates[b] = books_candidates.get(b, 0) + int(c)
                books_paired_groups += int(summ.get("paired_groups") or 0)
                for info in (bks.get("by_decision") or {}).values():
                    for a in (info or {}).get("annotations") or []:
                        books_annotations[a] = \
                            books_annotations.get(a, 0) + 1
                st = (rec.get("outcome") or {}).get("state") or "(absent)"
                outcome_states[st] = outcome_states.get(st, 0) + 1
    denom = agree + disagree
    vol_denom = clamped + unclamped
    return {
        "log": path,
        "n_scans": n,
        "n_bad_lines": bad,
        "inference_ms": _quantiles(inference),
        "model_ids": model_ids,
        "decisions": decisions,
        "abstain_reasons": abstain,
        "challenger_vs_baseline_top_agreement": (
            agree / denom if denom else None),
        "n_trade_decisions_compared": denom,
        "vol_scale_clamp_rate": (clamped / vol_denom if vol_denom else None),
        "scans_truncated": truncated_scans,
        "books": {
            "candidates_by_book": books_candidates,
            "paired_groups": books_paired_groups,
            "annotations": books_annotations,
        },
        "outcome_states": outcome_states,
        "note": "read-only summary of observe-only shadow records",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Oracle4 shadow log report")
    ap.add_argument("--report", action="store_true",
                    help="summarize the shadow JSONL (read-only)")
    ap.add_argument("--log", default=DEFAULT_JSONL)
    args = ap.parse_args(argv)
    if args.report:
        print(json.dumps(summarize_log(args.log), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
