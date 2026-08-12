"""
Phase E 4a — Oracle probability recorder (analytics only).

Records EVERY Oracle directional head it is shown — including no-trades —
to an append-only JSONL ledger, then resolves each prediction later against
the realised move of the underlying. Unlike episodes.db (which only carries
opened trades) this must score beliefs we did NOT act on, so it mirrors the
``candidate_resolution.jsonl`` append-only / fold-by-id idiom rather than the
episode store.

    prediction line   frozen at decision time (p_call/p_put/p_no_trade, mode,
                      expected_move_pct, entry_underlying_price, horizon_end)
    resolution line   appended once the horizon has actually elapsed, carrying
                      fwd_underlying_price, realized_return_pct/direction,
                      correct, brier_call

Point-in-time / no look-ahead: ``horizon_end`` is frozen at record time and a
prediction is resolved ONLY once ``now >= horizon_end``; ``price_fn`` is asked
for the underlying price AT ``horizon_end`` (a time strictly after ``as_of``),
so a backtest MarketView cannot leak a future bar into the label.

STRICTLY analytics: this module records and resolves beliefs. It never opens,
closes, sizes, blocks or alters any trade, and every public entry point is
fail-open (returns None / 0 rather than raising).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

LOG_TAG = "[ORACLE_PROB_RECORDER]"

JSONL_FILE_DEFAULT = "oracle_prob_predictions.jsonl"
RECORD_TYPE_PREDICTION = "prediction"
RECORD_TYPE_RESOLUTION = "resolution"

STATUS_RESOLVED = "resolved"
STATUS_MISSING_PRICE = "missing_price"

# Mode-dependent resolution horizon. Intraday beliefs are scored at the next
# close (~one trading session of minutes); swing beliefs at min(dte, N) days.
INTRADAY_HORIZON_MIN_DEFAULT = 390          # one RTH session
SWING_HORIZON_DAYS_DEFAULT = 5


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #
def _now(now: Optional[datetime]) -> datetime:
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _jsonl_path(jsonl_path: Optional[str] = None) -> str:
    """Resolve the ledger path (arg > env > default). Fail-open."""
    if jsonl_path:
        return jsonl_path
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_str(
            "ORACLE_PROB_PREDICTIONS_JSONL", JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def horizon_end(as_of: datetime, mode: Optional[str], dte=None,
                intraday_minutes: int = INTRADAY_HORIZON_MIN_DEFAULT,
                swing_days: int = SWING_HORIZON_DAYS_DEFAULT) -> datetime:
    """Frozen resolution target for a prediction. Pure.

    intraday -> as_of + one session of minutes (the next close);
    swing/other -> as_of + min(dte, swing_days) calendar days (>=1).
    """
    m = str(mode or "").strip().lower()
    if m == "intraday":
        return as_of + timedelta(minutes=max(1, int(intraday_minutes)))
    days = int(swing_days)
    d = _to_float(dte)
    if d is not None:
        days = min(int(d), int(swing_days))
    return as_of + timedelta(days=max(1, days))


def _predicted_direction(p_call: Optional[float],
                         p_put: Optional[float]) -> str:
    pc = _to_float(p_call) or 0.0
    pp = _to_float(p_put) or 0.0
    if pc > pp:
        return "call"
    if pp > pc:
        return "put"
    return "none"


def _realized_direction(ret_pct: Optional[float],
                        flat_band_pct: float = 0.0) -> str:
    r = _to_float(ret_pct)
    if r is None:
        return "unknown"
    if r > flat_band_pct:
        return "up"
    if r < -flat_band_pct:
        return "down"
    return "flat"


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
def record_prediction(fields: dict, *, jsonl_path: Optional[str] = None,
                      now: Optional[datetime] = None) -> Optional[str]:
    """Append ONE frozen prediction line. Returns prediction_id or None
    (fail-open). ``fields`` is typically the Oracle head payload plus symbol,
    mode, trend_horizon, expected_move_pct, entry_underlying_price and dte.
    """
    try:
        f = dict(fields or {})
        symbol = str(f.get("symbol") or "").upper()
        if not symbol:
            return None
        prob = f.get("probability") if isinstance(f.get("probability"),
                                                  dict) else f
        p_call = _to_float(prob.get("p_call"))
        p_put = _to_float(prob.get("p_put"))
        p_no_trade = _to_float(prob.get("p_no_trade"))
        if p_call is None and p_put is None and p_no_trade is None:
            return None
        ts = _now(now)
        as_of = _parse_ts(f.get("as_of")) or ts
        pid = f.get("prediction_id") or uuid.uuid4().hex[:12]
        h_end = horizon_end(as_of, f.get("mode"), f.get("dte"))
        rec = {
            "record_type": RECORD_TYPE_PREDICTION,
            "prediction_id": pid,
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "recorded_at": ts.isoformat(),
            "mode": f.get("mode"),
            "trend_horizon": f.get("trend_horizon"),
            "p_call": p_call,
            "p_put": p_put,
            "p_no_trade": p_no_trade,
            "expected_move_pct": _to_float(f.get("expected_move_pct")),
            "prior": _to_float(f.get("prior")),
            "dte": f.get("dte"),
            "entry_underlying_price": _to_float(
                f.get("entry_underlying_price")),
            "horizon_end": h_end.isoformat(),
            "resolution_status": None,
        }
        _append_jsonl(rec, _jsonl_path(jsonl_path))
        return pid
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} record ignored: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Load (fold by id)
# --------------------------------------------------------------------------- #
def load_predictions(jsonl_path: Optional[str] = None) -> List[dict]:
    """Fold the append-only ledger by prediction_id (last non-null wins so a
    resolution snapshot overrides the frozen prediction). Tolerates a missing
    file, malformed lines and partial records. Never raises."""
    path = _jsonl_path(jsonl_path)
    folded: Dict[str, dict] = {}
    order: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                pid = rec.get("prediction_id")
                if not pid:
                    continue
                if pid not in folded:
                    folded[pid] = {}
                    order.append(pid)
                merged = folded[pid]
                for k, v in rec.items():
                    if v is not None or k not in merged:
                        merged[k] = v
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"{LOG_TAG} load ignored: {exc}")
        return []
    return [folded[pid] for pid in order]


def _call_price_fn(price_fn, symbol, at_dt):
    """Call price_fn(symbol, at_dt) if it accepts the target time, else
    price_fn(symbol). Fail-open to None."""
    try:
        return price_fn(symbol, at_dt)
    except TypeError:
        try:
            return price_fn(symbol)
        except Exception:
            return None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Resolve
# --------------------------------------------------------------------------- #
def resolve_predictions(price_fn: Callable[..., Optional[float]], *,
                        now: Optional[datetime] = None,
                        jsonl_path: Optional[str] = None,
                        flat_band_pct: float = 0.0) -> int:
    """Resolve every prediction whose ``horizon_end`` has elapsed by asking
    ``price_fn`` for the underlying price AT ``horizon_end`` (a time strictly
    after ``as_of`` — no look-ahead). Appends one resolution line per resolved
    prediction (append-only; never rewrites). Returns the count newly resolved.

    ``price_fn`` may be ``price_fn(symbol, at_dt)`` (point-in-time, preferred)
    or ``price_fn(symbol)``. Never raises.
    """
    try:
        path = _jsonl_path(jsonl_path)
        recs = load_predictions(path)
        ref_now = _now(now)
        resolved = 0
        for rec in recs:
            if rec.get("resolution_status") == STATUS_RESOLVED:
                continue  # finalised; never re-resolve
            h_end = _parse_ts(rec.get("horizon_end"))
            if h_end is None or ref_now < h_end:
                continue  # not yet due -> append nothing (anti look-ahead)
            symbol = rec.get("symbol")
            fwd = _to_float(_call_price_fn(price_fn, symbol, h_end))
            entry = _to_float(rec.get("entry_underlying_price"))

            snap = {
                "record_type": RECORD_TYPE_RESOLUTION,
                "prediction_id": rec.get("prediction_id"),
                "resolved_at": ref_now.isoformat(),
                "fwd_underlying_price": fwd,
            }
            if fwd is None:
                snap["resolution_status"] = STATUS_MISSING_PRICE
                _append_jsonl(snap, path)
                continue

            ret_pct = None
            if entry not in (None, 0.0):
                ret_pct = round((fwd - entry) / entry * 100.0, 6)
            realized = _realized_direction(ret_pct, flat_band_pct)
            predicted = _predicted_direction(rec.get("p_call"),
                                             rec.get("p_put"))
            correct = None
            if predicted in ("call", "put") and realized in ("up", "down"):
                correct = ((predicted == "call" and realized == "up") or
                           (predicted == "put" and realized == "down"))
            brier_call = None
            pc = _to_float(rec.get("p_call"))
            if pc is not None and ret_pct is not None:
                outcome_up = 1.0 if ret_pct > 0.0 else 0.0
                brier_call = round((pc - outcome_up) ** 2, 6)

            snap.update({
                "realized_return_pct": ret_pct,
                "realized_direction": realized,
                "predicted_direction": predicted,
                "correct": correct,
                "brier_call": brier_call,
                "resolution_status": STATUS_RESOLVED,
            })
            _append_jsonl(snap, path)
            resolved += 1
        return resolved
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} resolve ignored: {exc}")
        return 0


# --------------------------------------------------------------------------- #
# Self-test (offline; no network, temp ledger)
# --------------------------------------------------------------------------- #
def _self_test() -> bool:
    import os
    import tempfile

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "probs.jsonl")
    base = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)

    # Pure horizon mapping.
    assert horizon_end(base, "intraday") == base + timedelta(minutes=390)
    assert horizon_end(base, "swing", dte=2) == base + timedelta(days=2)
    assert horizon_end(base, "swing", dte=99) == base + timedelta(days=5)
    assert horizon_end(base, None) == base + timedelta(days=5)

    # Pure direction helpers.
    assert _predicted_direction(0.6, 0.2) == "call"
    assert _predicted_direction(0.2, 0.6) == "put"
    assert _predicted_direction(0.3, 0.3) == "none"
    assert _realized_direction(1.2) == "up"
    assert _realized_direction(-0.4) == "down"
    assert _realized_direction(0.0) == "flat"

    # Record a bullish intraday prediction and a no-trade one.
    pid_call = record_prediction({
        "symbol": "SPY", "mode": "intraday", "trend_horizon": "intraday",
        "probability": {"p_call": 0.7, "p_put": 0.1, "p_no_trade": 0.2},
        "expected_move_pct": 1.5, "prior": 0.5,
        "entry_underlying_price": 100.0, "as_of": base.isoformat(),
    }, jsonl_path=path, now=base)
    pid_nt = record_prediction({
        "symbol": "QQQ", "mode": "swing", "trend_horizon": "swing",
        "probability": {"p_call": 0.2, "p_put": 0.2, "p_no_trade": 0.6},
        "entry_underlying_price": 200.0, "as_of": base.isoformat(), "dte": 3,
    }, jsonl_path=path, now=base)
    assert pid_call and pid_nt
    loaded = load_predictions(path)
    assert len(loaded) == 2, "two frozen predictions expected"

    # Not-yet-due: resolving before horizon appends nothing.
    early = base + timedelta(minutes=10)
    assert resolve_predictions(lambda s, at: 105.0, now=early,
                               jsonl_path=path) == 0
    assert all(r.get("resolution_status") is None
               for r in load_predictions(path))

    # Due intraday: underlying rose -> up, bullish head correct, low Brier.
    after_intraday = base + timedelta(minutes=400)
    n = resolve_predictions(lambda s, at: 103.0, now=after_intraday,
                            jsonl_path=path)
    assert n == 1, "only the intraday prediction is due"
    folded = {r["prediction_id"]: r for r in load_predictions(path)}
    call_row = folded[pid_call]
    assert call_row["resolution_status"] == STATUS_RESOLVED
    assert call_row["realized_direction"] == "up"
    assert call_row["predicted_direction"] == "call"
    assert call_row["correct"] is True
    assert abs(call_row["realized_return_pct"] - 3.0) < 1e-6
    assert abs(call_row["brier_call"] - (0.7 - 1.0) ** 2) < 1e-6
    # The swing prediction is still pending at this time.
    assert folded[pid_nt].get("resolution_status") is None

    # Due swing later: underlying fell -> a no-trade head is not "correct".
    after_swing = base + timedelta(days=4)
    n2 = resolve_predictions(lambda s, at: 190.0, now=after_swing,
                             jsonl_path=path)
    assert n2 == 1
    nt_row = {r["prediction_id"]: r
              for r in load_predictions(path)}[pid_nt]
    assert nt_row["realized_direction"] == "down"
    assert nt_row["predicted_direction"] == "none"
    assert nt_row["correct"] is None, "no-trade head has no direction to grade"

    # Idempotent: a finalised prediction is never re-resolved.
    assert resolve_predictions(lambda s, at: 999.0, now=after_swing,
                               jsonl_path=path) == 0

    # Missing price when due -> a missing_price line, still not finalised.
    path2 = os.path.join(tmp, "probs2.jsonl")
    pid = record_prediction({
        "symbol": "IWM", "mode": "intraday",
        "probability": {"p_call": 0.5, "p_put": 0.3, "p_no_trade": 0.2},
        "entry_underlying_price": 50.0, "as_of": base.isoformat(),
    }, jsonl_path=path2, now=base)
    assert resolve_predictions(lambda s, at: None,
                               now=base + timedelta(minutes=400),
                               jsonl_path=path2) == 0
    miss = load_predictions(path2)[0]
    assert miss["resolution_status"] == STATUS_MISSING_PRICE

    # Fail-open: bad fields / missing file do not raise.
    assert record_prediction({}, jsonl_path=path) is None
    assert load_predictions(os.path.join(tmp, "nope.jsonl")) == []
    return True


if __name__ == "__main__":
    ok = _self_test()
    print("oracle_prob_recorder self-test:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
