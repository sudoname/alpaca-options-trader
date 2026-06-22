"""
oracle_weights_snapshot — scheduled trigger that feeds the Adaptive Weights tile.

The adaptive-weights machinery (``oracle_weights``) is fully built but nothing
on the single-leg box ever *calls* it, so the weights store has zero history and
the dashboard tile stays INSUFFICIENT. This entrypoint closes that gap: once per
trading day it loads the closed-trade records (which now carry ``agent_votes``
via the shadow-Oracle stamp) and runs ``oracle_weights.update_weights`` — which
recomputes the per-agent weights and appends a history snapshot.

Contract (mirrors the repo's analytics idioms):
  * SHADOW + read-only over trade data. It learns credit-assignment weights from
    *already-closed* outcomes; it never opens, sizes, prices, gates, or closes a
    trade, and the live trading path never imports it.
  * FAIL-OPEN. A failed load/compute prints and returns an ERROR verdict; it
    never raises (so a systemd timer can never crash-loop on bad data).
  * update_weights only persists when the compute verdict is OK (>= min_samples
    stamped closed records), so runs before enough data simply no-op the store.

Run:
  python oracle_weights_snapshot.py            # take one snapshot
  python oracle_weights_snapshot.py --selftest # offline gate (no creds/network)
"""

import sys

import ev_attribution
import oracle_weights


def run_snapshot(*, config=None, load_records=None, update=None) -> dict:
    """Load closed records and append a weight snapshot. Never raises."""
    load_records = load_records or ev_attribution.load_closed_records
    update = update or oracle_weights.update_weights
    try:
        records = load_records()
    except Exception as e:  # pragma: no cover - fail-open
        print(f"[WEIGHTS SNAPSHOT] load failed: {e}")
        return {"verdict": "ERROR", "sample_size": 0}
    try:
        result = update(records, config) if config is not None else update(records)
    except Exception as e:  # pragma: no cover - fail-open
        print(f"[WEIGHTS SNAPSHOT] update failed: {e}")
        return {"verdict": "ERROR", "sample_size": 0}
    verdict = result.get("verdict")
    n = result.get("sample_size")
    if verdict == "OK":
        print(f"[WEIGHTS SNAPSHOT] saved snapshot from {n} stamped closed records")
    else:
        print(f"[WEIGHTS SNAPSHOT] no snapshot (verdict={verdict}, sample_size={n})")
    return result


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network, no disk side effects outside a temp file)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import os
    import tempfile
    import uuid
    from oracle_weights import (OracleWeightsConfig, _synthetic_records,
                                weight_history)

    ok = True
    tmp = os.path.join(tempfile.gettempdir(), f"ows_selftest_{uuid.uuid4().hex}.json")
    try:
        cfg = OracleWeightsConfig(weights_file=tmp)

        # 1) Populated records -> OK verdict and exactly one appended snapshot.
        res = run_snapshot(config=cfg, load_records=lambda: _synthetic_records())
        if res.get("verdict") != "OK":
            print("FAIL: populated records should snapshot OK", res.get("verdict")); ok = False
        if len(weight_history(cfg)) != 1:
            print("FAIL: expected exactly one snapshot"); ok = False

        # 2) A second run appends a second snapshot (drift becomes computable).
        run_snapshot(config=cfg, load_records=lambda: _synthetic_records(40))
        if len(weight_history(cfg)) != 2:
            print("FAIL: second run should append a snapshot"); ok = False

        # 3) Empty records -> INSUFFICIENT and nothing appended.
        res3 = run_snapshot(config=cfg, load_records=lambda: [])
        if res3.get("verdict") != "INSUFFICIENT_DATA":
            print("FAIL: empty records should be insufficient", res3.get("verdict")); ok = False
        if len(weight_history(cfg)) != 2:
            print("FAIL: insufficient run must not append"); ok = False

        # 4) A raising loader fails open to ERROR — never propagates.
        def _boom():
            raise RuntimeError("loader down")
        res4 = run_snapshot(config=cfg, load_records=_boom)
        if res4.get("verdict") != "ERROR":
            print("FAIL: raising loader should fail open to ERROR"); ok = False
        if len(weight_history(cfg)) != 2:
            print("FAIL: a failed load must not append"); ok = False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    print("oracle_weights_snapshot self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_self_test())
    _result = run_snapshot()
    sys.exit(0 if _result.get("verdict") != "ERROR" else 1)
