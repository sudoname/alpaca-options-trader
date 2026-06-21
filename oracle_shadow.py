"""
Shadow-Oracle metadata helper.

Computes an advisory Oracle opinion (regime label, model probabilities, agent
votes and contributions) for an underlying at trade-entry time so the Oracle
Intelligence-Layer dashboard tiles can populate on the single-leg bot.

It is **shadow-only**: the returned dict is recorded on the trade record and
used purely for offline analytics. It NEVER influences trade direction, sizing,
gating, pricing or execution. Every path is fail-open — any error (missing
creds, network, import) yields ``{}`` so the caller's order flow is byte
identical to having no shadow at all.

Follows the project idiom: pure helper + injectable builder for offline tests,
``_self_test() -> int``, and ``if __name__ == "__main__": sys.exit(_self_test())``.
"""

import sys
from typing import Callable, Dict, Optional

import oracle_intelligence_reports as oir


# Fields the dashboard reports filter closed trades on. Kept here so callers and
# tests share one source of truth.
SHADOW_FIELDS = (
    "regime_label",
    "regime_confidence",
    "model_p_call",
    "model_p_put",
    "agent_votes",
    "agent_contributions",
)


def compute_oracle_shadow(
    underlying: str,
    *,
    ctx: Optional[dict] = None,
    ctx_builder: Optional[Callable[[str], dict]] = None,
) -> Dict:
    """Return a shadow-Oracle opinion for ``underlying`` or ``{}`` on any failure.

    Args:
        underlying: the underlying symbol (e.g. "SPY").
        ctx: pre-built agent evidence context; if falsy it is built via
            ``ctx_builder`` (default ``explain_context.build_explain_context``).
        ctx_builder: injectable builder ``(symbol) -> ctx dict``; used for
            offline tests so no network call is made.

    The result, when non-empty, contains the six ``SHADOW_FIELDS`` keys.
    """
    try:
        if not ctx:
            if ctx_builder is None:
                # Imported lazily so the module loads without optional deps.
                import explain_context
                ctx_builder = explain_context.build_explain_context
            ctx = ctx_builder(underlying)
        if not ctx:
            return {}

        explain = oir.compute_oracle_explain(underlying, ctx=ctx)
        regime = oir.compute_oracle_regime_report(regime_raw=ctx, symbol=underlying)

        prob = explain.get("probability", {}) or {}
        expl = explain.get("explanation", {}) or {}

        agent_votes = {}
        for v in explain.get("votes", []) or []:
            if not isinstance(v, dict):
                continue
            name = v.get("name")
            if not name:
                continue
            agent_votes[name] = {
                "bullish_score": v.get("bullish_score"),
                "bearish_score": v.get("bearish_score"),
                "confidence": v.get("confidence"),
            }

        return {
            "regime_label": regime.get("label"),
            "regime_confidence": regime.get("confidence"),
            "model_p_call": prob.get("p_call"),
            "model_p_put": prob.get("p_put"),
            "agent_votes": agent_votes,
            "agent_contributions": expl.get("agent_contributions", {}) or {},
        }
    except Exception:  # fail-open — never disturb the caller's order flow
        return {}


def _self_test() -> int:
    ok = True

    fixed_ctx = {
        "trend": "up",
        "momentum": 0.04,
        "realized_vol": 0.012,
        "regime": "trending",
        "volume_ratio": 1.2,
        "rel_strength": 0.01,
    }

    # --- builder is invoked with the symbol and produces a full opinion ---
    seen = {}

    def _builder(sym):
        seen["sym"] = sym
        return dict(fixed_ctx)

    res = compute_oracle_shadow("SPY", ctx_builder=_builder)
    if seen.get("sym") != "SPY":
        print("FAIL: ctx_builder not called with symbol", seen); ok = False
    for k in SHADOW_FIELDS:
        if k not in res:
            print(f"FAIL: missing key {k}", res); ok = False
    if not isinstance(res.get("agent_votes"), dict):
        print("FAIL: agent_votes not dict", res.get("agent_votes")); ok = False
    if not isinstance(res.get("agent_contributions"), dict):
        print("FAIL: agent_contributions not dict",
              res.get("agent_contributions")); ok = False
    if res.get("model_p_call") is not None and not isinstance(
            res.get("model_p_call"), (int, float)):
        print("FAIL: model_p_call type", res.get("model_p_call")); ok = False

    # --- passing ctx directly skips the builder ---
    res2 = compute_oracle_shadow("SPY", ctx=dict(fixed_ctx),
                                 ctx_builder=lambda s: (_ for _ in ()).throw(
                                     AssertionError("builder must not run")))
    if set(SHADOW_FIELDS) - set(res2):
        print("FAIL: ctx-direct missing keys", res2); ok = False

    # --- empty ctx from builder -> {} (fail-open, no opinion) ---
    if compute_oracle_shadow("SPY", ctx_builder=lambda s: {}) != {}:
        print("FAIL: empty ctx should yield {}"); ok = False

    # --- raising builder -> {} (fail-open, never raises) ---
    def _boom(sym):
        raise RuntimeError("boom")

    if compute_oracle_shadow("SPY", ctx_builder=_boom) != {}:
        print("FAIL: raising builder should yield {}"); ok = False

    print("oracle_shadow self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
