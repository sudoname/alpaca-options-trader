"""
Oracle 3.0 — Upgrade 5: Bull / Bear / No-Trade adversarial thesis (ANALYTICS).

Built strictly ON TOP of the existing voting mass. The quant head
(:func:`oracle_voting.tally_votes` / :func:`oracle_voting.bayesian_probability`)
stays AUTHORITATIVE — this module never computes direction and never invents a
probability. It performs two pure, fail-open steps:

``build_theses(ctx, tally, probability, evidence) -> {bull, bear, no_trade}``
    Three structured theses over the SAME evidence set. Each carries
    ``support[]``, ``counter_evidence[]``, ``catalysts[]``, ``regime_alignment``,
    ``technical_state``, ``vol_state``, ``attention_state``,
    ``fundamental_state``, ``expected_move``, ``invalidation`` and a
    ``confidence`` seeded from ``p_bull``/``p_bear``/``p_neutral``.

``adversarial_review(theses, probability, config) -> ReviewResult``
    A bounded skeptic. It may detect incoherence (direction vs tally mismatch,
    ambiguous direction, stale catalysts, regime conflict, thin support) and
    respond ONLY by raising ``p_no_trade`` — within a hard cap
    (``THESIS_MAX_NO_TRADE_BOOST``, default 0.10) — while proportionally shaving
    the directional mass. It NEVER flips the leading direction, NEVER invents
    numbers, NEVER touches execution or hard risk. The result renormalizes to
    sum 1.0.

An optional ``llm_provider`` may be supplied to either function; it may ONLY
attach natural-language ``support``/``counter_evidence``/``flags`` text. Numeric
authority always stays with the quant. Provider output is expected to be
cached/mocked for deterministic replay; any provider error is swallowed.

Design rules (mirror oracle/thesis.py, oracle_voting.py):
  * Pure. No network, no creds, no file writes, no env-mutation.
  * Deterministic given inputs. Prefer None/empty over a guess.
  * Never raise on malformed input (fail-open); return the un-adjusted
    probability rather than crashing the gate.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

LOG_TAG = "[THESIS_DEBATE]"

# Direction labels (mirror oracle/thesis.py).
DIR_CALL = "call"
DIR_PUT = "put"
NO_TRADE = "no_trade"

# Hard cap on how much this skeptic may raise p_no_trade. The quant remains
# authoritative; the review can only ADD doubt, never manufacture conviction.
THESIS_MAX_NO_TRADE_BOOST_DEFAULT = 0.10

# Per-flag penalties (summed then capped at the max boost).
_PENALTY = {
    "direction_tally_mismatch": 0.05,
    "ambiguous_direction": 0.05,
    "stale_catalyst": 0.04,
    "regime_conflict": 0.04,
    "thin_support": 0.03,
}

# A directional gap this small is treated as "no clear side".
_AMBIGUOUS_GAP = 0.05
# A tally gap this small is treated as "no directional opinion" (mismatch check
# only fires when BOTH the quant head and the tally have a real opinion).
_TALLY_EPS = 0.05
# Below this p_no_trade the setup is considered "actionable".
_ACTIONABLE_NO_TRADE = 0.5

_STANCE_BULL = "bull"
_STANCE_BEAR = "bear"
_STANCE_NEUTRAL = "neutral"


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #
def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return default
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _norm_stance(value) -> str:
    """Map a free-form stance to bull/bear/neutral."""
    if not isinstance(value, str):
        return _STANCE_NEUTRAL
    v = value.strip().lower()
    if v in ("bull", "bullish", "call", "up", "long", "+"):
        return _STANCE_BULL
    if v in ("bear", "bearish", "put", "down", "short", "-"):
        return _STANCE_BEAR
    return _STANCE_NEUTRAL


def _norm_evidence(evidence) -> List[dict]:
    """Normalize evidence into a list of {name, stance, text, stale, catalyst}.

    Accepts a list of dicts (or strings). Anything else -> []. Never raises.
    """
    out: List[dict] = []
    if not evidence:
        return out
    try:
        items = list(evidence)
    except TypeError:
        return out
    for it in items:
        if isinstance(it, dict):
            name = str(it.get("name", it.get("id", "?")))
            stance = _norm_stance(it.get("stance"))
            text = str(it.get("text", it.get("detail", name)))
            stale = bool(it.get("stale", False))
            catalyst = bool(it.get("catalyst", it.get("kind") == "catalyst"))
        elif isinstance(it, str):
            name = it
            stance = _STANCE_NEUTRAL
            text = it
            stale = False
            catalyst = False
        else:
            continue
        out.append({"name": name, "stance": stance, "text": text,
                    "stale": stale, "catalyst": catalyst})
    return out


def _regime_alignment(direction: Optional[str], regime) -> str:
    """'aligned' / 'counter' / 'neutral' for a direction given a regime label."""
    if direction not in (DIR_CALL, DIR_PUT) or not isinstance(regime, str):
        return "neutral"
    r = regime.strip().lower()
    bull_like = any(k in r for k in ("bull", "up", "trend_up", "risk_on",
                                     "uptrend", "expansion"))
    bear_like = any(k in r for k in ("bear", "down", "trend_down", "risk_off",
                                     "downtrend", "contraction"))
    if direction == DIR_CALL:
        if bull_like:
            return "aligned"
        if bear_like:
            return "counter"
    else:  # DIR_PUT
        if bear_like:
            return "aligned"
        if bull_like:
            return "counter"
    return "neutral"


def _leading_direction(probability: dict) -> Optional[str]:
    """Which side the quant head favours (call/put), or None if tied/absent."""
    pc = _to_float((probability or {}).get("p_call"), 0.0) or 0.0
    pp = _to_float((probability or {}).get("p_put"), 0.0) or 0.0
    if abs(pc - pp) < 1e-9:
        return None
    return DIR_CALL if pc > pp else DIR_PUT


def _expected_move(ctx: dict, expected_move) -> Optional[float]:
    if isinstance(expected_move, dict):
        em = _to_float(expected_move.get("sigma1_pct"))
        if em is not None:
            return round(em, 6)
    em = _to_float((ctx or {}).get("expected_move_pct"))
    return round(em, 6) if em is not None else None


# --------------------------------------------------------------------------- #
# build_theses
# --------------------------------------------------------------------------- #
def _one_thesis(direction: Optional[str], confidence: float, ctx: dict,
                items: List[dict], em_pct: Optional[float]) -> dict:
    """Assemble a single thesis over the shared evidence set."""
    if direction == DIR_CALL:
        support_stance, counter_stance = _STANCE_BULL, _STANCE_BEAR
    elif direction == DIR_PUT:
        support_stance, counter_stance = _STANCE_BEAR, _STANCE_BULL
    else:  # no-trade: neutrality is "support", any directional item is counter
        support_stance, counter_stance = _STANCE_NEUTRAL, None

    support: List[str] = []
    counter: List[str] = []
    catalysts: List[str] = []
    for it in items:
        if it["catalyst"]:
            catalysts.append(it["text"])
        st = it["stance"]
        if direction in (DIR_CALL, DIR_PUT):
            if st == support_stance:
                support.append(it["text"])
            elif st == counter_stance:
                counter.append(it["text"])
        else:  # no-trade
            if st == _STANCE_NEUTRAL:
                support.append(it["text"])
            else:
                counter.append(it["text"])

    return {
        "direction": direction,
        "confidence": round(_clamp01(confidence), 6),
        "support": support,
        "counter_evidence": counter,
        "catalysts": catalysts,
        "regime_alignment": _regime_alignment(direction, ctx.get("regime")),
        "technical_state": ctx.get("technical_state"),
        "vol_state": ctx.get("vol_state"),
        "attention_state": ctx.get("attention_state"),
        "fundamental_state": ctx.get("fundamental_state"),
        "expected_move": em_pct,
        "invalidation": em_pct,   # a full adverse 1-sigma voids the thesis
    }


def build_theses(ctx: Optional[dict], tally: Optional[dict],
                 probability: Optional[dict], evidence=None,
                 expected_move=None,
                 llm_provider: Optional[Callable] = None) -> dict:
    """Return ``{"bull": .., "bear": .., "no_trade": ..}`` over one evidence set.

    Confidences are seeded verbatim from ``tally`` (``p_bull``/``p_bear``/
    ``p_neutral``). Direction is NEVER computed here. Never raises.
    """
    try:
        if not isinstance(ctx, dict):
            ctx = {}
        tally = tally if isinstance(tally, dict) else {}
        items = _norm_evidence(evidence)
        em_pct = _expected_move(ctx, expected_move)

        p_bull = _clamp01(_to_float(tally.get("p_bull"), 0.0) or 0.0)
        p_bear = _clamp01(_to_float(tally.get("p_bear"), 0.0) or 0.0)
        p_neut = _clamp01(_to_float(tally.get("p_neutral"), 0.0) or 0.0)

        theses = {
            "bull": _one_thesis(DIR_CALL, p_bull, ctx, items, em_pct),
            "bear": _one_thesis(DIR_PUT, p_bear, ctx, items, em_pct),
            NO_TRADE: _one_thesis(None, p_neut, ctx, items, em_pct),
        }

        # Optional LLM: text-only enrichment. Numbers are untouched; any error
        # is swallowed (deterministic replay expects a cached/mocked provider).
        if llm_provider is not None:
            try:
                enrich = llm_provider({"ctx": ctx, "tally": tally,
                                       "probability": probability,
                                       "theses": theses})
                if isinstance(enrich, dict):
                    for key in ("bull", "bear", NO_TRADE):
                        add = enrich.get(key)
                        if isinstance(add, dict):
                            for fld in ("support", "counter_evidence"):
                                extra = add.get(fld)
                                if isinstance(extra, list):
                                    theses[key][fld] = list(theses[key][fld]) + \
                                        [str(x) for x in extra]
            except Exception:  # pragma: no cover - fail-open
                pass

        return theses
    except Exception:  # pragma: no cover - fail-open
        empty = {"direction": None, "confidence": 0.0, "support": [],
                 "counter_evidence": [], "catalysts": [],
                 "regime_alignment": "neutral", "technical_state": None,
                 "vol_state": None, "attention_state": None,
                 "fundamental_state": None, "expected_move": None,
                 "invalidation": None}
        return {"bull": dict(empty, direction=DIR_CALL),
                "bear": dict(empty, direction=DIR_PUT),
                NO_TRADE: dict(empty)}


# --------------------------------------------------------------------------- #
# adversarial_review
# --------------------------------------------------------------------------- #
@dataclass
class ReviewResult:
    """Outcome of the bounded skeptic. Direction is preserved; only doubt added."""

    adjusted_probability: Dict[str, float]
    flags: List[str] = field(default_factory=list)
    invalidation: List[float] = field(default_factory=list)
    no_trade_boost: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "adjusted_probability": self.adjusted_probability,
            "flags": self.flags,
            "invalidation": self.invalidation,
            "no_trade_boost": self.no_trade_boost,
            "notes": self.notes,
        }


def _max_boost(config, max_no_trade_boost: Optional[float]) -> float:
    if max_no_trade_boost is not None:
        b = _to_float(max_no_trade_boost, THESIS_MAX_NO_TRADE_BOOST_DEFAULT)
        return _clamp01(b if b is not None else THESIS_MAX_NO_TRADE_BOOST_DEFAULT)
    try:
        if config is not None and hasattr(config, "get_float"):
            b = config.get_float("THESIS_MAX_NO_TRADE_BOOST",
                                 THESIS_MAX_NO_TRADE_BOOST_DEFAULT)
            return _clamp01(_to_float(b, THESIS_MAX_NO_TRADE_BOOST_DEFAULT)
                            or THESIS_MAX_NO_TRADE_BOOST_DEFAULT)
    except Exception:  # pragma: no cover - fail-open
        pass
    return THESIS_MAX_NO_TRADE_BOOST_DEFAULT


def _detect_flags(theses: dict, tally: dict, probability: dict,
                  lead: Optional[str]) -> List[str]:
    """Deterministic incoherence checks. Returns ordered, de-duplicated flags."""
    flags: List[str] = []

    pc = _to_float(probability.get("p_call"), 0.0) or 0.0
    pp = _to_float(probability.get("p_put"), 0.0) or 0.0
    pnt = _to_float(probability.get("p_no_trade"), 0.0) or 0.0
    p_bull = _to_float((tally or {}).get("p_bull"), 0.0) or 0.0
    p_bear = _to_float((tally or {}).get("p_bear"), 0.0) or 0.0

    # 1. direction vs tally mismatch (both must actually have an opinion).
    quant_gap = pc - pp
    tally_gap = p_bull - p_bear
    if (abs(quant_gap) >= _TALLY_EPS and abs(tally_gap) >= _TALLY_EPS and
            (quant_gap > 0) != (tally_gap > 0)):
        flags.append("direction_tally_mismatch")

    # 2. ambiguous direction while still actionable.
    if abs(quant_gap) < _AMBIGUOUS_GAP and pnt < _ACTIONABLE_NO_TRADE:
        flags.append("ambiguous_direction")

    # Leading-side thesis (the one the quant would trade).
    lead_thesis = None
    if lead == DIR_CALL:
        lead_thesis = theses.get("bull")
    elif lead == DIR_PUT:
        lead_thesis = theses.get("bear")

    if lead_thesis is not None and pnt < _ACTIONABLE_NO_TRADE:
        # 3. a stale catalyst is propping up the leading side.
        cats = lead_thesis.get("catalysts") or []
        support = lead_thesis.get("support") or []
        if cats and any(c in support for c in cats) is False and cats:
            # catalysts present but not among support -> not directly relevant;
            # only flag when a catalyst is BOTH support and marked stale below.
            pass
        # Recompute staleness from the raw flag threaded via support markers:
        if lead_thesis.get("_stale_support"):
            flags.append("stale_catalyst")

        # 4. regime conflict on the leading side.
        if lead_thesis.get("regime_alignment") == "counter":
            flags.append("regime_conflict")

        # 5. thin support: actionable but nothing supports the leading side.
        if not support:
            flags.append("thin_support")

    # De-dup preserving order.
    seen = set()
    ordered: List[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def adversarial_review(theses: Optional[dict], probability: Optional[dict],
                       config=None, tally: Optional[dict] = None, *,
                       max_no_trade_boost: Optional[float] = None,
                       evidence=None,
                       llm_provider: Optional[Callable] = None) -> ReviewResult:
    """Bounded skeptic: may only RAISE ``p_no_trade`` (capped), never flip.

    Returns a :class:`ReviewResult` whose ``adjusted_probability`` sums to 1.0.
    On any error it falls open to the input probability, un-adjusted.
    """
    base = {"p_call": 0.0, "p_put": 0.0, "p_no_trade": 1.0}
    try:
        if isinstance(probability, dict):
            base = {
                "p_call": _clamp01(_to_float(probability.get("p_call"), 0.0) or 0.0),
                "p_put": _clamp01(_to_float(probability.get("p_put"), 0.0) or 0.0),
                "p_no_trade": _clamp01(
                    _to_float(probability.get("p_no_trade"), 1.0) or 0.0),
            }
        theses = theses if isinstance(theses, dict) else {}

        # Thread staleness of the leading side's supporting catalysts. We derive
        # it here from the shared evidence (if supplied) so build_theses stays a
        # pure snapshot; absent evidence, staleness simply never fires.
        lead = _leading_direction(base)
        items = _norm_evidence(evidence)
        if lead in (DIR_CALL, DIR_PUT):
            want = _STANCE_BULL if lead == DIR_CALL else _STANCE_BEAR
            stale_support = any(
                it["stale"] and it["stance"] == want for it in items)
            key = "bull" if lead == DIR_CALL else "bear"
            if isinstance(theses.get(key), dict):
                theses[key]["_stale_support"] = stale_support

        flags = _detect_flags(theses, tally or {}, base, lead)

        # Optional LLM may only ADD advisory flag text (never numbers).
        if llm_provider is not None:
            try:
                extra = llm_provider({"theses": theses, "probability": base,
                                      "flags": flags})
                if isinstance(extra, dict):
                    add = extra.get("flags")
                    if isinstance(add, list):
                        for f in add:
                            fs = str(f)
                            if fs not in flags:
                                flags.append(fs)
            except Exception:  # pragma: no cover - fail-open
                pass

        cap = _max_boost(config, max_no_trade_boost)
        raw_boost = sum(_PENALTY.get(f, 0.0) for f in flags)
        boost = min(cap, raw_boost)

        # Clean up the private marker so it never leaks into a record.
        for key in ("bull", "bear"):
            if isinstance(theses.get(key), dict):
                theses[key].pop("_stale_support", None)

        # Apply: raise no-trade by `boost`, shave the directional mass in its
        # existing call:put ratio so the LEADING side never flips.
        directional = base["p_call"] + base["p_put"]
        new_no_trade = _clamp01(base["p_no_trade"] + boost)
        remaining = _clamp01(1.0 - new_no_trade)
        if directional > 0.0:
            call_share = base["p_call"] / directional
            new_call = remaining * call_share
            new_put = remaining * (1.0 - call_share)
        else:
            new_call = new_put = remaining / 2.0

        adj = {"p_call": new_call, "p_put": new_put, "p_no_trade": new_no_trade}
        s = adj["p_call"] + adj["p_put"] + adj["p_no_trade"]
        if s > 0.0:
            adj = {k: round(v / s, 6) for k, v in adj.items()}
        else:
            adj = dict(base)

        invalidation = []
        for key in ("bull", "bear", NO_TRADE):
            t = theses.get(key)
            if isinstance(t, dict):
                inv = _to_float(t.get("invalidation"))
                if inv is not None:
                    invalidation.append(round(inv, 6))

        notes = (f"boost={boost:.4f} (cap={cap:.4f}) flags={flags} "
                 f"lead={lead or 'none'}")
        return ReviewResult(adjusted_probability=adj, flags=flags,
                            invalidation=invalidation, no_trade_boost=round(boost, 6),
                            notes=notes)
    except Exception:  # pragma: no cover - fail-open
        return ReviewResult(adjusted_probability=base, flags=[],
                            invalidation=[], no_trade_boost=0.0,
                            notes="fail-open: review skipped")


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    def _sums_to_one(d):
        return abs(sum(d[k] for k in ("p_call", "p_put", "p_no_trade")) - 1.0) < 2e-6

    ctx = {"regime": "uptrend", "technical_state": "above_20ma",
           "vol_state": "normal", "attention_state": "elevated",
           "fundamental_state": "ok", "expected_move_pct": 4.0}
    evidence = [
        {"name": "trend", "stance": "bull", "text": "price > 20/50 MA"},
        {"name": "news", "stance": "bull", "text": "upgrade", "catalyst": True},
        {"name": "breadth", "stance": "bear", "text": "weak breadth"},
        {"name": "liquidity", "stance": "neutral", "text": "tight spread"},
    ]

    # --- build_theses: three theses over one evidence set -------------------
    tally = {"p_bull": 0.6, "p_bear": 0.25, "p_neutral": 0.15}
    theses = build_theses(ctx, tally, None, evidence, expected_move=None)
    for key in ("bull", "bear", "no_trade"):
        if key not in theses:
            print("FAIL: missing thesis", key); ok = False
    if abs(theses["bull"]["confidence"] - 0.6) > 1e-9:
        print("FAIL: bull confidence seeded from p_bull", theses["bull"]); ok = False
    if abs(theses["bear"]["confidence"] - 0.25) > 1e-9:
        print("FAIL: bear confidence seeded from p_bear"); ok = False
    if abs(theses["no_trade"]["confidence"] - 0.15) > 1e-9:
        print("FAIL: no_trade confidence seeded from p_neutral"); ok = False
    # Bull support should include the bullish items; bear items are counter.
    if "price > 20/50 MA" not in theses["bull"]["support"]:
        print("FAIL: bull support", theses["bull"]); ok = False
    if "weak breadth" not in theses["bull"]["counter_evidence"]:
        print("FAIL: bull counter", theses["bull"]); ok = False
    # Same evidence set -> bear's support/counter are the mirror.
    if "weak breadth" not in theses["bear"]["support"]:
        print("FAIL: bear support mirror", theses["bear"]); ok = False
    # Catalyst surfaced.
    if "upgrade" not in theses["bull"]["catalysts"]:
        print("FAIL: catalyst surfaced", theses["bull"]); ok = False
    # Regime alignment: uptrend aligns bull, counters bear.
    if theses["bull"]["regime_alignment"] != "aligned":
        print("FAIL: bull regime aligned", theses["bull"]); ok = False
    if theses["bear"]["regime_alignment"] != "counter":
        print("FAIL: bear regime counter", theses["bear"]); ok = False

    # Expected move flows through from an expected_move dict.
    th_em = build_theses(ctx, tally, None, evidence,
                         expected_move={"sigma1_pct": 7.5})
    if abs(th_em["bull"]["expected_move"] - 7.5) > 1e-9:
        print("FAIL: expected move flow", th_em["bull"]); ok = False
    if abs(th_em["bull"]["invalidation"] - 7.5) > 1e-9:
        print("FAIL: invalidation = 1 sigma", th_em["bull"]); ok = False

    # --- adversarial_review: clean coherent bull -> no flags ---------------
    prob = {"p_call": 0.6, "p_put": 0.2, "p_no_trade": 0.2}
    r_clean = adversarial_review(theses, prob, tally=tally, evidence=evidence)
    if r_clean.flags:
        print("FAIL: clean case should have no flags", r_clean.flags); ok = False
    if not _sums_to_one(r_clean.adjusted_probability):
        print("FAIL: clean adj not normalized", r_clean.adjusted_probability); ok = False
    # Direction preserved (call still leads).
    if not (r_clean.adjusted_probability["p_call"] >
            r_clean.adjusted_probability["p_put"]):
        print("FAIL: clean lead flipped", r_clean.adjusted_probability); ok = False

    # --- mismatch: quant says call, tally says bear -> flag + more no_trade -
    tally_bear = {"p_bull": 0.2, "p_bear": 0.7, "p_neutral": 0.1}
    theses_mm = build_theses(ctx, tally_bear, None, evidence)
    r_mm = adversarial_review(theses_mm, prob, tally=tally_bear, evidence=evidence)
    if "direction_tally_mismatch" not in r_mm.flags:
        print("FAIL: mismatch not flagged", r_mm.flags); ok = False
    if not (r_mm.adjusted_probability["p_no_trade"] > prob["p_no_trade"]):
        print("FAIL: no_trade should rise on mismatch", r_mm.adjusted_probability); ok = False
    # Never flips: call still >= put after the (doubt-only) adjustment.
    if not (r_mm.adjusted_probability["p_call"] >
            r_mm.adjusted_probability["p_put"]):
        print("FAIL: review must not flip direction", r_mm.adjusted_probability); ok = False
    if not _sums_to_one(r_mm.adjusted_probability):
        print("FAIL: mismatch adj not normalized", r_mm.adjusted_probability); ok = False

    # --- boost is capped: pile on flags, boost never exceeds the cap -------
    ctx_counter = dict(ctx, regime="downtrend")   # regime_conflict for a call
    ev_stale = [
        {"name": "news", "stance": "bull", "text": "old upgrade",
         "catalyst": True, "stale": True},
    ]
    prob_ambig = {"p_call": 0.46, "p_put": 0.44, "p_no_trade": 0.10}
    theses_pile = build_theses(ctx_counter, {"p_bull": 0.2, "p_bear": 0.7,
                                             "p_neutral": 0.1}, None, ev_stale)
    r_cap = adversarial_review(theses_pile, prob_ambig,
                               tally={"p_bull": 0.2, "p_bear": 0.7,
                                      "p_neutral": 0.1},
                               evidence=ev_stale, max_no_trade_boost=0.10)
    if r_cap.no_trade_boost > 0.10 + 1e-9:
        print("FAIL: boost exceeded cap", r_cap.no_trade_boost); ok = False
    if len(r_cap.flags) < 2:
        print("FAIL: expected multiple flags", r_cap.flags); ok = False
    if not _sums_to_one(r_cap.adjusted_probability):
        print("FAIL: capped adj not normalized", r_cap.adjusted_probability); ok = False
    # Raw penalties exceed the cap, so the applied boost equals the cap exactly.
    if abs(r_cap.no_trade_boost - 0.10) > 1e-9:
        print("FAIL: boost should saturate at cap", r_cap.no_trade_boost); ok = False

    # --- stale catalyst on the leading side is detected --------------------
    prob_call = {"p_call": 0.6, "p_put": 0.2, "p_no_trade": 0.2}
    theses_stale = build_theses(ctx, {"p_bull": 0.6, "p_bear": 0.2,
                                      "p_neutral": 0.2}, None, ev_stale)
    r_stale = adversarial_review(theses_stale, prob_call,
                                 tally={"p_bull": 0.6, "p_bear": 0.2,
                                        "p_neutral": 0.2}, evidence=ev_stale)
    if "stale_catalyst" not in r_stale.flags:
        print("FAIL: stale catalyst not flagged", r_stale.flags); ok = False
    # The private staleness marker must not leak into the thesis record.
    if "_stale_support" in theses_stale.get("bull", {}):
        print("FAIL: private marker leaked", theses_stale["bull"]); ok = False

    # --- determinism -------------------------------------------------------
    if (adversarial_review(theses_mm, prob, tally=tally_bear,
                           evidence=evidence).to_dict() !=
            adversarial_review(theses_mm, prob, tally=tally_bear,
                               evidence=evidence).to_dict()):
        print("FAIL: adversarial_review non-deterministic"); ok = False

    # --- fail-open on junk -------------------------------------------------
    for junk in (None, 42, "x", [], {"weird": object()}):
        try:
            build_theses(junk, junk, junk, junk)          # type: ignore[arg-type]
            rr = adversarial_review(junk, junk, junk)      # type: ignore[arg-type]
            if not _sums_to_one(rr.adjusted_probability):
                print("FAIL: junk review not normalized", junk, rr.adjusted_probability)
                ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    # --- LLM provider may only add TEXT, never move numbers ----------------
    def _fake_llm(_payload):
        return {"bull": {"support": ["llm: momentum intact"]},
                "flags": ["llm_advisory_note"]}

    th_llm = build_theses(ctx, tally, None, evidence, llm_provider=_fake_llm)
    if "llm: momentum intact" not in th_llm["bull"]["support"]:
        print("FAIL: llm text not appended", th_llm["bull"]); ok = False
    r_llm = adversarial_review(th_llm, prob, tally=tally, evidence=evidence,
                               llm_provider=_fake_llm)
    if "llm_advisory_note" not in r_llm.flags:
        print("FAIL: llm advisory flag missing", r_llm.flags); ok = False
    # An advisory (non-scoring) flag carries 0 penalty -> numbers unchanged vs clean.
    if r_llm.no_trade_boost != r_clean.no_trade_boost:
        print("FAIL: llm advisory must not change boost", r_llm.no_trade_boost); ok = False

    print("oracle.thesis_debate self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
