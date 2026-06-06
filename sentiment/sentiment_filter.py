"""
Sentiment-based trade risk filter.

Single, well-defined integration point between the sentiment module and the
trading strategies. Strategies call ``adjust_trade_risk_by_sentiment(candidate,
sentiment)`` and receive a decision describing whether the trade is allowed and
how its size should change — plus a human-readable reason for logging.

Policy (keyed off the primary score's classification):
    Extreme Fear  (0-25)  -> block aggressive longs; only allow high-confidence,
                             and cut size hard.
    Fear          (26-45) -> reduce size, require stronger confirmation.
    Neutral       (46-55) -> normal.
    Greed         (56-75) -> normal / slightly aggressive.
    Extreme Greed (76-100)-> trim size to avoid over-leveraging into euphoria.

The function NEVER raises and ALWAYS returns a decision dict. If sentiment is
disabled/unavailable it returns an "allowed, unchanged" decision so trading is
never blocked by a sentiment failure.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# High-confidence threshold used to decide whether a fearful market still
# permits a (reduced) long.
_HIGH_CONFIDENCE = 80.0


def _decision(allowed, original_size, adjusted_size, reason,
              classification=None, score=None, size_multiplier=1.0,
              confidence_floor=None):
    return {
        "allowed": allowed,
        "original_size": original_size,
        "adjusted_size": adjusted_size,
        "size_multiplier": round(size_multiplier, 3),
        "reason": reason,
        "classification": classification,
        "score": score,
        "confidence_floor": confidence_floor,
    }


def adjust_trade_risk_by_sentiment(trade_candidate: dict,
                                   sentiment: Optional[dict]) -> dict:
    """Adjust a trade candidate's risk based on market sentiment.

    Args:
        trade_candidate: dict with at least ``size`` (int/float, e.g. contracts);
            optional ``confidence`` (0-100) and ``direction`` ("CALL"/"PUT").
        sentiment: the payload from ``SentimentService.get_sentiment()`` (may be
            None or an error payload).

    Returns:
        Decision dict, e.g.
            {"allowed": True, "original_size": 100, "adjusted_size": 75,
             "size_multiplier": 0.75,
             "reason": "Fear sentiment detected; reducing position size by 25%",
             "classification": "Fear", "score": 38.0, "confidence_floor": None}
    """
    original_size = (trade_candidate or {}).get("size", 1)
    confidence = (trade_candidate or {}).get("confidence")
    direction = (trade_candidate or {}).get("direction")

    # --- No usable sentiment => pass through unchanged (fail-open). ---
    primary = (sentiment or {}).get("primary_score") if sentiment else None
    if not primary or primary.get("status") != "available" or primary.get("score") is None:
        return _decision(
            True, original_size, original_size,
            "Sentiment unavailable; no adjustment applied",
        )

    score = primary.get("score")
    classification = primary.get("classification") or "Unknown"

    # A "long" exposure here means a directional debit (CALL or PUT buy). The
    # fear/greed adjustments target over-aggressive directional bets.
    is_long = direction in (None, "CALL", "PUT")

    if classification == "Extreme Fear":
        # Only allow high-confidence directional trades; cut size hard.
        if is_long and confidence is not None and confidence < _HIGH_CONFIDENCE:
            return _decision(
                False, original_size, 0,
                (f"Extreme Fear (score {score}); blocking aggressive long — "
                 f"confidence {confidence:.0f}% < {_HIGH_CONFIDENCE:.0f}% floor"),
                classification, score, size_multiplier=0.0,
                confidence_floor=_HIGH_CONFIDENCE,
            )
        mult = 0.5
        return _decision(
            True, original_size, _scale(original_size, mult),
            (f"Extreme Fear (score {score}); high-confidence only, "
             f"reducing position size by 50%"),
            classification, score, size_multiplier=mult,
            confidence_floor=_HIGH_CONFIDENCE,
        )

    if classification == "Fear":
        mult = 0.75
        return _decision(
            True, original_size, _scale(original_size, mult),
            (f"Fear sentiment detected (score {score}); "
             f"reducing position size by 25%"),
            classification, score, size_multiplier=mult,
        )

    if classification == "Neutral":
        return _decision(
            True, original_size, original_size,
            f"Neutral sentiment (score {score}); normal sizing",
            classification, score, size_multiplier=1.0,
        )

    if classification == "Greed":
        return _decision(
            True, original_size, original_size,
            f"Greed sentiment (score {score}); normal sizing",
            classification, score, size_multiplier=1.0,
        )

    if classification == "Extreme Greed":
        mult = 0.75
        return _decision(
            True, original_size, _scale(original_size, mult),
            (f"Extreme Greed (score {score}); trimming size by 25% "
             f"to avoid over-leveraging into euphoria"),
            classification, score, size_multiplier=mult,
        )

    # Unknown classification: pass through.
    return _decision(
        True, original_size, original_size,
        f"Unrecognized sentiment classification '{classification}'; no change",
        classification, score,
    )


def _scale(size, multiplier):
    """Scale a position size by a multiplier, preserving int-ness and a 1 floor.

    Position size is in contracts, so we keep it an integer and never round a
    still-allowed trade down to zero.
    """
    try:
        if isinstance(size, int):
            scaled = int(round(size * multiplier))
            return max(1, scaled) if size >= 1 and multiplier > 0 else scaled
        return round(size * multiplier, 4)
    except (TypeError, ValueError):
        return size


def summarize_for_log(sentiment: Optional[dict]) -> str:
    """One-line human summary of a sentiment payload for logging."""
    if not sentiment:
        return "sentiment: none"
    primary = sentiment.get("primary_score") or {}
    src = sentiment.get("primary_source")
    score = primary.get("score")
    label = primary.get("classification", "Unknown")
    status = primary.get("status")
    custom = sentiment.get("custom_score") or {}
    unavailable = custom.get("unavailable_components") or []
    cache_note = " (cached)" if sentiment.get("from_cache") else ""
    if sentiment.get("from_stale_cache"):
        cache_note = " (STALE cache)"
    if status != "available" or score is None:
        return f"sentiment: unavailable ({status}){cache_note}"
    msg = f"sentiment: {score} {label} via {src}{cache_note}"
    if unavailable:
        msg += f" | unavailable components: {', '.join(unavailable)}"
    return msg
