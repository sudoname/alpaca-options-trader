"""
CNN Fear & Greed Index — unofficial scraper.

CNN publishes its Fear & Greed Index via an undocumented JSON endpoint used by
their data-viz widget:

    https://production.dataviz.cnn.io/index/fearandgreed/graphdata

This is an UNOFFICIAL, undocumented endpoint. It can change shape, rate-limit,
require a browser-like User-Agent, or disappear without notice. Every access is
therefore defensive: timeouts, browser headers, broad exception handling, and a
normalized return shape that always tells the caller whether data is usable.

This module NEVER raises to the caller. On any failure it returns a dict with
status="error" so the sentiment service can fall back to the custom score.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from .sentiment_config import SentimentConfig, classify_score

logger = logging.getLogger(__name__)

# Browser-like headers — the endpoint rejects requests without a UA / referer.
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
    "Origin": "https://www.cnn.com",
}


def _error(message: str, raw: Optional[dict] = None) -> dict:
    return {
        "source": "cnn_unofficial",
        "status": "error",
        "score": None,
        "classification": "Unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": message,
        "raw": raw,
    }


def _parse_payload(payload: dict) -> dict:
    """Extract the headline score from CNN's JSON payload.

    Expected shape (subject to change):
        {"fear_and_greed": {"score": 61.2, "rating": "greed", "timestamp": ...},
         "fear_and_greed_historical": {...}, ...}

    Falls back to the most recent historical point if the headline block is
    missing. Returns a normalized dict.
    """
    headline = payload.get("fear_and_greed")
    score = None
    classification = None
    src_timestamp = None

    if isinstance(headline, dict):
        score = headline.get("score")
        classification = headline.get("rating")
        src_timestamp = headline.get("timestamp")

    # Fallback: last historical data point.
    if score is None:
        hist = payload.get("fear_and_greed_historical")
        if isinstance(hist, dict):
            data = hist.get("data")
            if isinstance(data, list) and data:
                last = data[-1]
                if isinstance(last, dict):
                    score = last.get("y")
                    classification = last.get("rating")

    if score is None:
        return _error("CNN payload did not contain a usable score", raw=payload)

    try:
        score_val = round(float(score), 2)
    except (TypeError, ValueError):
        return _error(f"CNN score not numeric: {score!r}", raw=payload)

    # Prefer CNN's own label; otherwise derive from our bands.
    if isinstance(classification, str) and classification.strip():
        label = classification.strip().title()
    else:
        label = classify_score(score_val)

    return {
        "source": "cnn_unofficial",
        "status": "available",
        "score": score_val,
        "classification": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_timestamp": src_timestamp,
    }


def fetch_cnn_fear_greed(config: Optional[SentimentConfig] = None,
                         session: Optional[requests.Session] = None) -> dict:
    """Fetch the CNN Fear & Greed Index.

    Always returns a normalized dict and never raises. ``session`` is accepted
    mainly to make testing/mocking straightforward.

    Returns:
        {"source": "cnn_unofficial",
         "status": "available" | "error",
         "score": float | None,        # 0-100
         "classification": str,
         "timestamp": iso8601,
         ...}
    """
    config = config or SentimentConfig.from_env()
    url = config.cnn_url
    timeout = config.cnn_timeout

    try:
        getter = session.get if session is not None else requests.get
        response = getter(url, headers=_DEFAULT_HEADERS, timeout=timeout)
    except requests.exceptions.Timeout:
        logger.warning("CNN F&G request timed out after %ss", timeout)
        return _error(f"timeout after {timeout}s")
    except requests.exceptions.RequestException as exc:
        logger.warning("CNN F&G request failed: %s", exc)
        return _error(f"request error: {exc}")
    except Exception as exc:  # never let an unexpected error escape
        logger.warning("CNN F&G unexpected error: %s", exc)
        return _error(f"unexpected error: {exc}")

    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        logger.warning("CNN F&G returned HTTP %s", status_code)
        return _error(f"HTTP {status_code}")

    try:
        payload = response.json()
    except Exception as exc:
        logger.warning("CNN F&G response was not valid JSON: %s", exc)
        return _error(f"invalid JSON: {exc}")

    if not isinstance(payload, dict):
        return _error("CNN payload was not a JSON object", raw=None)

    result = _parse_payload(payload)
    if result["status"] == "available":
        logger.info(
            "CNN F&G: %.1f (%s)", result["score"], result["classification"]
        )
    return result
