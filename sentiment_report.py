"""
sentiment_report.py — READ-ONLY Fear & Greed adapter for the Oracle dashboard.

Surfaces the project's existing market-sentiment engine (``sentiment/``) to the
read-only web dashboard. The heavy lifting already exists:

  * ``sentiment.SentimentService`` blends CNN's Fear & Greed index with a
    self-computed 7-component CNN-style score and caches the result.
  * ``sentiment.AlpacaMarketDataProvider`` feeds the custom score daily closes
    via Alpaca's market-data bars endpoint — HTTP GET only, never an order.

This adapter builds that service from read-only Alpaca creds, calls
``get_sentiment()``, and reshapes the payload into the dashboard's
verdict-carrying convention:

    {"verdict": "OK" | "INSUFFICIENT_DATA",
     "score": float|None, "classification": str, "source": "blend"|"custom"|"cnn",
     "components": [ {name, available, score, detail}, ... ],
     "cnn_score": float|None, "custom_score": float|None,
     "available_count": int, "unavailable_components": [...],
     "from_cache": bool, "timestamp": iso8601}

It is read-only: only HTTP GETs flow through the underlying provider/scraper and
no trade/order/mutation path is imported. On missing creds / no network / any
error it FAILS OPEN to ``verdict: INSUFFICIENT_DATA`` so the widget degrades
cleanly. The service factory is injectable so unit tests run fully offline.
"""

from typing import Callable, Dict, List, Optional

VERDICT_OK = "OK"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"


def _alpaca_headers() -> Optional[Dict[str, str]]:
    """Read-only Alpaca auth headers from config, or None when creds are absent."""
    try:
        from config_loader import ConfigLoader
        env = ConfigLoader()
        key = env.get("ALPACA_API_KEY", "")
        secret = env.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            return None
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    except Exception:
        return None


def _default_service_factory():
    """Build a live, read-only SentimentService. None when creds are absent.

    The custom score still needs Alpaca creds for its market-data feed; without
    them only the CNN scraper could contribute, so we report INSUFFICIENT rather
    than half a score. The service itself fails open internally.
    """
    headers = _alpaca_headers()
    if not headers:
        return None
    try:
        from config_loader import ConfigLoader
        env = ConfigLoader()
        feed = env.get("SCREENER_ALPACA_FEED", "iex") or "iex"
        data_url = env.get("ALPACA_DATA_URL", "https://data.alpaca.markets") \
            or "https://data.alpaca.markets"
    except Exception:
        feed, data_url = "iex", "https://data.alpaca.markets"
    from sentiment import AlpacaMarketDataProvider, SentimentService
    provider = AlpacaMarketDataProvider(data_url=data_url, headers=headers,
                                        feed=feed)
    return SentimentService(provider)


def _shape(payload: Optional[dict]) -> dict:
    """Reshape a SentimentService payload into the dashboard's verdict dict."""
    payload = payload if isinstance(payload, dict) else {}
    primary = payload.get("primary_score") or {}
    custom = payload.get("custom_score") or {}
    cnn = payload.get("cnn_score") or {}

    score = primary.get("score")
    status = primary.get("status")
    verdict = VERDICT_OK if (status == "available" and score is not None) \
        else VERDICT_INSUFFICIENT

    # The 7 CNN-style sub-components live on the custom score; fall back to the
    # blend's source breakdown when the custom score is unavailable.
    components: List[dict] = custom.get("components") or primary.get("components") or []

    return {
        "verdict": verdict,
        "score": score,
        "classification": primary.get("classification") or "Unknown",
        "source": payload.get("primary_source"),
        "components": components,
        "available_count": custom.get("available_count"),
        "unavailable_components": custom.get("unavailable_components") or [],
        "cnn_score": cnn.get("score") if isinstance(cnn, dict) else None,
        "custom_score": custom.get("score") if isinstance(custom, dict) else None,
        "from_cache": bool(payload.get("from_cache")),
        "timestamp": payload.get("timestamp"),
    }


def compute_sentiment_report(
    *,
    service_factory: Optional[Callable[[], object]] = None,
) -> dict:
    """Assemble the dashboard Fear & Greed report. Never raises.

    ``service_factory`` is injectable for offline tests; it must return an
    object exposing ``get_sentiment()`` (or ``None`` to signal no creds).
    """
    try:
        factory = service_factory or _default_service_factory
        service = factory()
        if service is None:
            return {"verdict": VERDICT_INSUFFICIENT,
                    "error": "no market data credentials",
                    "components": []}
        payload = service.get_sentiment()
        return _shape(payload)
    except Exception as ex:
        return {"verdict": VERDICT_INSUFFICIENT, "error": str(ex),
                "components": []}


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses injected fake services)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # No creds / no factory output -> INSUFFICIENT (no network).
    r = compute_sentiment_report(service_factory=lambda: None)
    if r.get("verdict") != VERDICT_INSUFFICIENT:
        print("FAIL: None service should be INSUFFICIENT:", r); ok = False

    # A raising factory must still fail open.
    def _boom():
        raise RuntimeError("network down")
    if compute_sentiment_report(service_factory=_boom).get("verdict") != VERDICT_INSUFFICIENT:
        print("FAIL: raising factory should be INSUFFICIENT"); ok = False

    # An available payload -> verdict OK, score + classification + components
    # passed through.
    class _AvailService:
        def get_sentiment(self):
            return {
                "cnn_score": {"status": "available", "score": 60.0},
                "custom_score": {
                    "source": "custom", "status": "available", "score": 50.0,
                    "available_count": 4,
                    "unavailable_components": ["put_call_ratio"],
                    "components": [
                        {"name": "market_momentum", "available": True,
                         "score": 55.0, "detail": "SPY vs MA"},
                        {"name": "market_volatility", "available": True,
                         "score": 45.0, "detail": "VIX inverse pct"},
                    ],
                },
                "primary_score": {"source": "blend", "status": "available",
                                  "score": 55.0, "classification": "Neutral"},
                "primary_source": "blend",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "from_cache": False,
            }

    rep = compute_sentiment_report(service_factory=lambda: _AvailService())
    if rep.get("verdict") != VERDICT_OK:
        print("FAIL: available payload should be OK:", rep); ok = False
    if rep.get("score") != 55.0 or rep.get("classification") != "Neutral":
        print("FAIL: score/classification passthrough:", rep); ok = False
    if rep.get("source") != "blend":
        print("FAIL: source should be blend:", rep); ok = False
    if not rep.get("components") or rep.get("cnn_score") != 60.0 \
            or rep.get("custom_score") != 50.0:
        print("FAIL: components/sub-scores not surfaced:", rep); ok = False

    # An error payload (too few components) -> INSUFFICIENT, never a crash.
    class _ErrService:
        def get_sentiment(self):
            return {
                "cnn_score": None,
                "custom_score": {"source": "custom", "status": "error",
                                 "score": None, "components": []},
                "primary_score": {"source": "none", "status": "error",
                                  "score": None, "classification": "Unknown"},
                "primary_source": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }

    er = compute_sentiment_report(service_factory=lambda: _ErrService())
    if er.get("verdict") != VERDICT_INSUFFICIENT:
        print("FAIL: error payload should be INSUFFICIENT:", er); ok = False

    print("sentiment_report self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
