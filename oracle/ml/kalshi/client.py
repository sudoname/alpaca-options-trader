"""Read-only public Kalshi REST client (Oracle 4.0.2 P1-6).

Rules:
  * research-path only -- never imported by the production trading path;
  * strictly read-only: GET requests against the public market-data API;
    no order endpoints exist in this module at all;
  * no credentials required for public market data; an optional API key
    header is supported but never logged;
  * typed refusals: every failure raises ``KalshiUnavailable(reason,
    detail)`` -- callers must handle it, never a silent default.

Tests monkeypatch the module-level ``_http_get_json`` symbol, same
pattern as ``oracle.ml.bars_feed`` / ``bars_backfill``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

#: public trade API v2 base (elections host serves all public market data)
DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

DEFAULT_PAGE_LIMIT = 200
DEFAULT_MAX_PAGES = 50
DEFAULT_TIMEOUT = 20.0


class KalshiUnavailable(Exception):
    """Typed refusal: Kalshi data could not be fetched."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _http_get_json(url: str, headers: Dict[str, str], params: Dict,
                   timeout: float) -> Dict:
    """Thin HTTP wrapper; tests monkeypatch this symbol."""
    import requests

    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json() or {}


def get_markets(
    *,
    base_url: str = DEFAULT_BASE_URL,
    series_ticker: Optional[str] = None,
    event_ticker: Optional[str] = None,
    status: Optional[str] = "open",
    api_key: Optional[str] = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[Dict]:
    """Fetch markets (paginated) from GET /markets.

    Returns the raw market dicts as served by the API (prices in cents).
    Raises ``KalshiUnavailable`` on any transport/shape failure.
    """
    url = f"{base_url.rstrip('/')}/markets"
    headers: Dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base_params: Dict = {"limit": int(limit)}
    if series_ticker:
        base_params["series_ticker"] = str(series_ticker)
    if event_ticker:
        base_params["event_ticker"] = str(event_ticker)
    if status:
        base_params["status"] = str(status)

    out: List[Dict] = []
    cursor: Optional[str] = None
    for _ in range(max(1, int(max_pages))):
        params = dict(base_params)
        if cursor:
            params["cursor"] = cursor
        try:
            payload = _http_get_json(url, headers, params, timeout) or {}
        except Exception as exc:  # noqa: BLE001 - typed refusal boundary
            raise KalshiUnavailable("FETCH_FAILED", str(exc)) from exc
        markets = payload.get("markets")
        if markets is None:
            raise KalshiUnavailable(
                "BAD_PAYLOAD", "response missing 'markets' list")
        if not isinstance(markets, list):
            raise KalshiUnavailable(
                "BAD_PAYLOAD", f"'markets' is {type(markets).__name__}")
        out.extend(m for m in markets if isinstance(m, dict))
        cursor = payload.get("cursor") or None
        if not cursor:
            break
    return out
