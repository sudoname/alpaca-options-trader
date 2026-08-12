"""
Oracle 2.1 — Robinhood price-book order-flow imbalance (Feature 2b).

Robinhood publishes a Level-II style price book (resting bid/ask depth) that
Alpaca's market-data plan does not. This module turns that book into a single
order-flow imbalance number for the intraday profile:

    imbalance = (Σ bid_size − Σ ask_size) / (Σ bid_size + Σ ask_size)   ∈ [-1, 1]

positive = more resting demand than supply near the touch (a short-horizon
bullish tell), negative = the opposite.

This is a *network* module, but it is engineered to never disturb the trade
path:
  * The pure math (:func:`_imbalance_from_book`) is offline-testable.
  * ``robin_stocks`` / ``pyotp`` are imported lazily; absent -> ``None``.
  * Credentials come from the shell-env-first ``ConfigLoader`` (RH_USERNAME /
    RH_PASSWORD / RH_MFA_SECRET / RH_PICKLE_PATH); absent -> ``None``.
  * A short TTL cache throttles calls; a circuit breaker stops hammering a
    failing session for a cooldown; every public path HARD fails open to
    ``None`` on any error. Only called in intraday mode, behind
    ``USE_ORDERBOOK_IMBALANCE``.
"""

import time as _time
from typing import Dict, List, Optional

RH_PRICEBOOK_TTL_SEC = 5.0
RH_CIRCUIT_MAX_FAILS = 3
RH_CIRCUIT_COOLDOWN_SEC = 300.0
DEFAULT_LEVELS = 5


# --------------------------------------------------------------------------- #
# Pure math (offline-testable)
# --------------------------------------------------------------------------- #
def _num(x) -> Optional[float]:
    try:
        f = float(x)
        return f
    except (TypeError, ValueError):
        return None


def _level_size(level) -> Optional[float]:
    """Resting quantity at a book level, tolerant of RH's nested shapes."""
    if isinstance(level, dict):
        for key in ("quantity", "size", "qty"):
            v = _num(level.get(key))
            if v is not None:
                return v
        return None
    return _num(level)


def _imbalance_from_book(bids, asks, levels: int = DEFAULT_LEVELS) -> Optional[Dict]:
    """Pure order-book imbalance over the top ``levels`` on each side.

    Returns ``{orderbook_imbalance, orderbook_depth_ratio, orderbook_bid_size,
    orderbook_ask_size, orderbook_levels}`` or ``None`` when the book is empty /
    unparseable. Never raises.
    """
    try:
        bids = list(bids or [])[:max(1, int(levels))]
        asks = list(asks or [])[:max(1, int(levels))]
        bid_sz = sum(s for s in (_level_size(b) for b in bids) if s is not None)
        ask_sz = sum(s for s in (_level_size(a) for a in asks) if s is not None)
        total = bid_sz + ask_sz
        if total <= 0:
            return None
        imbalance = (bid_sz - ask_sz) / total
        depth_ratio = (bid_sz / ask_sz) if ask_sz > 0 else float("inf")
        return {
            "orderbook_imbalance": round(imbalance, 6),
            "orderbook_depth_ratio": (round(depth_ratio, 6)
                                      if depth_ratio != float("inf") else None),
            "orderbook_bid_size": round(bid_sz, 4),
            "orderbook_ask_size": round(ask_sz, 4),
            "orderbook_levels": min(len(bids), len(asks)),
        }
    except Exception:  # pragma: no cover - fail-open
        return None


# --------------------------------------------------------------------------- #
# Live client (network; lazy, cached, circuit-broken, fail-open)
# --------------------------------------------------------------------------- #
class RHPriceBookClient:
    """Thin Robinhood price-book reader. Every public path fails open to None."""

    def __init__(self, username: str, password: str,
                 mfa_secret: Optional[str] = None,
                 pickle_path: Optional[str] = None):
        self.username = username
        self.password = password
        self.mfa_secret = mfa_secret
        self.pickle_path = pickle_path
        self._logged_in = False
        self._cache: Dict = {}          # symbol -> (ts, result)
        self._fail_count = 0
        self._circuit_open_until = 0.0

    # --- session ------------------------------------------------------------
    def _ensure_login(self) -> bool:
        if self._logged_in:
            return True
        try:
            import robin_stocks.robinhood as rh
        except Exception:
            return False
        try:
            mfa_code = None
            if self.mfa_secret:
                try:
                    import pyotp
                    mfa_code = pyotp.TOTP(self.mfa_secret).now()
                except Exception:
                    mfa_code = None
            kwargs = {"username": self.username, "password": self.password,
                      "store_session": True}
            if mfa_code:
                kwargs["mfa_code"] = mfa_code
            if self.pickle_path:
                kwargs["pickle_name"] = self.pickle_path
            rh.login(**kwargs)
            self._logged_in = True
            return True
        except Exception:
            return False

    # --- circuit breaker ----------------------------------------------------
    def _circuit_blocked(self) -> bool:
        return _time.time() < self._circuit_open_until

    def _note_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count >= RH_CIRCUIT_MAX_FAILS:
            self._circuit_open_until = _time.time() + RH_CIRCUIT_COOLDOWN_SEC
            self._fail_count = 0

    def _note_success(self) -> None:
        self._fail_count = 0

    # --- public -------------------------------------------------------------
    def get_order_book_imbalance(self, symbol: str,
                                 levels: int = DEFAULT_LEVELS) -> Optional[Dict]:
        try:
            now = _time.time()
            cached = self._cache.get(symbol)
            if cached and (now - cached[0]) < RH_PRICEBOOK_TTL_SEC:
                return cached[1]
            if self._circuit_blocked():
                return None
            if not self._ensure_login():
                self._note_failure()
                return None
            try:
                import robin_stocks.robinhood as rh
                book = rh.get_pricebook_by_symbol(symbol)
            except Exception:
                self._note_failure()
                return None
            if not isinstance(book, dict):
                self._note_failure()
                return None
            result = _imbalance_from_book(
                book.get("bids"), book.get("asks"), levels)
            self._note_success()
            self._cache[symbol] = (now, result)
            return result
        except Exception:  # pragma: no cover - fail-open
            return None


# --------------------------------------------------------------------------- #
# Module singleton
# --------------------------------------------------------------------------- #
_singleton: Optional[RHPriceBookClient] = None
_singleton_tried = False


def get_client() -> Optional[RHPriceBookClient]:
    """Return a cached client, or ``None`` when creds/library are unavailable."""
    global _singleton, _singleton_tried
    if _singleton is not None:
        return _singleton
    if _singleton_tried:
        return None
    _singleton_tried = True
    try:
        from config_loader import ConfigLoader
        env = ConfigLoader()
        user = env.get("RH_USERNAME", "")
        pw = env.get("RH_PASSWORD", "")
        if not user or not pw:
            return None
        _singleton = RHPriceBookClient(
            user, pw,
            mfa_secret=env.get("RH_MFA_SECRET", "") or None,
            pickle_path=env.get("RH_PICKLE_PATH", "") or None,
        )
        return _singleton
    except Exception:
        return None


def get_order_book_imbalance(symbol: str,
                             levels: int = DEFAULT_LEVELS) -> Optional[Dict]:
    """Convenience wrapper: order-book imbalance for ``symbol`` or ``None``."""
    client = get_client()
    if client is None:
        return None
    return client.get_order_book_imbalance(symbol, levels)


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no robin_stocks required)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Bid-heavy book -> positive imbalance, depth ratio > 1.
    bids = [{"price": {"amount": "100.00"}, "quantity": 300},
            {"price": {"amount": "99.99"}, "quantity": 200}]
    asks = [{"price": {"amount": "100.01"}, "quantity": 100},
            {"price": {"amount": "100.02"}, "quantity": 50}]
    r = _imbalance_from_book(bids, asks)
    if not r or r["orderbook_imbalance"] <= 0:
        print("FAIL: bid-heavy book should be positive", r); ok = False
    if not r or r["orderbook_depth_ratio"] <= 1.0:
        print("FAIL: bid-heavy depth ratio should exceed 1", r); ok = False

    # Ask-heavy book -> negative imbalance.
    r2 = _imbalance_from_book(asks, bids)  # swap roles
    if not r2 or r2["orderbook_imbalance"] >= 0:
        print("FAIL: ask-heavy book should be negative", r2); ok = False

    # Balanced book -> ~0 imbalance.
    bal = [{"quantity": 100}]
    r3 = _imbalance_from_book(bal, bal)
    if not r3 or abs(r3["orderbook_imbalance"]) > 1e-9:
        print("FAIL: balanced book should be ~0", r3); ok = False

    # Levels cap: only the top N per side count.
    many_bids = [{"quantity": 100} for _ in range(20)]
    many_asks = [{"quantity": 1} for _ in range(20)]
    r4 = _imbalance_from_book(many_bids, many_asks, levels=5)
    if not r4 or r4["orderbook_levels"] != 5:
        print("FAIL: levels should cap to 5", r4); ok = False
    if not r4 or abs(r4["orderbook_bid_size"] - 500) > 1e-9:
        print("FAIL: only top-5 bid size (500) should count", r4); ok = False

    # Empty / garbage -> None.
    for junk_b, junk_a in ((None, None), ([], []), ("x", "y"), ([{"z": 1}], [{"z": 1}])):
        if _imbalance_from_book(junk_b, junk_a) is not None:
            print("FAIL: empty/garbage book should be None", junk_b, junk_a); ok = False

    # Offline: no creds / no robin_stocks -> get_order_book_imbalance is None.
    # (get_client reads ConfigLoader; without RH_USERNAME/PASSWORD it returns None.)
    if get_order_book_imbalance("SPY") is not None:
        # Only a real configured+logged-in session could make this non-None;
        # in the offline self-test environment it must fail open.
        print("FAIL: offline order-book imbalance should be None"); ok = False

    print("rh_price_book self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
