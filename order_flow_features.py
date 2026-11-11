"""
Oracle 2.1 — order-flow features (ANALYTICS ONLY, pure, no I/O).

Phase 4 of the master plan: timestamped underlying trade-print features
(Time & Sales) and freshness-aware Level 2 confirmation, with top-of-book and
prints-only treated as their own variants. Missing, stale, or crossed data is
represented EXPLICITLY (``status`` field) and a confirmation variant is
applicable only when its required inputs qualify.

Design rules (mirror intraday_features.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer a safe omission over a guess; never raise on malformed input.
  * Offline-testable with synthetic prints/books.

Inputs are plain dicts so the same code runs on captured JSONL, live feeds,
and fixtures:
  * print:  {"ts": iso-or-epoch, "price": float, "size": float,
             "bid": float?, "ask": float?}   (bid/ask = NBBO at print time)
  * book:   {"ts": iso-or-epoch?, "bids": [level...], "asks": [level...]}
            where a level carries a size under quantity/size/qty (or is a bare
            number) and optionally a price under price / price.amount —
            the same duck-typed shapes ``rh_price_book`` already accepts.
  * top-of-book: {"ts": ..., "bid": float, "ask": float,
                  "bid_size": float, "ask_size": float}

Aggressor classification for prints: quote rule first (price at/above the ask
= buy, at/below the bid = sell), tick rule as fallback (vs the last different
price). Unclassifiable prints are excluded and reported via ``n_classified``
so coverage stays visible. The data SOURCE is the caller's to stamp
(``provenance`` passes through untouched).
"""

from datetime import datetime, timezone
from typing import List, Optional

# Explicit defaults so replays and live read the same numbers.
PRINT_WINDOW_S = 60.0          # look-back window for buy/sell volume
PRINT_MAX_AGE_S = 10.0         # newest print older than this => stale
BOOK_MAX_AGE_S = 5.0           # book/top-of-book older than this => stale
CONFIRM_IMBALANCE_MIN = 0.2    # |imbalance| needed to confirm a direction
DEPTH_LEVELS = 5               # L2 levels per side for verified depth

VARIANTS = ("prints_only", "top_of_book", "verified_depth")


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ts(x) -> Optional[float]:
    """Epoch seconds from an ISO-8601 string (Z ok) or a number; None if not."""
    n = _num(x)
    if n is not None:
        return n
    if isinstance(x, str):
        try:
            dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _level_size(level) -> Optional[float]:
    if isinstance(level, dict):
        for key in ("quantity", "size", "qty"):
            v = _num(level.get(key))
            if v is not None:
                return v
        return None
    return _num(level)


def _level_price(level) -> Optional[float]:
    if not isinstance(level, dict):
        return None
    p = level.get("price")
    if isinstance(p, dict):
        p = p.get("amount")
    return _num(p)


# --------------------------------------------------------------------------- #
# Trade prints (Time & Sales)
# --------------------------------------------------------------------------- #
def compute_print_features(prints, *, now_ts,
                           window_s: float = PRINT_WINDOW_S,
                           max_age_s: float = PRINT_MAX_AGE_S) -> dict:
    """Windowed buy/sell print volume and imbalance, freshness-aware.

    Returns a dict with ``status``:
      * ``missing`` — no parseable prints at all
      * ``stale``   — newest print older than ``max_age_s``
      * ``ok``      — fresh prints exist (classified volume may still be 0)
    plus ``n_prints`` (in window), ``n_classified``, ``buy_volume``,
    ``sell_volume``, ``print_imbalance`` in [-1, 1] (None when no classified
    volume), ``last_print_age_s``, ``window_s``. Never raises.
    """
    try:
        now = _ts(now_ts)
        if now is None:
            return {"status": "missing", "reason": "bad_now_ts"}
        rows = []
        for p in (prints or []):
            if not isinstance(p, dict):
                continue
            t, price, size = _ts(p.get("ts")), _num(p.get("price")), _num(p.get("size"))
            if t is None or price is None or size is None or size <= 0:
                continue
            if t > now:          # future-stamped print: not usable evidence
                continue
            rows.append((t, price, size, _num(p.get("bid")), _num(p.get("ask"))))
        if not rows:
            return {"status": "missing", "reason": "no_parseable_prints"}
        rows.sort(key=lambda r: r[0])
        last_age = now - rows[-1][0]
        status = "stale" if last_age > max_age_s else "ok"

        windowed = [r for r in rows if (now - r[0]) <= window_s]
        buy = sell = 0.0
        n_classified = 0
        last_price: Optional[float] = None
        last_dir = 0
        for t, price, size, bid, ask in rows:  # tick state over ALL prints
            side = 0
            if ask is not None and price >= ask:
                side = 1
            elif bid is not None and price <= bid:
                side = -1
            elif last_price is not None:
                if price > last_price:
                    side = 1
                elif price < last_price:
                    side = -1
                else:
                    side = last_dir   # zero-tick: inherit last direction
            if price != last_price:
                last_dir = (1 if last_price is not None and price > last_price
                            else (-1 if last_price is not None else 0))
                last_price = price
            if (t, price, size, bid, ask) in windowed and side != 0:
                n_classified += 1
                if side > 0:
                    buy += size
                else:
                    sell += size
        total = buy + sell
        return {
            "status": status,
            "n_prints": len(windowed),
            "n_classified": n_classified,
            "buy_volume": round(buy, 4),
            "sell_volume": round(sell, 4),
            "print_imbalance": (round((buy - sell) / total, 6)
                                if total > 0 else None),
            "last_print_age_s": round(last_age, 3),
            "window_s": window_s,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"status": "missing", "reason": "error"}


# --------------------------------------------------------------------------- #
# Level 2 book / top-of-book, freshness-aware
# --------------------------------------------------------------------------- #
def assess_book(book, *, now_ts=None,
                max_age_s: float = BOOK_MAX_AGE_S,
                levels: int = DEPTH_LEVELS) -> dict:
    """Freshness-aware L2 assessment over the top ``levels`` per side.

    ``status``: ``missing`` (no usable sides), ``stale`` (timestamp known and
    older than ``max_age_s`` — freshness requires BOTH ``book['ts']`` and
    ``now_ts``; without them the book cannot qualify as verified), ``crossed``
    (best bid above best ask), or ``ok``. Only ``ok`` books qualify for the
    verified-depth variant. Never raises.
    """
    try:
        if not isinstance(book, dict):
            return {"status": "missing", "reason": "no_book"}
        bids = list(book.get("bids") or [])[:max(1, int(levels))]
        asks = list(book.get("asks") or [])[:max(1, int(levels))]
        bid_sz = sum(s for s in (_level_size(b) for b in bids) if s is not None)
        ask_sz = sum(s for s in (_level_size(a) for a in asks) if s is not None)
        if bid_sz <= 0 or ask_sz <= 0:
            return {"status": "missing", "reason": "empty_side"}
        t, now = _ts(book.get("ts")), _ts(now_ts)
        if t is None or now is None:
            return {"status": "stale", "reason": "no_timestamp",
                    "book_age_s": None}
        age = now - t
        if age > max_age_s or age < 0:
            return {"status": "stale", "reason": "age_exceeded",
                    "book_age_s": round(age, 3)}
        best_bid, best_ask = _level_price(bids[0]) if bids else None, \
            _level_price(asks[0]) if asks else None
        if best_bid is not None and best_ask is not None and best_bid > best_ask:
            return {"status": "crossed", "reason": "bid_above_ask",
                    "best_bid": best_bid, "best_ask": best_ask,
                    "book_age_s": round(age, 3)}
        total = bid_sz + ask_sz
        return {
            "status": "ok",
            "orderbook_imbalance": round((bid_sz - ask_sz) / total, 6),
            "orderbook_bid_size": round(bid_sz, 4),
            "orderbook_ask_size": round(ask_sz, 4),
            "orderbook_levels": min(len(bids), len(asks)),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "book_age_s": round(age, 3),
        }
    except Exception:  # pragma: no cover - fail-open
        return {"status": "missing", "reason": "error"}


def assess_top_of_book(quote, *, now_ts=None,
                       max_age_s: float = BOOK_MAX_AGE_S) -> dict:
    """Freshness-aware top-of-book assessment (its own variant, not L2).

    ``quote``: {"ts", "bid", "ask", "bid_size", "ask_size"}. Same status
    vocabulary as :func:`assess_book`. Never raises.
    """
    try:
        if not isinstance(quote, dict):
            return {"status": "missing", "reason": "no_quote"}
        bid, ask = _num(quote.get("bid")), _num(quote.get("ask"))
        bsz, asz = _num(quote.get("bid_size")), _num(quote.get("ask_size"))
        if bsz is None or asz is None or bsz <= 0 or asz <= 0:
            return {"status": "missing", "reason": "no_size"}
        t, now = _ts(quote.get("ts")), _ts(now_ts)
        if t is None or now is None:
            return {"status": "stale", "reason": "no_timestamp",
                    "quote_age_s": None}
        age = now - t
        if age > max_age_s or age < 0:
            return {"status": "stale", "reason": "age_exceeded",
                    "quote_age_s": round(age, 3)}
        if bid is not None and ask is not None and bid > ask:
            return {"status": "crossed", "reason": "bid_above_ask",
                    "best_bid": bid, "best_ask": ask,
                    "quote_age_s": round(age, 3)}
        total = bsz + asz
        return {
            "status": "ok",
            "tob_imbalance": round((bsz - asz) / total, 6),
            "bid_size": round(bsz, 4),
            "ask_size": round(asz, 4),
            "best_bid": bid,
            "best_ask": ask,
            "quote_age_s": round(age, 3),
        }
    except Exception:  # pragma: no cover - fail-open
        return {"status": "missing", "reason": "error"}


# --------------------------------------------------------------------------- #
# Confirmation variants
# --------------------------------------------------------------------------- #
def confirm_order_flow(direction: str, *, variant: str = "prints_only",
                       prints_feats: Optional[dict] = None,
                       book_feats: Optional[dict] = None,
                       top_feats: Optional[dict] = None,
                       min_imbalance: float = CONFIRM_IMBALANCE_MIN) -> dict:
    """Directional order-flow confirmation, applied ONLY when inputs qualify.

    ``variant``:
      * ``prints_only``    — needs ``prints_feats`` with status ``ok`` and a
                             defined ``print_imbalance``
      * ``top_of_book``    — needs ``top_feats`` with status ``ok``
      * ``verified_depth`` — needs ``book_feats`` with status ``ok``
    Returns {variant, direction, applicable, reason, imbalance, threshold,
    confirmed}. ``confirmed`` is None when not applicable: the caller must
    treat the variant as NOT run rather than as a rejection. Never raises.
    """
    try:
        if direction not in ("long", "short") or variant not in VARIANTS:
            return {"variant": variant, "direction": direction,
                    "applicable": False, "reason": "bad_arguments",
                    "imbalance": None, "threshold": min_imbalance,
                    "confirmed": None}
        src = {"prints_only": (prints_feats, "print_imbalance"),
               "top_of_book": (top_feats, "tob_imbalance"),
               "verified_depth": (book_feats, "orderbook_imbalance")}[variant]
        feats, key = src
        if not isinstance(feats, dict) or feats.get("status") != "ok":
            reason = ("input_missing" if not isinstance(feats, dict)
                      else f"input_{feats.get('status', 'unusable')}")
            return {"variant": variant, "direction": direction,
                    "applicable": False, "reason": reason,
                    "imbalance": None, "threshold": min_imbalance,
                    "confirmed": None}
        imb = _num(feats.get(key))
        if imb is None:
            return {"variant": variant, "direction": direction,
                    "applicable": False, "reason": "no_imbalance",
                    "imbalance": None, "threshold": min_imbalance,
                    "confirmed": None}
        signed = imb if direction == "long" else -imb
        return {"variant": variant, "direction": direction,
                "applicable": True, "reason": "ok",
                "imbalance": imb, "threshold": min_imbalance,
                "confirmed": signed >= min_imbalance}
    except Exception:  # pragma: no cover - fail-open
        return {"variant": variant, "direction": direction,
                "applicable": False, "reason": "error",
                "imbalance": None, "threshold": min_imbalance,
                "confirmed": None}


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    now = 1_700_000_100.0

    # --- Prints: quote-rule buys dominate -> positive imbalance ----------- #
    prints = [
        {"ts": now - 30, "price": 100.01, "size": 300, "bid": 100.00, "ask": 100.01},
        {"ts": now - 20, "price": 100.01, "size": 200, "bid": 100.00, "ask": 100.01},
        {"ts": now - 10, "price": 100.00, "size": 100, "bid": 100.00, "ask": 100.01},
        {"ts": now - 2, "price": 100.02, "size": 150, "bid": 100.01, "ask": 100.02},
    ]
    pf = compute_print_features(prints, now_ts=now)
    if pf.get("status") != "ok":
        print("FAIL: fresh prints should be ok", pf); ok = False
    if pf.get("print_imbalance") is None or pf["print_imbalance"] <= 0:
        print("FAIL: buy-heavy tape -> positive imbalance", pf); ok = False
    if pf.get("buy_volume") != 650.0 or pf.get("sell_volume") != 100.0:
        print("FAIL: quote-rule volumes", pf); ok = False

    # Tick rule fallback when no NBBO on the print.
    tick = [{"ts": now - 9, "price": 100.00, "size": 100},
            {"ts": now - 8, "price": 100.05, "size": 200},
            {"ts": now - 7, "price": 100.01, "size": 50}]
    tf = compute_print_features(tick, now_ts=now)
    if tf.get("buy_volume") != 200.0 or tf.get("sell_volume") != 50.0:
        print("FAIL: tick-rule classification", tf); ok = False
    if tf.get("n_classified") != 2:  # the first print has no reference
        print("FAIL: unclassifiable prints must be excluded", tf); ok = False

    # Stale / missing prints are explicit.
    st = compute_print_features(prints, now_ts=now + 120)
    if st.get("status") != "stale":
        print("FAIL: old tape should be stale", st); ok = False
    if compute_print_features([], now_ts=now).get("status") != "missing":
        print("FAIL: empty tape should be missing"); ok = False
    if compute_print_features([{"ts": now + 60, "price": 1, "size": 1}],
                              now_ts=now).get("status") != "missing":
        print("FAIL: future-stamped prints are not evidence"); ok = False

    # --- Book: ok / stale / crossed / missing ------------------------------ #
    book = {"ts": now - 1,
            "bids": [{"price": {"amount": "100.00"}, "quantity": 300},
                     {"price": {"amount": "99.99"}, "quantity": 200}],
            "asks": [{"price": {"amount": "100.01"}, "quantity": 100},
                     {"price": {"amount": "100.02"}, "quantity": 50}]}
    bf = assess_book(book, now_ts=now)
    if bf.get("status") != "ok" or bf.get("orderbook_imbalance", 0) <= 0:
        print("FAIL: fresh bid-heavy book", bf); ok = False
    if assess_book(book, now_ts=now + 60).get("status") != "stale":
        print("FAIL: old book should be stale"); ok = False
    if assess_book(dict(book, ts=None), now_ts=now).get("status") != "stale":
        print("FAIL: unstamped book cannot be verified"); ok = False
    xb = dict(book, bids=[{"price": {"amount": "100.05"}, "quantity": 300}])
    if assess_book(xb, now_ts=now).get("status") != "crossed":
        print("FAIL: bid above ask should be crossed"); ok = False
    if assess_book({"bids": [], "asks": []}, now_ts=now).get("status") != "missing":
        print("FAIL: empty book should be missing"); ok = False

    # --- Top of book -------------------------------------------------------- #
    top = {"ts": now - 1, "bid": 100.0, "ask": 100.01,
           "bid_size": 900, "ask_size": 100}
    tb = assess_top_of_book(top, now_ts=now)
    if tb.get("status") != "ok" or tb.get("tob_imbalance", 0) <= 0:
        print("FAIL: fresh bid-heavy top of book", tb); ok = False
    if assess_top_of_book(dict(top, bid=100.05), now_ts=now).get("status") != "crossed":
        print("FAIL: crossed top of book"); ok = False

    # --- Confirmation variants only apply when inputs qualify -------------- #
    c1 = confirm_order_flow("long", variant="prints_only", prints_feats=pf)
    if not (c1["applicable"] and c1["confirmed"] is True):
        print("FAIL: buy tape should confirm long", c1); ok = False
    c2 = confirm_order_flow("short", variant="prints_only", prints_feats=pf)
    if not (c2["applicable"] and c2["confirmed"] is False):
        print("FAIL: buy tape should NOT confirm short", c2); ok = False
    c3 = confirm_order_flow("long", variant="verified_depth", book_feats=bf)
    if not (c3["applicable"] and c3["confirmed"] is True):
        print("FAIL: bid-heavy verified depth should confirm long", c3); ok = False
    c4 = confirm_order_flow("long", variant="verified_depth",
                            book_feats=assess_book(book, now_ts=now + 60))
    if c4["applicable"] or c4["confirmed"] is not None or c4["reason"] != "input_stale":
        print("FAIL: stale book -> variant not applicable", c4); ok = False
    c5 = confirm_order_flow("long", variant="top_of_book", top_feats=tb)
    if not (c5["applicable"] and c5["confirmed"] is True):
        print("FAIL: bid-heavy top of book should confirm long", c5); ok = False
    c6 = confirm_order_flow("long", variant="prints_only")
    if c6["applicable"] or c6["reason"] != "input_missing":
        print("FAIL: absent prints -> not applicable", c6); ok = False
    c7 = confirm_order_flow("up", variant="prints_only", prints_feats=pf)
    if c7["applicable"] or c7["reason"] != "bad_arguments":
        print("FAIL: bad direction -> bad_arguments", c7); ok = False

    print("order_flow_features self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
