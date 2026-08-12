"""
Oracle 2.1 — Feature 3: options-flow / positioning features.

Two layers, cleanly split so the analytics stay offline-testable:

  * ``compute_flow_features(chain_rows)`` — PURE. Turns an option-chain snapshot
    (list of duck-typed dicts, same shape as ``option_chain._process_contract``)
    into four positioning features:

      - ``cp_volume_skew`` — (call_vol − put_vol) / (call_vol + put_vol) in
        [−1, 1]. Positive = call-side flow (typically bullish); negative =
        put-side flow.
      - ``cp_oi_skew`` — the same ratio on open interest: standing positioning
        rather than today's flow.
      - ``unusual_volume`` — total contract volume / total open interest. Values
        above ~1 mean today's volume rivals the entire standing OI: a fresh,
        conviction-driven positioning day.
      - ``iv_term_structure`` — front-expiry mean IV minus back-expiry mean IV.
        Positive = backwardation (near-term stress / event premium); negative =
        the usual contango.

    Returns ``{"flow_status": "OK", ...}`` or ``{"flow_status": "INSUFFICIENT",
    "flow_reason": ...}``. Never raises.

  * ``fetch_chain_snapshot(underlying, headers, feed, ...)`` — thin, injectable
    network layer over Alpaca ``GET /v1beta1/options/snapshots/{underlying}``.
    Returns chain rows in the same dict schema the pure layer consumes (parsing
    the OCC option symbol for strike / expiry / type). Bounded to strikes within
    ``pct_window`` of ``spot`` and the nearest ``max_expiries`` expiries so a
    liquid underlying doesn't pull thousands of contracts. Fail-open to ``[]``.
"""

from typing import List, Optional


# --------------------------------------------------------------------------- #
# Pure analytics
# --------------------------------------------------------------------------- #
def _num(x) -> Optional[float]:
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _row_get(row, *keys):
    if not isinstance(row, dict):
        return None
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _row_type(row) -> Optional[str]:
    t = _row_get(row, "type", "option_type", "right")
    if t is None:
        return None
    t = str(t).strip().lower()
    if t in ("c", "call"):
        return "call"
    if t in ("p", "put"):
        return "put"
    return None


def _skew(call_amt: float, put_amt: float) -> Optional[float]:
    total = call_amt + put_amt
    if total <= 0:
        return None
    return max(-1.0, min(1.0, (call_amt - put_amt) / total))


def compute_flow_features(chain_rows) -> dict:
    """Positioning features for an option-chain snapshot.

    Returns ``{flow_status: "OK", cp_volume_skew, cp_oi_skew, unusual_volume,
    iv_term_structure}`` (any individual field may be ``None`` when its inputs
    are missing) or ``{flow_status: "INSUFFICIENT", flow_reason}``. Never raises.
    """
    try:
        rows = []
        for r in (chain_rows or []):
            t = _row_type(r)
            if t is None:
                continue
            rows.append({
                "type": t,
                "expiration_date": _row_get(r, "expiration_date", "expiry", "expiration"),
                "volume": _num(_row_get(r, "volume", "vol")) or 0.0,
                "oi": _num(_row_get(r, "open_interest", "oi", "openInterest")) or 0.0,
                "iv": _num(_row_get(r, "iv", "implied_volatility", "impliedVolatility")),
            })
        if not rows:
            return {"flow_status": "INSUFFICIENT", "flow_reason": "no chain rows"}

        call_vol = sum(r["volume"] for r in rows if r["type"] == "call")
        put_vol = sum(r["volume"] for r in rows if r["type"] == "put")
        call_oi = sum(r["oi"] for r in rows if r["type"] == "call")
        put_oi = sum(r["oi"] for r in rows if r["type"] == "put")

        cp_volume_skew = _skew(call_vol, put_vol)
        cp_oi_skew = _skew(call_oi, put_oi)

        total_vol = call_vol + put_vol
        total_oi = call_oi + put_oi
        unusual_volume = round(total_vol / total_oi, 4) if total_oi > 0 else None

        # IV term structure: mean IV of the nearest expiry minus the farthest.
        by_exp: dict = {}
        for r in rows:
            iv = r["iv"]
            exp = r["expiration_date"]
            if iv is None or exp is None:
                continue
            by_exp.setdefault(str(exp), []).append(iv)
        iv_term_structure = None
        if len(by_exp) >= 2:
            exps = sorted(by_exp)
            front = sum(by_exp[exps[0]]) / len(by_exp[exps[0]])
            back = sum(by_exp[exps[-1]]) / len(by_exp[exps[-1]])
            iv_term_structure = round(front - back, 6)

        if (cp_volume_skew is None and cp_oi_skew is None
                and unusual_volume is None and iv_term_structure is None):
            return {"flow_status": "INSUFFICIENT",
                    "flow_reason": "no volume/oi/iv"}

        return {
            "flow_status": "OK",
            "cp_volume_skew": round(cp_volume_skew, 6) if cp_volume_skew is not None else None,
            "cp_oi_skew": round(cp_oi_skew, 6) if cp_oi_skew is not None else None,
            "unusual_volume": unusual_volume,
            "iv_term_structure": iv_term_structure,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"flow_status": "INSUFFICIENT", "flow_reason": "error"}


# --------------------------------------------------------------------------- #
# Thin injectable network layer (Alpaca options snapshots)
# --------------------------------------------------------------------------- #
DATA_URL = "https://data.alpaca.markets"


def _parse_occ(sym: str) -> Optional[dict]:
    """Parse an OCC option symbol → {type, strike, expiration_date}.

    Layout: ``<root><yymmdd><C|P><strike*1000, 8 digits>``. The root is
    variable-length, so the trailing 15 chars carry everything we need.
    """
    if not isinstance(sym, str) or len(sym) < 16:
        return None
    tail = sym[-15:]
    yy, mm, dd, cp, strike_raw = tail[0:2], tail[2:4], tail[4:6], tail[6], tail[7:15]
    if cp.upper() not in ("C", "P") or not (yy + mm + dd + strike_raw).isdigit():
        return None
    try:
        strike = int(strike_raw) / 1000.0
    except ValueError:
        return None
    return {
        "type": "call" if cp.upper() == "C" else "put",
        "strike": strike,
        "expiration_date": f"20{yy}-{mm}-{dd}",
    }


def fetch_chain_snapshot(underlying: str, headers: dict, feed: str = "indicative",
                         spot: Optional[float] = None, pct_window: float = 0.15,
                         max_expiries: int = 3, page_limit: int = 5,
                         requests_mod=None) -> List[dict]:
    """Fetch an option-chain snapshot for ``underlying`` as pure chain rows.

    Injectable: pass ``requests_mod`` to stub the HTTP layer in tests. Returns a
    list of ``{symbol, type, strike, expiration_date, volume, gamma, iv, ...}``
    dicts consumable by :func:`compute_flow_features` / ``oracle.gex.compute_gex``,
    or ``[]`` on any failure (fail-open — never raises).
    """
    try:
        if requests_mod is None:
            import requests as requests_mod  # local import keeps the pure layer clean
        rows: List[dict] = []
        page_token = None
        pages = 0
        while pages < max(1, page_limit):
            params = {"feed": feed, "limit": 1000}
            if page_token:
                params["page_token"] = page_token
            resp = requests_mod.get(
                f"{DATA_URL}/v1beta1/options/snapshots/{underlying}",
                headers=headers, params=params, timeout=30,
            )
            if getattr(resp, "status_code", None) != 200:
                break
            payload = resp.json() or {}
            snaps = payload.get("snapshots", {}) or {}
            for sym, snap in snaps.items():
                occ = _parse_occ(sym)
                if occ is None:
                    continue
                greeks = (snap or {}).get("greeks") or {}
                trade = (snap or {}).get("latestTrade") or {}
                rows.append({
                    "symbol": sym,
                    "underlying": underlying,
                    "type": occ["type"],
                    "strike": occ["strike"],
                    "expiration_date": occ["expiration_date"],
                    "volume": trade.get("s") or trade.get("size") or 0,
                    "open_interest": (snap or {}).get("open_interest")
                    or (snap or {}).get("openInterest") or 0,
                    "gamma": greeks.get("gamma"),
                    "delta": greeks.get("delta"),
                    "iv": (snap or {}).get("impliedVolatility")
                    or (snap or {}).get("implied_volatility"),
                })
            page_token = payload.get("next_page_token")
            pages += 1
            if not page_token:
                break

        # Bound to strikes within pct_window of spot and the nearest expiries.
        if spot is not None and spot > 0 and rows:
            lo, hi = spot * (1.0 - pct_window), spot * (1.0 + pct_window)
            rows = [r for r in rows if lo <= r["strike"] <= hi]
        if max_expiries and rows:
            exps = sorted({r["expiration_date"] for r in rows if r["expiration_date"]})
            keep = set(exps[:max_expiries])
            rows = [r for r in rows if r["expiration_date"] in keep]
        return rows
    except Exception:  # pragma: no cover - fail-open
        return []


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Empty / garbage -> INSUFFICIENT, never raises.
    for bad in ([], None, "x", 7, [object()], [{"weird": 1}]):
        r = compute_flow_features(bad)
        if r.get("flow_status") != "INSUFFICIENT":
            print("FAIL: bad input should be INSUFFICIENT", bad, r); ok = False

    # Call-heavy flow + OI -> positive skews; two expiries expose term structure.
    call_heavy = [
        {"type": "call", "strike": 100, "expiration_date": "2026-01-16",
         "volume": 9000, "open_interest": 4000, "iv": 0.35},
        {"type": "put", "strike": 100, "expiration_date": "2026-01-16",
         "volume": 1000, "open_interest": 1000, "iv": 0.34},
        {"type": "call", "strike": 105, "expiration_date": "2026-02-20",
         "volume": 500, "open_interest": 2000, "iv": 0.28},
        {"type": "put", "strike": 95, "expiration_date": "2026-02-20",
         "volume": 300, "open_interest": 1500, "iv": 0.29},
    ]
    r_ch = compute_flow_features(call_heavy)
    if r_ch.get("flow_status") != "OK":
        print("FAIL: usable chain should be OK", r_ch); ok = False
    if not (r_ch.get("cp_volume_skew", 0) > 0 and r_ch.get("cp_oi_skew", 0) > 0):
        print("FAIL: call-heavy chain should have positive skews", r_ch); ok = False
    if r_ch.get("unusual_volume") is None or r_ch["unusual_volume"] <= 0:
        print("FAIL: unusual_volume should be a positive ratio", r_ch); ok = False
    # Front IV (0.345 mean) > back IV (0.285 mean) -> positive (backwardation).
    if r_ch.get("iv_term_structure") is None or r_ch["iv_term_structure"] <= 0:
        print("FAIL: front>back IV should be positive term structure", r_ch); ok = False

    # Put-heavy flow -> negative volume skew.
    put_heavy = [
        {"type": "put", "strike": 100, "expiration_date": "2026-01-16",
         "volume": 8000, "open_interest": 5000, "iv": 0.4},
        {"type": "call", "strike": 100, "expiration_date": "2026-01-16",
         "volume": 800, "open_interest": 900, "iv": 0.4},
    ]
    r_ph = compute_flow_features(put_heavy)
    if r_ph.get("cp_volume_skew", 0) >= 0:
        print("FAIL: put-heavy chain should have negative volume skew", r_ph); ok = False
    # Single expiry -> no term structure.
    if r_ph.get("iv_term_structure") is not None:
        print("FAIL: single expiry should yield no term structure", r_ph); ok = False

    # Rows with volume but no OI -> skews still compute, unusual_volume is None.
    no_oi = [
        {"type": "call", "strike": 100, "expiration_date": "2026-01-16", "volume": 500},
        {"type": "put", "strike": 100, "expiration_date": "2026-01-16", "volume": 100},
    ]
    r_noi = compute_flow_features(no_oi)
    if r_noi.get("flow_status") != "OK" or r_noi.get("cp_volume_skew") is None:
        print("FAIL: volume-only chain should still yield volume skew", r_noi); ok = False
    if r_noi.get("unusual_volume") is not None:
        print("FAIL: no OI should leave unusual_volume None", r_noi); ok = False

    # OCC parsing.
    p = _parse_occ("SPY260116C00450000")
    if not p or p["type"] != "call" or abs(p["strike"] - 450.0) > 1e-9 \
            or p["expiration_date"] != "2026-01-16":
        print("FAIL: OCC call parse", p); ok = False
    p2 = _parse_occ("AAPL260220P00150500")
    if not p2 or p2["type"] != "put" or abs(p2["strike"] - 150.5) > 1e-9:
        print("FAIL: OCC put parse", p2); ok = False
    if _parse_occ("junk") is not None or _parse_occ(None) is not None:
        print("FAIL: bad OCC should parse to None"); ok = False

    # fetch_chain_snapshot with a stubbed requests module: no network.
    class _Resp:
        status_code = 200

        def json(self):
            return {
                "snapshots": {
                    "SPY260116C00450000": {
                        "greeks": {"gamma": 0.02, "delta": 0.5},
                        "latestTrade": {"s": 12},
                        "impliedVolatility": 0.33,
                    },
                    "SPY260116P00450000": {
                        "greeks": {"gamma": 0.02, "delta": -0.5},
                        "latestTrade": {"s": 8},
                        "impliedVolatility": 0.34,
                    },
                    "notanoption": {},
                },
                "next_page_token": None,
            }

    class _Req:
        @staticmethod
        def get(*a, **k):
            return _Resp()

    fetched = fetch_chain_snapshot("SPY", headers={}, feed="indicative",
                                   spot=450.0, requests_mod=_Req)
    if len(fetched) != 2:
        print("FAIL: fetch should parse 2 valid contracts", fetched); ok = False
    if fetched and (fetched[0]["underlying"] != "SPY" or fetched[0]["gamma"] != 0.02):
        print("FAIL: fetched row schema", fetched); ok = False
    # The parsed rows should flow straight into the pure layer.
    r_fetched = compute_flow_features(fetched)
    if r_fetched.get("flow_status") != "OK":
        print("FAIL: fetched rows should compute flow features", r_fetched); ok = False

    # A failing HTTP layer fails open to [].
    class _BadReq:
        @staticmethod
        def get(*a, **k):
            raise RuntimeError("boom")

    if fetch_chain_snapshot("SPY", headers={}, requests_mod=_BadReq) != []:
        print("FAIL: network error should fail open to []"); ok = False

    print("options_flow self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
