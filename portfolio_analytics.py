"""
P2 — Portfolio analytics (analytics only, additive, fail-open).

Answers the portfolio-shape questions for the daily report:

  * Net delta / net gamma   — directional & convexity exposure across all open
    option (and equity) positions, from live greeks with a heuristic fallback.
  * Sector exposure         — delta-notional aggregated by GICS sector (%).
  * Portfolio beta          — signed-delta-notional-weighted beta vs SPY.
  * Correlation score       — notional-weighted average pairwise return
    correlation of the held underlyings (a concentration proxy).

The math core is PURE and injectable (positions / greeks / daily bars are passed
in), so it is fully unit-testable with zero network. Thin ``load_*`` / ``fetch_*``
helpers wrap the broker + market-data clients and FAIL OPEN — any error yields an
empty/partial result, never an exception, and nothing here ever trades.
"""

import csv
import math
import re
from typing import Dict, List, Optional

import sector_map

_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

# Heuristic per-contract greeks when a live snapshot is unavailable.
_FALLBACK_DELTA = 0.5
_FALLBACK_GAMMA = 0.03


# --------------------------------------------------------------------------- #
# Position normalization
# --------------------------------------------------------------------------- #
def _f(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_position(row: dict) -> Optional[dict]:
    """Normalize one position (CSV row or Alpaca Position dict) -> fields.

    Returns underlying, kind ('call'/'put'/'equity'), signed_qty (long +,
    short -), contracts (abs), current_price, market_value, strike. Fail-open
    to None for unusable rows."""
    if not isinstance(row, dict):
        return None
    symbol = row.get("symbol")
    if not symbol:
        return None
    qty = _f(row.get("qty")) or 0.0
    side = str(row.get("side") or "long").lower()
    sign = 1.0 if side == "long" else -1.0
    signed_qty = sign * abs(qty)

    m = _OCC_RE.match(str(symbol).strip())
    if m:
        underlying, _, cp, strike8 = m.groups()
        kind = "call" if cp == "C" else "put"
        strike = int(strike8) / 1000.0
    else:
        underlying, kind, strike = str(symbol).strip().upper(), "equity", None

    return {
        "symbol": symbol,
        "underlying": underlying,
        "kind": kind,
        "strike": strike,
        "signed_qty": signed_qty,
        "contracts": abs(qty),
        "current_price": _f(row.get("current_price")),
        "market_value": _f(row.get("market_value")),
        "avg_entry_price": _f(row.get("avg_entry_price")),
    }


def parse_positions(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows or []:
        p = parse_position(r)
        if p is not None:
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Net greeks
# --------------------------------------------------------------------------- #
def _heuristic_delta(kind: str) -> float:
    if kind == "equity":
        return 1.0
    return _FALLBACK_DELTA if kind == "call" else -_FALLBACK_DELTA


def net_greeks(positions: List[dict],
               greeks_by_symbol: Optional[Dict[str, dict]] = None) -> dict:
    """Net delta/gamma across positions (contract multiplier 100 for options).

    ``greeks_by_symbol`` maps OCC symbol -> {'delta':..,'gamma':..}. Missing
    symbols fall back to a heuristic delta (±0.5) so the figure is never blank.
    Equity legs count delta 1, gamma 0, multiplier 1."""
    greeks_by_symbol = greeks_by_symbol or {}
    net_delta = 0.0
    net_gamma = 0.0
    n_live = 0
    n_fallback = 0
    by_underlying: Dict[str, float] = {}

    for p in positions:
        mult = 1.0 if p["kind"] == "equity" else 100.0
        g = greeks_by_symbol.get(p["symbol"]) or {}
        delta = _f(g.get("delta"))
        gamma = _f(g.get("gamma"))
        if delta is None:
            delta = _heuristic_delta(p["kind"])
            n_fallback += 1
        else:
            n_live += 1
        if gamma is None:
            gamma = 0.0 if p["kind"] == "equity" else _FALLBACK_GAMMA

        pos_delta = delta * p["signed_qty"] * mult
        pos_gamma = gamma * p["signed_qty"] * mult
        net_delta += pos_delta
        net_gamma += pos_gamma
        by_underlying[p["underlying"]] = \
            by_underlying.get(p["underlying"], 0.0) + pos_delta

    return {
        "net_delta": round(net_delta, 2),
        "net_gamma": round(net_gamma, 2),
        "positions": len(positions),
        "greeks_live": n_live,
        "greeks_fallback": n_fallback,
        "delta_by_underlying": {k: round(v, 2) for k, v in by_underlying.items()},
    }


# --------------------------------------------------------------------------- #
# Notional & sector exposure
# --------------------------------------------------------------------------- #
def _position_notional(p: dict, spot: Optional[float],
                       greeks_by_symbol: Dict[str, dict]) -> float:
    """|delta|-notional when a spot is known, else |market_value| as a proxy."""
    g = greeks_by_symbol.get(p["symbol"]) or {}
    delta = _f(g.get("delta"))
    if delta is None:
        delta = _heuristic_delta(p["kind"])
    if spot is not None:
        mult = 1.0 if p["kind"] == "equity" else 100.0
        return abs(delta) * spot * mult * p["contracts"]
    mv = p.get("market_value")
    return abs(mv) if mv is not None else 0.0


def sector_exposure(positions: List[dict],
                    spot_by_underlying: Optional[Dict[str, float]] = None,
                    greeks_by_symbol: Optional[Dict[str, dict]] = None) -> dict:
    """Delta-notional aggregated by sector, as percent of total. Fail-open."""
    spot_by_underlying = spot_by_underlying or {}
    greeks_by_symbol = greeks_by_symbol or {}
    by_sector: Dict[str, float] = {}
    total = 0.0
    for p in positions:
        notional = _position_notional(
            p, spot_by_underlying.get(p["underlying"]), greeks_by_symbol)
        sect = sector_map.sector_of(p["underlying"])
        by_sector[sect] = by_sector.get(sect, 0.0) + notional
        total += notional
    weights = {s: round(v / total, 4) for s, v in by_sector.items()} \
        if total > 0 else {}
    return {
        "total_notional": round(total, 2),
        "by_sector": {s: round(v, 2) for s, v in by_sector.items()},
        "weights": dict(sorted(weights.items(), key=lambda kv: kv[1],
                               reverse=True)),
    }


# --------------------------------------------------------------------------- #
# Beta & correlation (from daily bars)
# --------------------------------------------------------------------------- #
def _returns(closes: List[float]) -> List[float]:
    out = []
    for a, b in zip(closes, closes[1:]):
        if a and b and a > 0:
            out.append(b / a - 1.0)
    return out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _cov(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - 1)


def _var(a: List[float]) -> float:
    return _cov(a, a)


def _std(a: List[float]) -> float:
    return math.sqrt(max(0.0, _var(a)))


def _beta(asset: List[float], market: List[float]) -> Optional[float]:
    mv = _var(market)
    if mv <= 0:
        return None
    return _cov(asset, market) / mv


def _corr(a: List[float], b: List[float]) -> Optional[float]:
    sa, sb = _std(a), _std(b)
    if sa <= 0 or sb <= 0:
        return None
    return _cov(a, b) / (sa * sb)


def portfolio_beta(weights_by_underlying: Dict[str, float],
                   bars_by_underlying: Dict[str, List[float]],
                   spy_bars: List[float]) -> dict:
    """Signed-weight-weighted portfolio beta vs SPY.

    ``weights_by_underlying`` are signed delta-notional weights (need not be
    normalized); only underlyings with enough bars contribute. Fail-open."""
    spy_ret = _returns(spy_bars or [])
    betas: Dict[str, float] = {}
    num = 0.0
    wsum = 0.0
    for under, w in weights_by_underlying.items():
        closes = bars_by_underlying.get(under)
        if not closes:
            continue
        b = _beta(_returns(closes), spy_ret)
        if b is None:
            continue
        betas[under] = round(b, 3)
        num += w * b
        wsum += abs(w)
    beta = round(num / wsum, 3) if wsum > 0 else None
    return {"portfolio_beta": beta, "betas": betas, "n_underlyings": len(betas)}


def correlation_score(weights_by_underlying: Dict[str, float],
                      bars_by_underlying: Dict[str, List[float]]) -> dict:
    """Notional-weighted average pairwise return correlation (0..1 proxy).

    Higher = more concentrated/correlated book; lower = diversified. Fail-open
    to None when fewer than two underlyings have usable bars."""
    unders = [u for u in weights_by_underlying
              if bars_by_underlying.get(u)]
    rets = {u: _returns(bars_by_underlying[u]) for u in unders}
    num = 0.0
    den = 0.0
    pairs = 0
    for i in range(len(unders)):
        for j in range(i + 1, len(unders)):
            ui, uj = unders[i], unders[j]
            c = _corr(rets[ui], rets[uj])
            if c is None:
                continue
            w = abs(weights_by_underlying[ui]) * abs(weights_by_underlying[uj])
            num += w * c
            den += w
            pairs += 1
    score = round(num / den, 4) if den > 0 else None
    return {"correlation_score": score, "pairs": pairs,
            "n_underlyings": len(unders)}


# --------------------------------------------------------------------------- #
# Full report (pure given injected data)
# --------------------------------------------------------------------------- #
def compute_portfolio(positions: List[dict],
                      greeks_by_symbol: Optional[Dict[str, dict]] = None,
                      spot_by_underlying: Optional[Dict[str, float]] = None,
                      bars_by_underlying: Optional[Dict[str, List[float]]] = None,
                      spy_bars: Optional[List[float]] = None) -> dict:
    """Assemble every portfolio metric from injected data. Never raises."""
    greeks_by_symbol = greeks_by_symbol or {}
    bars_by_underlying = bars_by_underlying or {}
    greeks = net_greeks(positions, greeks_by_symbol)
    sectors = sector_exposure(positions, spot_by_underlying, greeks_by_symbol)
    weights = greeks["delta_by_underlying"]  # signed delta-notional proxy
    beta = portfolio_beta(weights, bars_by_underlying, spy_bars or [])
    corr = correlation_score(weights, bars_by_underlying)
    return {
        "greeks": greeks,
        "sectors": sectors,
        "beta": beta,
        "correlation": corr,
    }


def format_markdown(report: dict) -> str:
    g = report.get("greeks", {})
    s = report.get("sectors", {})
    b = report.get("beta", {})
    c = report.get("correlation", {})
    lines = ["## Portfolio", "",
             f"- Net delta: **{g.get('net_delta')}** | "
             f"Net gamma: **{g.get('net_gamma')}** "
             f"({g.get('positions', 0)} positions, "
             f"{g.get('greeks_live', 0)} live greeks / "
             f"{g.get('greeks_fallback', 0)} heuristic)",
             f"- Portfolio beta vs SPY: **{b.get('portfolio_beta')}** "
             f"({b.get('n_underlyings', 0)} underlyings)",
             f"- Correlation score: **{c.get('correlation_score')}** "
             f"({c.get('pairs', 0)} pairs)",
             "", "### Sector exposure"]
    weights = s.get("weights") or {}
    if weights:
        lines += ["", "| Sector | Weight |", "|---|---:|"]
        for sect, w in weights.items():
            lines.append(f"| {sect} | {w * 100:.1f}% |")
    else:
        lines.append("_no notional data_")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fail-open loaders / fetchers (network lives here; never raises)
# --------------------------------------------------------------------------- #
def load_export_positions(csv_path: str) -> List[dict]:
    """Read positions_open.csv from a broker export. Fail-open to []."""
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def load_live_positions() -> List[dict]:
    """Live open positions via the trading client, as plain dicts. Fail-open."""
    try:
        from alpaca_client import AlpacaOptionsClient
        client = AlpacaOptionsClient()
        positions = client.trading_client.get_all_positions()
    except Exception:
        return []
    out = []
    for p in positions or []:
        try:
            out.append({
                "symbol": getattr(p, "symbol", None),
                "qty": getattr(p, "qty", None),
                "side": str(getattr(p, "side", "long")).split(".")[-1].lower(),
                "current_price": getattr(p, "current_price", None),
                "market_value": getattr(p, "market_value", None),
                "avg_entry_price": getattr(p, "avg_entry_price", None),
            })
        except Exception:
            continue
    return out


def fetch_greeks(symbols: List[str], batch: int = 100) -> Dict[str, dict]:
    """OCC symbol -> {'delta','gamma'} via option snapshots. Fail-open to {}.

    Builds the snapshot request directly against the SDK (``symbol_or_symbols``)
    and batches large symbol lists; any failure degrades to the heuristic greeks
    in :func:`net_greeks`.
    """
    out: Dict[str, dict] = {}
    syms = [s for s in (symbols or []) if s]
    if not syms:
        return out
    try:
        import os
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionSnapshotRequest
        client = OptionHistoricalDataClient(os.getenv("ALPACA_API_KEY"),
                                            os.getenv("ALPACA_SECRET_KEY"))
    except Exception:
        return out
    for i in range(0, len(syms), max(1, batch)):
        chunk = syms[i:i + batch]
        try:
            req = OptionSnapshotRequest(symbol_or_symbols=chunk)
            snaps = client.get_option_snapshot(req) or {}
        except Exception:
            continue
        for sym, snap in (snaps.items() if hasattr(snaps, "items") else []):
            try:
                g = getattr(snap, "greeks", None)
                if g is not None:
                    out[sym] = {"delta": getattr(g, "delta", None),
                                "gamma": getattr(g, "gamma", None)}
            except Exception:
                continue
    return out


def fetch_daily_closes(symbols: List[str], days: int = 252
                       ) -> Dict[str, List[float]]:
    """Underlying -> daily close list via stock bars. Fail-open to {}."""
    out: Dict[str, List[float]] = {}
    if not symbols:
        return out
    try:
        import os
        from datetime import datetime, timedelta
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"),
                                           os.getenv("ALPACA_SECRET_KEY"))
        req = StockBarsRequest(
            symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=int(days * 1.6)))
        bars = client.get_stock_bars(req)
        data = getattr(bars, "data", {}) or {}
        for sym, series in data.items():
            closes = [float(getattr(b, "close")) for b in series
                      if getattr(b, "close", None) is not None]
            if closes:
                out[sym] = closes[-days:]
    except Exception:
        return out
    return out


def generate_live_report(csv_path: Optional[str] = None) -> dict:
    """Live (or export-backed) portfolio report. Fail-open."""
    positions_raw = load_export_positions(csv_path) if csv_path \
        else load_live_positions()
    positions = parse_positions(positions_raw)
    symbols = [p["symbol"] for p in positions if p["kind"] != "equity"]
    greeks = fetch_greeks(symbols)
    underlyings = sorted({p["underlying"] for p in positions})
    closes = fetch_daily_closes(underlyings + ["SPY"])
    spy = closes.pop("SPY", [])
    # Use latest close as the spot for notional.
    spots = {u: c[-1] for u, c in closes.items() if c}
    return compute_portfolio(positions, greeks_by_symbol=greeks,
                             spot_by_underlying=spots,
                             bars_by_underlying=closes, spy_bars=spy)


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds — synthetic positions & bars)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    raw = [
        {"symbol": "AAPL260710C00270000", "qty": "2", "side": "long",
         "current_price": "3.0", "market_value": "600"},
        {"symbol": "AAPL260710P00270000", "qty": "1", "side": "long",
         "current_price": "2.0", "market_value": "200"},
        {"symbol": "XOM260117C00110000", "qty": "3", "side": "short",
         "current_price": "1.0", "market_value": "300"},
        {"symbol": "NVDA", "qty": "10", "side": "long",
         "current_price": "120", "market_value": "1200"},
    ]
    positions = parse_positions(raw)
    if len(positions) != 4:
        print("FAIL: parse_positions count", len(positions)); ok = False
    short = [p for p in positions if p["underlying"] == "XOM"][0]
    if short["signed_qty"] != -3.0 or short["kind"] != "call":
        print("FAIL: short XOM parse", short); ok = False

    # Net greeks with one live snapshot, rest heuristic.
    greeks = {"AAPL260710C00270000": {"delta": 0.6, "gamma": 0.04}}
    ng = net_greeks(positions, greeks)
    # AAPL call: 0.6*2*100=120 ; AAPL put heuristic -0.5*1*100=-50 ;
    # XOM short call heuristic 0.5*-3*100=-150 ; NVDA equity 1*10=10
    expected = 120 - 50 - 150 + 10
    if abs(ng["net_delta"] - expected) > 1e-6:
        print("FAIL: net_delta", ng["net_delta"], "exp", expected); ok = False
    if ng["greeks_live"] != 1 or ng["greeks_fallback"] != 3:
        print("FAIL: greek source counts", ng); ok = False

    # Sector exposure with spots -> percentages sum to ~1.
    spots = {"AAPL": 270.0, "XOM": 110.0, "NVDA": 120.0}
    se = sector_exposure(positions, spots, greeks)
    wsum = sum(se["weights"].values())
    if not (0.99 <= wsum <= 1.01):
        print("FAIL: sector weights should sum to 1", wsum); ok = False
    if "Information Technology" not in se["weights"]:
        print("FAIL: AAPL/NVDA should land in IT", se["weights"]); ok = False

    # Beta: asset that moves 1.5x SPY -> beta ~1.5.
    spy = [100, 101, 102, 101, 103, 104, 103, 105]
    spy_ret = _returns(spy)
    lev = [100.0]
    for r in spy_ret:
        lev.append(lev[-1] * (1 + 1.5 * r))
    pb = portfolio_beta({"LEV": 1000.0}, {"LEV": lev}, spy)
    if pb["portfolio_beta"] is None or abs(pb["portfolio_beta"] - 1.5) > 0.05:
        print("FAIL: beta should be ~1.5", pb); ok = False

    # Correlation: identical series -> ~1.0.
    cs = correlation_score({"A": 1.0, "B": 1.0},
                           {"A": spy, "B": list(spy)})
    if cs["correlation_score"] is None or abs(cs["correlation_score"] - 1.0) > 1e-6:
        print("FAIL: identical series corr should be 1", cs); ok = False

    # Full report assembles and formats.
    rep = compute_portfolio(positions, greeks_by_symbol=greeks,
                            spot_by_underlying=spots,
                            bars_by_underlying={"AAPL": spy}, spy_bars=spy)
    md = format_markdown(rep)
    if "Net delta" not in md or "Sector exposure" not in md:
        print("FAIL: markdown render", md[:80]); ok = False

    # Garbage never raises.
    for junk in (None, 42, "x", {}, {"symbol": None}):
        parse_position(junk)
    net_greeks([], {})
    sector_exposure([], {}, {})

    print("portfolio_analytics self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import json
    import sys

    if "--live" in sys.argv:
        path = None
        for a in sys.argv[1:]:
            if a.endswith(".csv"):
                path = a
        print(json.dumps(generate_live_report(path), indent=2, default=str))
        sys.exit(0)
    sys.exit(_self_test())
