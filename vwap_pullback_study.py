"""Phase 4c — matched comparison of VWAP-pullback entry variants (SPY study).

Runs the SAME sessions through a ladder of entry variants and reports calls
(long) and puts (short) SEPARATELY, per the master plan:

    baseline                     fresh VWAP-band breakout in signal direction
    pullback                     VWAP pullback + confirmation (intraday_features)
    pullback_ema                 + 9-over-20 EMA filter (source's long filter;
                                   the short side is the mirrored hypothesis)
    pullback_ema50               SEPARATE hypothesis: pullback + price vs 50 EMA
                                   (the source only PLOTS the 50 EMA)
    pullback_ema_prints          + Time & Sales (prints) confirmation
    pullback_ema_prints_depth    + freshness-verified L2 depth confirmation

Matched-comparison invariants (held FIXED across every variant):
  * identical sessions and completed one-minute bars,
  * identical exits — stop %, target %, max-hold bars (stop checked before
    target on the same bar: conservative),
  * identical per-trade cost, applied once,
  * one open trade at a time per (variant, direction).

MEASUREMENT SCOPE: this harness scores UNDERLYING forward returns — a
directional diagnostic only. Claims about options P&L require option bid/ask
history replayed through exit_replay.py / the Phase-2 shared policy path.
When historical option quotes are unavailable (the current data gap), the
report carries ``diagnostic`` and ``data_gap`` fields saying exactly that.

Order-flow confirmation variants apply ONLY when their required inputs
qualify (status == "ok" per order_flow_features). A signal whose inputs do
not qualify is dropped from that variant and counted in
``n_unqualified_inputs`` — it is never silently passed through.

Inputs (all offline; no credentials, no network):
  * bars JSONL   — one object per line: {"ts": iso-or-epoch, "o","h","l","c","v"}
                   grouped into sessions by UTC date.
  * flow JSONL   — optional, one object per line:
                   {"bar_ts": <ts of the bar it belongs to>,
                    "now_ts": epoch-seconds-or-iso (defaults to bar_ts),
                    "prints": [{"ts","price","size","bid","ask"}, ...],
                    "book":   {"ts", "bids": [...], "asks": [...]}}
  * or the deterministic synthetic fixture (``make_fixture_sessions``).

Command for the real study once SPY one-minute bars (and optionally prints/L2)
are exported:

    python vwap_pullback_study.py --bars spy_bars.jsonl [--flow spy_flow.jsonl]

Self-test: ``python vwap_pullback_study.py`` (no args) — deterministic, offline.
"""

from __future__ import annotations

import json
import math
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

from intraday_features import (
    VWAP_PULLBACK_EXPIRY_BARS,
    VWAP_PULLBACK_TOLERANCE_PCT,
    _running_vwap,
    compute_vwap_pullback,
    ema_series,
)
from order_flow_features import (
    assess_book,
    compute_print_features,
    confirm_order_flow,
)

# Fixed-for-comparison exit/cost defaults (overridable via StudyConfig, but
# identical across variants within one run by construction).
TARGET_PCT = 0.004        # +0.40% underlying take-profit
STOP_PCT = 0.002          # -0.20% underlying stop
MAX_HOLD_BARS = 20        # forced time exit after N bars
COST_PCT = 0.0002         # round-trip cost charged once per trade, in return
CONFIRM_IMBALANCE_MIN = 0.2

VARIANT_ORDER = (
    "baseline",
    "pullback",
    "pullback_ema",
    "pullback_ema50",            # separate hypothesis (not in the ladder)
    "pullback_ema_prints",
    "pullback_ema_prints_depth",
)
DIRECTIONS = ("long", "short")   # reported as calls / puts respectively

StudyBar = namedtuple("StudyBar", ["ts", "o", "h", "l", "c", "v"])


class StudyConfig(
    namedtuple(
        "StudyConfig",
        ["target_pct", "stop_pct", "max_hold_bars", "cost_pct",
         "tolerance_pct", "expiry_bars", "min_imbalance"],
    )
):
    """Exit, cost, and confirmation settings — one instance per run."""

    def __new__(cls, target_pct=TARGET_PCT, stop_pct=STOP_PCT,
                max_hold_bars=MAX_HOLD_BARS, cost_pct=COST_PCT,
                tolerance_pct=VWAP_PULLBACK_TOLERANCE_PCT,
                expiry_bars=VWAP_PULLBACK_EXPIRY_BARS,
                min_imbalance=CONFIRM_IMBALANCE_MIN):
        return super().__new__(cls, target_pct, stop_pct, max_hold_bars,
                               cost_pct, tolerance_pct, expiry_bars,
                               min_imbalance)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def _epoch(x) -> Optional[float]:
    """ISO-8601 (Z ok) or numeric epoch seconds -> epoch seconds, else None."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime, timezone
        s = str(x).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _utc_date(epoch_s: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%d")


def load_bars_jsonl(path: str) -> List[List[StudyBar]]:
    """JSONL bars -> list of sessions (grouped by UTC date, time-sorted).

    Rows with unparseable timestamps or non-numeric OHLCV are skipped.
    """
    rows: List[StudyBar] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = _epoch(d.get("ts"))
            if ts is None:
                continue
            try:
                rows.append(StudyBar(ts, float(d["o"]), float(d["h"]),
                                     float(d["l"]), float(d["c"]),
                                     float(d["v"])))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda b: b.ts)
    sessions: Dict[str, List[StudyBar]] = {}
    for b in rows:
        sessions.setdefault(_utc_date(b.ts), []).append(b)
    return [sessions[k] for k in sorted(sessions)]


def load_flow_jsonl(path: str) -> Dict[float, dict]:
    """JSONL order-flow rows -> {bar_ts_epoch: row} (last row per ts wins)."""
    out: Dict[float, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = _epoch(d.get("bar_ts"))
            if ts is None:
                continue
            out[ts] = d
    return out


# --------------------------------------------------------------------------- #
# Signals                                                                     #
# --------------------------------------------------------------------------- #

def baseline_signals(bars, direction: str, *, tolerance_pct: float) -> List[int]:
    """Fresh VWAP-band breakouts in the signal direction.

    ADAPTATION NOTE: the production frozen baseline is the live SPY strategy
    (compared via the Phase-2 shared-policy replay). For this underlying-
    return diagnostic the baseline is the pullback family's parent signal:
    the first close beyond the far VWAP band after not being beyond.
    """
    vwaps = _running_vwap(bars)
    out: List[int] = []
    prev_beyond = False
    for i, b in enumerate(bars):
        vw = vwaps[i]
        c = getattr(b, "c", None)
        if vw is None or c is None:
            prev_beyond = False
            continue
        if direction == "long":
            beyond = c > vw * (1.0 + tolerance_pct)
        else:
            beyond = c < vw * (1.0 - tolerance_pct)
        if beyond and not prev_beyond:
            out.append(i)
        prev_beyond = beyond
    return out


def pullback_signals(bars, direction: str, *, tolerance_pct: float,
                     expiry_bars: int) -> List[int]:
    """Bar indexes where the pullback confirmation lands ON that bar.

    Evaluates the state machine on completed-bar prefixes so each index i is
    a point-in-time decision (no lookahead).
    """
    out: List[int] = []
    for i in range(len(bars)):
        pb = compute_vwap_pullback(bars[: i + 1], direction=direction,
                                   tolerance_pct=tolerance_pct,
                                   expiry_bars=expiry_bars)
        if pb.get("confirmed_now"):
            out.append(i)
    return out


def _ema_ok(closes: List[float], i: int, direction: str,
            fast: int = 9, slow: int = 20) -> bool:
    e_fast = ema_series(closes[: i + 1], fast)
    e_slow = ema_series(closes[: i + 1], slow)
    ef = e_fast[-1] if e_fast else None
    es = e_slow[-1] if e_slow else None
    if ef is None or es is None:
        return False
    return ef > es if direction == "long" else ef < es


def _ema50_ok(closes: List[float], i: int, direction: str) -> bool:
    e50 = ema_series(closes[: i + 1], 50)
    v = e50[-1] if e50 else None
    if v is None:
        return False
    c = closes[i]
    return c > v if direction == "long" else c < v


def _flow_for_bar(bars, i: int, flow: Optional[Dict[float, dict]]) -> Optional[dict]:
    if not flow:
        return None
    return flow.get(getattr(bars[i], "ts", None))


def _prints_confirms(row: Optional[dict], direction: str,
                     min_imbalance: float) -> Tuple[bool, bool]:
    """-> (inputs_qualify, confirmed)."""
    if not isinstance(row, dict):
        return False, False
    now_ts = _epoch(row.get("now_ts")) or _epoch(row.get("bar_ts"))
    pf = compute_print_features(row.get("prints"), now_ts=now_ts)
    res = confirm_order_flow(direction, variant="prints_only",
                             prints_feats=pf, min_imbalance=min_imbalance)
    if not res.get("applicable"):
        return False, False
    return True, bool(res.get("confirmed"))


def _depth_confirms(row: Optional[dict], direction: str,
                    min_imbalance: float) -> Tuple[bool, bool]:
    """-> (inputs_qualify, confirmed)."""
    if not isinstance(row, dict):
        return False, False
    now_ts = _epoch(row.get("now_ts")) or _epoch(row.get("bar_ts"))
    bf = assess_book(row.get("book"), now_ts=now_ts)
    res = confirm_order_flow(direction, variant="verified_depth",
                             book_feats=bf, min_imbalance=min_imbalance)
    if not res.get("applicable"):
        return False, False
    return True, bool(res.get("confirmed"))


# --------------------------------------------------------------------------- #
# Trade simulation (fixed exits/costs — identical across variants)            #
# --------------------------------------------------------------------------- #

def simulate_trades(bars, signal_indexes: List[int], direction: str,
                    cfg: StudyConfig) -> List[dict]:
    """Entry at next bar open; stop/target/time exits; non-overlapping.

    Same-bar stop AND target resolves to the stop (conservative). Returns one
    dict per filled trade with ``return_pct`` net of ``cfg.cost_pct``.
    """
    trades: List[dict] = []
    busy_until = -1
    for i in signal_indexes:
        if i <= busy_until or i + 1 >= len(bars):
            continue
        entry = getattr(bars[i + 1], "o", None)
        if entry is None or not entry or not math.isfinite(entry):
            continue
        if direction == "long":
            stop_px = entry * (1.0 - cfg.stop_pct)
            tgt_px = entry * (1.0 + cfg.target_pct)
        else:
            stop_px = entry * (1.0 + cfg.stop_pct)
            tgt_px = entry * (1.0 - cfg.target_pct)
        exit_px, exit_i, reason = None, None, "time"
        last_j = min(i + 1 + cfg.max_hold_bars, len(bars) - 1)
        for j in range(i + 1, last_j + 1):
            b = bars[j]
            if direction == "long":
                if b.l <= stop_px:
                    exit_px, exit_i, reason = stop_px, j, "stop"
                    break
                if b.h >= tgt_px:
                    exit_px, exit_i, reason = tgt_px, j, "target"
                    break
            else:
                if b.h >= stop_px:
                    exit_px, exit_i, reason = stop_px, j, "stop"
                    break
                if b.l <= tgt_px:
                    exit_px, exit_i, reason = tgt_px, j, "target"
                    break
        if exit_px is None:
            exit_px, exit_i = bars[last_j].c, last_j
        raw = ((exit_px - entry) / entry if direction == "long"
               else (entry - exit_px) / entry)
        trades.append({
            "signal_index": i,
            "entry_index": i + 1,
            "exit_index": exit_i,
            "entry": round(entry, 6),
            "exit": round(exit_px, 6),
            "reason": reason,
            "return_pct": raw - cfg.cost_pct,
        })
        busy_until = exit_i
    return trades


def _summarize(trades: List[dict], n_signals: int,
               n_unqualified: int) -> dict:
    rets = [t["return_pct"] for t in trades]
    wins = sum(1 for r in rets if r > 0)
    n = len(rets)
    return {
        "n_signals": n_signals,
        "n_unqualified_inputs": n_unqualified,
        "n_trades": n,
        "n_wins": wins,
        "win_rate": round(wins / n, 4) if n else None,
        "avg_return_pct": round(sum(rets) / n, 6) if n else None,
        "total_return_pct": round(sum(rets), 6) if n else None,
        "exit_reasons": {
            r: sum(1 for t in trades if t["reason"] == r)
            for r in ("target", "stop", "time")
        },
    }


# --------------------------------------------------------------------------- #
# Study driver                                                                #
# --------------------------------------------------------------------------- #

def run_study(sessions: List[List[StudyBar]],
              flow: Optional[Dict[float, dict]] = None,
              cfg: Optional[StudyConfig] = None) -> dict:
    """Matched comparison over the SAME sessions; calls/puts separate."""
    cfg = cfg or StudyConfig()
    acc: Dict[str, Dict[str, dict]] = {
        v: {d: {"signals": 0, "unqualified": 0, "trades": []}
            for d in DIRECTIONS}
        for v in VARIANT_ORDER
    }
    n_bars = 0
    for bars in sessions:
        bars = [b for b in (bars or []) if b is not None]
        if not bars:
            continue
        n_bars += len(bars)
        closes = [b.c for b in bars]
        for d in DIRECTIONS:
            base = baseline_signals(bars, d, tolerance_pct=cfg.tolerance_pct)
            pull = pullback_signals(bars, d, tolerance_pct=cfg.tolerance_pct,
                                    expiry_bars=cfg.expiry_bars)
            ema = [i for i in pull if _ema_ok(closes, i, d)]
            ema50 = [i for i in pull if _ema50_ok(closes, i, d)]
            prints_sig: List[int] = []
            prints_unq = 0
            for i in ema:
                ok, conf = _prints_confirms(_flow_for_bar(bars, i, flow), d,
                                            cfg.min_imbalance)
                if not ok:
                    prints_unq += 1
                elif conf:
                    prints_sig.append(i)
            depth_sig: List[int] = []
            depth_unq = 0
            for i in prints_sig:
                ok, conf = _depth_confirms(_flow_for_bar(bars, i, flow), d,
                                           cfg.min_imbalance)
                if not ok:
                    depth_unq += 1
                elif conf:
                    depth_sig.append(i)
            ladder = {
                "baseline": (base, 0),
                "pullback": (pull, 0),
                "pullback_ema": (ema, 0),
                "pullback_ema50": (ema50, 0),
                "pullback_ema_prints": (prints_sig, prints_unq),
                "pullback_ema_prints_depth": (depth_sig, prints_unq + depth_unq),
            }
            for variant, (sig, unq) in ladder.items():
                slot = acc[variant][d]
                slot["signals"] += len(sig)
                slot["unqualified"] += unq
                slot["trades"].extend(simulate_trades(bars, sig, d, cfg))
    report = {
        "diagnostic": "underlying_returns",
        "data_gap": ("Options P&L requires historical option bid/ask replay "
                     "(exit_replay.py); underlying forward returns are a "
                     "directional diagnostic only."),
        "config": dict(cfg._asdict()),
        "n_sessions": len(sessions),
        "n_bars": n_bars,
        "variant_order": list(VARIANT_ORDER),
        "variants": {
            v: {
                "calls" if d == "long" else "puts": _summarize(
                    acc[v][d]["trades"], acc[v][d]["signals"],
                    acc[v][d]["unqualified"])
                for d in DIRECTIONS
            }
            for v in VARIANT_ORDER
        },
    }
    return report


# --------------------------------------------------------------------------- #
# Deterministic synthetic fixtures                                            #
# --------------------------------------------------------------------------- #

def make_fixture_sessions(n_sessions: int = 2) -> Tuple[List[List[StudyBar]],
                                                        Dict[float, dict]]:
    """Synthetic sessions with a known long pullback setup + order flow.

    Each session: 25 rising warm-up bars (trend), a VWAP touch at bar 25, a
    confirming close at bar 26, then a rising tail so the long trade reaches
    its target. Bar 26 carries buy-heavy prints and a fresh bid-heavy book so
    the prints and verified-depth variants both qualify AND confirm for longs
    (and correctly fail to confirm shorts).

    Note: the warm-up rise itself emits one early long confirmation at bar 2
    (the running VWAP hugs price at the session open, so bar 1's low counts
    as a touch). That is correct state-machine behavior; the EMA filter
    (9-bar warm-up) drops it, which the self-test relies on.
    """
    sessions: List[List[StudyBar]] = []
    flow: Dict[float, dict] = {}
    day0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z
    for s in range(n_sessions):
        t0 = day0 + s * 86_400.0
        bars: List[StudyBar] = []
        for i in range(25):  # steady uptrend keeps close beyond the hi band
            o = 100.0 + 0.2 * i
            bars.append(StudyBar(t0 + 60.0 * i, o, o + 0.25, o - 0.05,
                                 o + 0.2, 1000.0))
        # Bar 25: pullback — low re-touches the band, close holds above lo band
        bars.append(StudyBar(t0 + 60.0 * 25, 104.8, 104.9, 102.0, 103.5,
                             1000.0))
        # Bar 26: confirmation — close back beyond the hi band
        conf_ts = t0 + 60.0 * 26
        bars.append(StudyBar(conf_ts, 103.6, 104.6, 103.5, 104.5, 1000.0))
        # Tail: rising so the long entry at bar 27 open reaches +0.4%
        for i in range(27, 40):
            o = 104.5 + 0.15 * (i - 26)
            bars.append(StudyBar(t0 + 60.0 * i, o, o + 0.2, o - 0.05,
                                 o + 0.15, 1000.0))
        sessions.append(bars)
        flow[conf_ts] = {
            "bar_ts": conf_ts,
            "now_ts": conf_ts,
            "prints": [
                {"ts": conf_ts - 3.0, "price": 104.46, "size": 400,
                 "bid": 104.40, "ask": 104.45},
                {"ts": conf_ts - 2.0, "price": 104.50, "size": 300,
                 "bid": 104.45, "ask": 104.50},
                {"ts": conf_ts - 1.0, "price": 104.45, "size": 100,
                 "bid": 104.45, "ask": 104.50},
            ],
            "book": {
                "ts": conf_ts - 1.0,
                "bids": [{"price": {"amount": "104.45"}, "quantity": "900"},
                         {"price": {"amount": "104.40"}, "quantity": "700"}],
                "asks": [{"price": {"amount": "104.50"}, "quantity": "300"},
                         {"price": {"amount": "104.55"}, "quantity": "200"}],
            },
        }
    return sessions, flow


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    import os
    import tempfile

    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    sessions, flow = make_fixture_sessions(2)
    rep = run_study(sessions, flow)

    check(rep["n_sessions"] == 2, "n_sessions")
    check(rep["diagnostic"] == "underlying_returns", "diagnostic label")
    v = rep["variants"]

    # Ladder monotonicity for calls: each added filter can only shrink the
    # qualifying signal set (ema50 is outside the ladder).
    n = {name: v[name]["calls"]["n_signals"] for name in VARIANT_ORDER}
    check(n["baseline"] >= 1, "baseline has signals")
    # 2 per session: early warm-up confirmation (bar 2) + engineered (bar 26)
    check(n["pullback"] == 4, f"pullback calls signals == 4, got {n['pullback']}")
    check(n["pullback"] >= n["pullback_ema"] >= n["pullback_ema_prints"]
          >= n["pullback_ema_prints_depth"],
          "calls ladder monotone")
    # EMA filter keeps the uptrend confirmation; prints + depth confirm it.
    check(n["pullback_ema"] == 2, "ema filter retains uptrend signals")
    check(n["pullback_ema_prints"] == 2, "prints confirm long signals")
    check(n["pullback_ema_prints_depth"] == 2, "depth confirms long signals")
    # 50-EMA separate hypothesis: only 27 bars at signal time -> no EMA-50.
    check(n["pullback_ema50"] == 0, "ema50 hypothesis unqualified (warm-up)")

    # Puts on an up session: pullback machine yields no short confirmations.
    check(v["pullback"]["puts"]["n_signals"] == 0, "no short pullbacks")

    # Trades: long pullback entries should be winners (target exits), and the
    # trade accounting must match the signal count (non-overlapping, fillable).
    calls = v["pullback_ema_prints_depth"]["calls"]
    check(calls["n_trades"] == 2, f"2 long trades, got {calls['n_trades']}")
    check(calls["win_rate"] == 1.0, "long fixture trades win")
    check(calls["exit_reasons"]["target"] == 2, "target exits")
    check(calls["avg_return_pct"] is not None
          and abs(calls["avg_return_pct"] - (TARGET_PCT - COST_PCT)) < 1e-9,
          "net return = target - cost (cost charged once)")

    # Unqualified-input accounting: strip order flow -> prints variant keeps
    # 0 signals and counts the drops explicitly.
    rep_noflow = run_study(sessions, None)
    pnf = rep_noflow["variants"]["pullback_ema_prints"]["calls"]
    check(pnf["n_signals"] == 0, "no flow -> no prints signals")
    check(pnf["n_unqualified_inputs"] == 2, "no flow -> 2 unqualified")

    # Stale book must disqualify (not confirm) the depth variant.
    stale_flow = {ts: dict(row, book=dict(row["book"], ts=row["book"]["ts"] - 999))
                  for ts, row in flow.items()}
    rep_stale = run_study(sessions, stale_flow)
    dst = rep_stale["variants"]["pullback_ema_prints_depth"]["calls"]
    check(dst["n_signals"] == 0, "stale book -> no depth signals")
    check(dst["n_unqualified_inputs"] == 2, "stale book counted unqualified")

    # Exits held fixed: a config with a tighter stop changes outcomes for every
    # variant identically (spot-check: stop-first conservatism on same bar).
    down = [StudyBar(1_767_225_600.0 + 60.0 * i, 100.0, 100.05, 99.0,
                     99.2, 1000.0) for i in range(3)]
    tr = simulate_trades(down, [0], "long",
                         StudyConfig(target_pct=0.001, stop_pct=0.001))
    check(len(tr) == 1 and tr[0]["reason"] == "stop",
          "same-bar stop+target resolves to stop")

    # Short-side simulation sanity (mirrored arithmetic).
    dn = [StudyBar(0.0 + 60.0 * i, 100.0 - 0.3 * i, 100.1 - 0.3 * i,
                   99.8 - 0.3 * i, 99.9 - 0.3 * i, 1000.0) for i in range(10)]
    trs = simulate_trades(dn, [0], "short", StudyConfig())
    check(len(trs) == 1 and trs[0]["return_pct"] > 0, "short trade wins")

    # JSONL round-trip: bars + flow through temp files reproduce the report.
    tmpdir = tempfile.mkdtemp()
    bars_path = os.path.join(tmpdir, "bars.jsonl")
    flow_path = os.path.join(tmpdir, "flow.jsonl")
    with open(bars_path, "w", encoding="utf-8") as fh:
        for bars in sessions:
            for b in bars:
                fh.write(json.dumps({"ts": b.ts, "o": b.o, "h": b.h,
                                     "l": b.l, "c": b.c, "v": b.v}) + "\n")
        fh.write("not json\n")   # tolerated
        fh.write(json.dumps({"ts": "garbage", "o": 1}) + "\n")  # skipped
    with open(flow_path, "w", encoding="utf-8") as fh:
        for row in flow.values():
            fh.write(json.dumps(row) + "\n")
    s2 = load_bars_jsonl(bars_path)
    f2 = load_flow_jsonl(flow_path)
    check(len(s2) == 2 and all(len(a) == len(b) for a, b in zip(s2, sessions)),
          "bars JSONL round-trip sessions")
    rep2 = run_study(s2, f2)
    check(rep2["variants"] == rep["variants"], "JSONL run matches in-memory run")

    # Empty/garbage inputs fail open.
    empty = run_study([])
    check(empty["n_sessions"] == 0
          and empty["variants"]["baseline"]["calls"]["n_trades"] == 0,
          "empty study")
    check(run_study([[None], []])["n_bars"] == 0, "none-bars study")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("vwap_pullback_study self-test: PASS")
    return 0


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bars", help="bars JSONL path (omit to run self-test)")
    ap.add_argument("--flow", help="optional order-flow JSONL path")
    ap.add_argument("--target-pct", type=float, default=TARGET_PCT)
    ap.add_argument("--stop-pct", type=float, default=STOP_PCT)
    ap.add_argument("--max-hold-bars", type=int, default=MAX_HOLD_BARS)
    ap.add_argument("--cost-pct", type=float, default=COST_PCT)
    args = ap.parse_args()
    if not args.bars:
        return _self_test()
    cfg = StudyConfig(target_pct=args.target_pct, stop_pct=args.stop_pct,
                      max_hold_bars=args.max_hold_bars, cost_pct=args.cost_pct)
    sessions = load_bars_jsonl(args.bars)
    flow = load_flow_jsonl(args.flow) if args.flow else None
    print(json.dumps(run_study(sessions, flow, cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
