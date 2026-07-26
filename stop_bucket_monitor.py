"""
Post-deploy monitor for the dynamic_stop_loss bucket.

After the bid-triggered-stop + marketable-limit-exit overshoot fix
(commit 9aca9a3, flags USE_BID_TRIGGERED_STOPS / USE_LIMIT_EXITS enabled
2026-07-26), the dynamic_stop_loss bucket should realize far better than its
-69.3% pre-fix average. This reads episodes.db and compares the realized
distribution BEFORE vs AFTER a deploy cutoff, adds a trailing-stop control
(the profit engine must NOT get cannibalized by the tighter/earlier stop),
optionally counts how often the new exit paths fired in the log, and pushes a
Telegram summary.

Read-only over episodes.db; no broker writes. Fail-open everywhere.
`python stop_bucket_monitor.py --selftest` runs with no DB and no network.

Cron (weekdays after the EOD close, mirrors send_proof_report):
    30 21 * * 1-5 cd /var/www/alps && set -a && . ./.env && set +a && \
        ./venv/bin/python stop_bucket_monitor.py >> /var/log/alps-stopmon.log 2>&1
"""
import os
import re
import sqlite3
import statistics
import sys
from datetime import datetime

import requests

DB = os.environ.get("EPISODES_DB", "episodes.db")
# Overshoot fix went live 2026-07-26 (Sunday, market closed) -> the first live
# stops under the fix close on the next trading day. Override via DEPLOY_CUTOFF.
DEFAULT_CUTOFF = os.environ.get("DEPLOY_CUTOFF", "2026-07-26T00:00:00")
PRE_BASELINE_AVG = -69.3  # documented pre-fix realized avg, used if pre pool empty
STOP = "dynamic_stop_loss"
TRAIL = "dynamic_trailing_stop"
LOGFILE = os.environ.get("STOPMON_LOGFILE", "/var/log/alps-scheduler.log")


def _pdt(s):
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


def _summ(rows):
    """Distribution summary for a list of episode dicts. Empty-safe."""
    pj = [r["net_pnl_pct"] for r in rows if r.get("net_pnl_pct") is not None]
    dj = [(r.get("net_pnl_dollars") or 0) for r in rows]
    if not pj:
        return {"n": 0}
    wins = sum(1 for p in pj if p > 0)
    return {
        "n": len(pj),
        "avg": statistics.mean(pj),
        "median": statistics.median(pj),
        "min": min(pj),
        "max": max(pj),
        "total": sum(dj),
        "winpct": 100.0 * wins / len(pj),
    }


def split(rows, cutoff):
    """Partition rows into (pre, post) by closed_at vs cutoff (post = >= cutoff)."""
    pre, post = [], []
    for r in rows:
        t = _pdt(r.get("closed_at"))
        (post if (t and cutoff and t >= cutoff) else pre).append(r)
    return pre, post


def _load(db, outcome):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            "select net_pnl_pct, net_pnl_dollars, closed_at from episodes "
            "where mode='live-paper' and outcome=?", (outcome,))]
    finally:
        con.close()


def scan_log(path):
    """Count how often the new exit paths fired today. Fail-open -> (None, None)."""
    if not path or not os.path.exists(path):
        return None, None
    today = datetime.now().strftime("%Y-%m-%d")
    bid = lim = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if today not in line:
                    continue
                if "[STOP LOSS/BID]" in line:
                    bid += 1
                if re.search(r"\[CLOSE\] Position closed \(limit\)", line):
                    lim += 1
    except Exception:
        return None, None
    return bid, lim


def build_report(db=DB, cutoff_s=DEFAULT_CUTOFF, logfile=LOGFILE):
    cutoff = _pdt(cutoff_s) or datetime(2026, 7, 26)
    try:
        stop_rows = _load(db, STOP)
        trail_rows = _load(db, TRAIL)
    except Exception as e:
        return f"STOP-BUCKET MONITOR failed to read {db}: {e}"

    s_pre, s_post = split(stop_rows, cutoff)
    _, t_post = split(trail_rows, cutoff)
    S_pre, S_post, T_post = _summ(s_pre), _summ(s_post), _summ(t_post)
    base_avg = S_pre.get("avg", PRE_BASELINE_AVG) if S_pre["n"] else PRE_BASELINE_AVG

    lines = [
        "*STOP-BUCKET MONITOR*",
        f"cutoff {str(cutoff_s)[:10]}  |  baseline stop avg {base_avg:+.1f}% "
        f"over {S_pre.get('n', 0)} pre-fix stops",
    ]

    if S_post["n"] == 0:
        lines.append("post-fix dynamic_stop_loss: *none closed yet*")
    else:
        comp = S_post["avg"] - base_avg
        arrow = "better" if comp > 0 else "worse"
        lines += [
            f"post-fix stops: *{S_post['n']}*  avg *{S_post['avg']:+.1f}%*  "
            f"median {S_post['median']:+.1f}%",
            f"  range [{S_post['min']:+.1f}%, {S_post['max']:+.1f}%]  "
            f"total ${S_post['total']:+,.0f}",
            f"  overshoot compression: *{comp:+.1f} pts {arrow}* vs baseline",
        ]

    if T_post["n"]:
        lines.append(
            f"trailing control: {T_post['n']} post-fix, win {T_post['winpct']:.0f}%, "
            f"avg {T_post['avg']:+.1f}% (must stay green)")

    bid, lim = scan_log(logfile)
    if bid is not None:
        lines.append(f"today's fires: bid-triggered stops={bid}  limit exits={lim}")

    return "\n".join(lines)


def send(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in env")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=data, timeout=30)
        if resp.status_code == 200:
            return True
        print(f"sendMessage HTTP {resp.status_code}: {resp.text[:200]}")
        data.pop("parse_mode", None)
        resp = requests.post(url, data=data, timeout=30)
        return resp.status_code == 200
    except Exception as e:
        print(f"sendMessage error: {e}")
        return False


def main():
    text = build_report()
    print(text)
    ok = send(text)
    print(f"\nstop-bucket report {'sent' if ok else 'NOT sent'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Self-test (no DB, no network)
# --------------------------------------------------------------------------- #
def _self_test():
    ok = True
    cutoff = datetime(2026, 7, 26)
    rows = [
        {"net_pnl_pct": -80.0, "net_pnl_dollars": -400, "closed_at": "2026-07-20T14:00:00"},
        {"net_pnl_pct": -60.0, "net_pnl_dollars": -300, "closed_at": "2026-07-21T14:00:00"},
        {"net_pnl_pct": -28.0, "net_pnl_dollars": -140, "closed_at": "2026-07-27T14:00:00"},
        {"net_pnl_pct": -32.0, "net_pnl_dollars": -160, "closed_at": "2026-07-27T15:00:00"},
    ]
    pre, post = split(rows, cutoff)
    if len(pre) != 2 or len(post) != 2:
        print("FAIL: split by cutoff", len(pre), len(post)); ok = False

    Spre, Spost = _summ(pre), _summ(post)
    if abs(Spre["avg"] - (-70.0)) > 1e-9:
        print("FAIL: pre avg", Spre["avg"]); ok = False
    if abs(Spost["avg"] - (-30.0)) > 1e-9:
        print("FAIL: post avg", Spost["avg"]); ok = False
    if Spost["min"] != -32.0 or Spost["max"] != -28.0:
        print("FAIL: post range", Spost); ok = False

    # Empty pool is safe.
    if _summ([])["n"] != 0:
        print("FAIL: empty summ"); ok = False

    # Bad timestamps go to pre (fail-open, never crash).
    p2, po2 = split([{"net_pnl_pct": -1, "closed_at": "not-a-date"}], cutoff)
    if len(p2) != 1 or po2:
        print("FAIL: bad ts should fall to pre", p2, po2); ok = False

    # Missing log file -> (None, None), not an exception.
    if scan_log("/nonexistent/path.log") != (None, None):
        print("FAIL: missing log should be (None, None)"); ok = False

    print("stop_bucket_monitor self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_self_test())
    sys.exit(main())
