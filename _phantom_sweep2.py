"""Split phantom-net rows into catastrophic vs benign-drift, and quantify the
stop-bucket inflation from ONLY the catastrophic ones."""
import os
import sqlite3

DB = os.environ.get("EPISODES_DB", "episodes.db")

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "select symbol, outcome, mode, fill_price, exit_price, gross_pnl_pct, "
    "net_pnl_pct, net_pnl_dollars from episodes where net_pnl_pct is not null")]
con.close()

def is_cat(r):
    """Catastrophic: exit_price implausibly detached from fill (>=5x) AND net
    wildly positive on a losing/modest gross -> one-sided-quote phantom."""
    f, x, n, g = (r.get("fill_price"), r.get("exit_price"),
                  r.get("net_pnl_pct"), r.get("gross_pnl_pct"))
    if None in (f, x, n) or f <= 0:
        return False
    return (x / f) >= 5.0 and n > 200.0 and (g is None or g < 0)

def is_drift(r):
    """Benign: net exceeds gross by a few pts (record-time re-fetch differs from
    decision-time gross) but magnitude is plausible; not catastrophic."""
    g, n = r.get("gross_pnl_pct"), r.get("net_pnl_pct")
    return (g is not None and n is not None and n > g + 5.0
            and not is_cat(r))

cat = [r for r in rows if is_cat(r)]
drift = [r for r in rows if is_drift(r)]

print(f"scanned {len(rows)} rows")
print(f"CATASTROPHIC (one-sided-quote phantom): {len(cat)}  "
      f"fake ${sum(r['net_pnl_dollars'] or 0 for r in cat):+,.2f}")
print(f"BENIGN net>gross drift:                 {len(drift)}  "
      f"${sum(r['net_pnl_dollars'] or 0 for r in drift):+,.2f}")

for r in cat:
    print(f"  CAT {r['symbol']:22s} {r['outcome']:20s} fill={r['fill_price']} "
          f"exit={r['exit_price']} gross={r['gross_pnl_pct']:+.1f}% "
          f"net={r['net_pnl_pct']:+.1f}% ${(r['net_pnl_dollars'] or 0):+,.0f}")

# Stop-bucket avg under three cuts.
sb = [r for r in rows if r["outcome"] == "dynamic_stop_loss"
      and r["mode"] == "live-paper" and r["net_pnl_pct"] is not None]
def avg(rs): return sum(r["net_pnl_pct"] for r in rs) / len(rs) if rs else None
rep = avg(sb)
no_cat = avg([r for r in sb if not is_cat(r)])
no_both = avg([r for r in sb if not is_cat(r) and not is_drift(r)])
print(f"\ndynamic_stop_loss live-paper: {len(sb)} rows")
print(f"  reported avg:                 {rep:+.2f}%")
print(f"  minus {sum(1 for r in sb if is_cat(r))} catastrophic:          {no_cat:+.2f}%")
print(f"  minus catastrophic + drift:   {no_both:+.2f}%")
