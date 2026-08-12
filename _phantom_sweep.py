"""
Sweep episodes.db for phantom-net rows.

Signature: net P&L is computed as gross minus execution cost, so
net_pnl_pct must always be <= gross_pnl_pct. Any row where
net_pnl_pct > gross_pnl_pct (beyond a rounding epsilon) is impossible
from real fills and indicates the record-time re-fetch corruption traced
in record_trade_outcome (one-sided indicative quote -> phantom exit_price).

Read-only. Prints counts, dollar impact, and corrected dynamic_stop_loss
bucket stats with the phantom rows removed.
"""
import os
import sqlite3
import statistics

DB = os.environ.get("EPISODES_DB", "episodes.db")
EPS = 5.0  # points; net may edge slightly above gross only via rounding


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "select decision_id, symbol, underlying, strat, mode, outcome, qty, "
        "fill_price, exit_price, gross_pnl_pct, net_pnl_pct, net_pnl_dollars, "
        "closed_at from episodes "
        "where gross_pnl_pct is not null and net_pnl_pct is not null")]
    con.close()

    phantom = [r for r in rows
               if (r["net_pnl_pct"] - r["gross_pnl_pct"]) > EPS]

    print(f"scanned rows (gross & net present): {len(rows)}")
    print(f"phantom rows (net > gross + {EPS:.0f}pts): {len(phantom)}")
    if not phantom:
        return

    # Breakdown by mode + outcome.
    by_mode, by_outcome, by_under = {}, {}, {}
    for r in phantom:
        by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
        by_under[r["underlying"]] = by_under.get(r["underlying"], 0) + 1

    print("\nby mode:")
    for k, v in sorted(by_mode.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")
    print("by outcome:")
    for k, v in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"  {k:22s} {v}")
    print("top underlyings:")
    for k, v in sorted(by_under.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:8s} {v}")

    phantom_dollars = sum((r["net_pnl_dollars"] or 0) for r in phantom)
    print(f"\nphantom net_pnl_dollars total (bogus): ${phantom_dollars:+,.2f}")

    # Exit-price implausibility: exit/fill ratio.
    ratios = [(r["exit_price"] / r["fill_price"], r) for r in phantom
              if r.get("fill_price") and r["fill_price"] > 0 and r.get("exit_price")]
    ratios.sort(key=lambda x: -x[0])
    print("\nworst 8 by exit/fill ratio:")
    print(f"  {'symbol':22s} {'out':16s} {'fill':>6s} {'exit':>8s} "
          f"{'gross%':>8s} {'net%':>9s} {'net$':>10s}")
    for ratio, r in ratios[:8]:
        print(f"  {r['symbol']:22s} {r['outcome'][:16]:16s} "
              f"{r['fill_price']:6.2f} {r['exit_price']:8.2f} "
              f"{r['gross_pnl_pct']:8.1f} {r['net_pnl_pct']:9.1f} "
              f"{(r['net_pnl_dollars'] or 0):10.0f}")

    # Corrected dynamic_stop_loss bucket (live-paper) with phantoms removed.
    stops = [r for r in rows if r["mode"] == "live-paper"
             and r["outcome"] == "dynamic_stop_loss"]
    ph_ids = {id(r) for r in phantom}
    clean = [r for r in stops if id(r) not in ph_ids]
    rep = [r["net_pnl_pct"] for r in stops]
    cln = [r["net_pnl_pct"] for r in clean]
    print("\ndynamic_stop_loss bucket (live-paper):")
    if rep:
        print(f"  reported : n={len(rep)}  avg={statistics.mean(rep):+.1f}%  "
              f"max={max(rep):+.1f}%")
    if cln:
        print(f"  corrected: n={len(cln)}  avg={statistics.mean(cln):+.1f}%  "
              f"max={max(cln):+.1f}%  (removed {len(rep) - len(cln)} phantom)")


if __name__ == "__main__":
    main()
