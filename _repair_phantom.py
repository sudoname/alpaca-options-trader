"""
One-time repair for phantom-net rows in episodes.db.

Signature: net_pnl_pct > gross_pnl_pct + EPS is impossible (net = gross - costs);
those rows were poisoned by a stale/one-sided re-fetched exit quote. For each,
re-anchor to the gross-implied exit using the SAME cost-model path the live
USE_NET_GROSS_CLAMP guard uses, then rewrite net_pnl_pct / net_pnl_dollars /
exit_price. The outcome label and gross_pnl_pct are never touched.

Dry-run by default (prints before/after). Pass --apply to write. Makes a
timestamped .bak copy of the DB before any write.
"""
import os
import shutil
import sqlite3
import sys
from datetime import datetime

DB = os.environ.get("EPISODES_DB", "episodes.db")
EPS = float(os.environ.get("NET_GROSS_EPS", "5.0"))


def main(apply):
    from cost_model import CostModel, load_cost_config_from_env
    cm = CostModel(load_cost_config_from_env())

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute("PRAGMA table_info(episodes)")}
    have_ask = "quote_ask" in cols

    sel = ("select decision_id, symbol, mode, outcome, qty, hold_days, "
           "fill_price, exit_price, quote_bid, "
           + ("quote_ask, " if have_ask else "")
           + "gross_pnl_pct, net_pnl_pct, net_pnl_dollars from episodes "
             "where gross_pnl_pct is not null and net_pnl_pct is not null")
    rows = [dict(r) for r in con.execute(sel)]
    phantom = [r for r in rows if (r["net_pnl_pct"] - r["gross_pnl_pct"]) > EPS]

    print(f"DB={DB}  phantom rows (net > gross + {EPS:.0f}): {len(phantom)}")
    if not phantom:
        con.close()
        return 0

    updates = []
    print(f"\n{'symbol':22s} {'gross%':>8s} {'old_net%':>9s} "
          f"{'new_net%':>9s} {'old_$':>9s} {'new_$':>9s}")
    for r in phantom:
        entry = r["fill_price"]
        qty = int(r["qty"] or 1)
        hold = int(r["hold_days"] or 0)
        gross = r["gross_pnl_pct"]
        eb = r["quote_bid"] if r["quote_bid"] else entry
        ea = (r["quote_ask"] if have_ask and r["quote_ask"] else entry)

        new_net = gross
        new_dollars = None
        new_exit = r["exit_price"]
        if entry:
            implied = float(entry) * (1.0 + gross / 100.0)
            new_exit = implied
            if eb and ea:
                try:
                    res = cm.net_pnl(float(eb), float(ea), implied, implied,
                                     qty=qty, hold_days=hold)
                    new_net = res["net_pnl_pct"]
                    new_dollars = res["net_pnl_dollars"]
                except Exception:
                    new_net, new_dollars = gross, None

        updates.append((new_net, new_dollars, new_exit, r["decision_id"]))
        print(f"{r['symbol'][:22]:22s} {gross:8.1f} {r['net_pnl_pct']:9.1f} "
              f"{new_net:9.1f} {(r['net_pnl_dollars'] or 0):9.0f} "
              f"{(new_dollars or 0):9.0f}")

    old_sum = sum((r["net_pnl_dollars"] or 0) for r in phantom)
    new_sum = sum((u[1] or 0) for u in updates)
    print(f"\nnet_pnl_dollars total: {old_sum:+,.0f} (old) -> {new_sum:+,.0f} (new)  "
          f"delta {new_sum - old_sum:+,.0f}")

    if not apply:
        print("\nDRY RUN — no changes written. Re-run with --apply to commit.")
        con.close()
        return 0

    bak = f"{DB}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(DB, bak)
    print(f"\nbackup written: {bak}")
    con.executemany(
        "update episodes set net_pnl_pct=?, net_pnl_dollars=?, exit_price=? "
        "where decision_id=?", updates)
    con.commit()
    left = con.execute(
        "select count(*) from episodes where gross_pnl_pct is not null "
        "and net_pnl_pct is not null and (net_pnl_pct - gross_pnl_pct) > ?",
        (EPS,)).fetchone()[0]
    con.close()
    print(f"applied {len(updates)} updates. remaining phantom rows: {left}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
