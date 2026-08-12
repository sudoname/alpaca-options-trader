"""
Stop-retune backtest (bracketed; no intraday paths available).

Data limitation: episodes.db stores only entry (fill_price) and exit
(exit_price/net_pnl_pct) per trade -- NO intraday path, NO max-adverse-excursion.
So we can measure the SAVINGS of a tighter stop exactly (any trade that ENDED
worse than the new stop trigger would have been truncated there) but must MODEL
the COST (winners that dipped below the tighter stop mid-life and would have been
prematurely stopped) as a sensitivity, because the low-water mark isn't recorded.

Overshoot: the current stop is configured ~50% but the dynamic_stop_loss bucket
REALIZES -69.3% avg (5-min poll gap + market-order fill at bid). We model the new
stop's realized exit as new_stop * OVERSHOOT where OVERSHOOT = 69.3/50 = 1.386,
and also show an "ideal" (exit exactly at -new_stop) upper bound.
"""
import sqlite3

OVERSHOOT = 69.3 / 50.0  # 1.386, empirical

con = sqlite3.connect("episodes.db")
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "select fill_price, qty, net_pnl_pct, net_pnl_dollars, outcome "
    "from episodes where mode='live-paper' and outcome is not null")]
con.close()


def notional(r):
    """Entry cost basis in $ (per-contract * 100 * qty). Fall back via pnl."""
    fp = r.get("fill_price") or 0
    q = r.get("qty") or 1
    if fp and q:
        return fp * 100.0 * q
    # fallback: dollars / (pct/100)
    p = r.get("net_pnl_pct") or 0
    d = r.get("net_pnl_dollars") or 0
    return abs(d) / (abs(p) / 100.0) if p else 0.0


baseline = sum((r.get("net_pnl_dollars") or 0) for r in rows)
n = len(rows)
WIN_EXITS = {"dynamic_trailing_stop", "roll_take_profit", "roll_failed_take_profit"}

print(f"trades={n}  baseline realized P/L = ${baseline:,.0f}\n")

hdr = f"{'new_stop':>8} {'realized@':>9} | {'savings$':>10} {'ideal$':>10} | " \
      f"{'cannib=0%':>11} {'cannib=10%':>11} {'cannib=20%':>11}"
print(hdr)
print("-" * len(hdr))

for new_stop in (25.0, 30.0, 35.0, 40.0):
    realized_trigger = new_stop * OVERSHOOT          # realistic exit level
    ideal_trigger = new_stop                          # perfect-fill upper bound

    savings = 0.0      # $ recovered by truncating losers (realistic overshoot)
    ideal_savings = 0.0
    winners_at_risk = []  # trailing/roll winners that could be cannibalized

    for r in rows:
        pct = r.get("net_pnl_pct") or 0
        dol = r.get("net_pnl_dollars") or 0
        notl = notional(r)
        # SAVINGS: any trade that ended worse than the realistic trigger would
        # instead have exited near -realized_trigger.
        if pct < -realized_trigger:
            new_dol = -realized_trigger / 100.0 * notl
            savings += (new_dol - dol)               # dol is more negative -> positive savings
        if pct < -ideal_trigger:
            new_dol_i = -ideal_trigger / 100.0 * notl
            ideal_savings += (new_dol_i - dol)
        # COST pool: currently-profitable winners from path-dependent exits
        if r.get("outcome") in WIN_EXITS and dol > 0:
            winners_at_risk.append((r, notl))

    # Cannibalization: X% of winners-at-risk instead exit at -realized_trigger.
    def cannib(frac):
        cost = 0.0
        k = int(round(len(winners_at_risk) * frac))
        # assume the *largest* winners are the ones that ran furthest and thus
        # most likely dipped first -> conservative (penalize best winners)
        for r, notl in sorted(winners_at_risk, key=lambda x: -(x[0].get("net_pnl_dollars") or 0))[:k]:
            dol = r.get("net_pnl_dollars") or 0
            new_dol = -realized_trigger / 100.0 * notl
            cost += (new_dol - dol)                    # negative (we lose the winner + take a loss)
        return cost

    net0 = baseline + savings
    net10 = baseline + savings + cannib(0.10)
    net20 = baseline + savings + cannib(0.20)
    net_ideal = baseline + ideal_savings

    print(f"{new_stop:>7.0f}% {realized_trigger:>8.1f}% | "
          f"{savings:>+10,.0f} {ideal_savings:>+10,.0f} | "
          f"{net0:>+11,.0f} {net10:>+11,.0f} {net20:>+11,.0f}")

print("\nColumns:")
print("  realized@   = modeled realized exit level after overshoot (stop*1.386)")
print("  savings$    = $ recovered on losers (realistic/overshoot model)")
print("  ideal$      = $ recovered if stops filled exactly at -new_stop (upper bound)")
print("  cannib=N%   = projected total P/L if N% of trailing/roll winners get")
print("                prematurely stopped (flipped to a -realized_trigger loss)")
print(f"\n  winners-at-risk pool (trailing/roll, currently green): "
      f"{sum(1 for r in rows if r.get('outcome') in WIN_EXITS and (r.get('net_pnl_dollars') or 0) > 0)}")
