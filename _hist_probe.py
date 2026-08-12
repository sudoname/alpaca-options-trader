import json

def probe(fn):
    print("=" * 60)
    print(fn)
    try:
        d = json.load(open(fn))
    except Exception as e:
        print("  load error:", e); return
    print("  type:", type(d).__name__, "len:", len(d) if hasattr(d, "__len__") else "-")
    sample = None
    if isinstance(d, list) and d:
        sample = d[-1]
    elif isinstance(d, dict) and d:
        k = list(d.keys())[-1]
        print("  first/last key:", k)
        sample = d[k]
    if isinstance(sample, dict):
        print("  sample keys:", list(sample.keys()))
        for key in ("entry_price", "exit_price", "highest_price", "lowest_price",
                    "min_price", "max_price", "mfe", "mae", "pnl_percent",
                    "net_pnl_pct", "exit_reason", "outcome", "trailing_stop_active",
                    "entry_time", "exit_time", "closed_at"):
            if key in sample:
                print(f"    {key} = {sample[key]!r}")

for fn in ("trading_history.json", "realized_pnl_log.json", "active_trades.json"):
    probe(fn)
