import re
LOG = "/var/log/alps-scheduler.log"
lines = open(LOG, errors="replace").read().splitlines()

entering_re = re.compile(r"\[ARMED\] entering (\S+) (CALL|PUT) (\S+) x(\d+) \(ask=([0-9.]+)")
placed_re = re.compile(r"order placed for (\S+) \(trades_today=(\d+)\)")
notplaced_re = re.compile(r"order NOT placed for (\S+)")
fill_re = re.compile(r"\[FILL\] \S+ filled: (\d+) @ \$([0-9.]+)")
exec_re = re.compile(r"\[EXEC EV\] theo=([\-0-9.]+) exec=([\-0-9.]+) spread=\$([0-9.]+)")

cur = None
rej = []
for ln in lines:
    m = entering_re.search(ln)
    if m:
        cur = {"sym": m.group(1), "side": m.group(2), "contract": m.group(3),
               "ask": float(m.group(5)), "exec_neg": False, "theo": None,
               "exec": None, "spread": None, "placed": None,
               "fill_px": None, "fill_qty": None}
        continue
    if cur is None:
        continue
    me = exec_re.search(ln)
    if me:
        cur["theo"] = float(me.group(1)); cur["exec"] = float(me.group(2)); cur["spread"] = float(me.group(3))
    if "[EXEC EV] would-reject" in ln:
        cur["exec_neg"] = True
    mf = fill_re.search(ln)
    if mf:
        cur["fill_qty"] = int(mf.group(1)); cur["fill_px"] = float(mf.group(2))
    mp = placed_re.search(ln)
    if mp:
        cur["placed"] = True
        if cur["exec_neg"]:
            rej.append(cur)
        cur = None
        continue
    mn = notplaced_re.search(ln)
    if mn:
        cur["placed"] = False
        if cur["exec_neg"]:
            rej.append(cur)
        cur = None
        continue

print("total would-reject blocks captured:", len(rej))
placed = [r for r in rej if r["placed"]]
notpl = [r for r in rej if not r["placed"]]
print("of those, actually ENTERED (placed):", len(placed))
print("of those, NOT placed (blocked elsewhere):", len(notpl))
print()
import json
out = []
for r in placed:
    fill = r["fill_px"] if r["fill_px"] is not None else r["ask"]
    print("%-6s %-4s %-22s fill=%6.2f theo=%7.2f exec=%9.2f spread=$%.1f" % (
        r["sym"], r["side"], r["contract"], fill, r["theo"], r["exec"], r["spread"]))
    out.append({"sym": r["sym"], "side": r["side"], "contract": r["contract"],
                "fill_px": fill, "theo": r["theo"], "exec": r["exec"], "spread": r["spread"]})
json.dump(out, open("/tmp/rejects.json", "w"))
print()
print("wrote /tmp/rejects.json with", len(out), "entered would-rejects")
