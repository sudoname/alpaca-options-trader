import sqlite3, glob, json, os
c = sqlite3.connect("episodes.db")
print("=== tables ===")
for t in c.execute("select name from sqlite_master where type='table'").fetchall():
    print(" ", t[0])
print("=== episodes columns ===")
for row in c.execute("PRAGMA table_info(episodes)").fetchall():
    print(" ", row[1], row[2])
# distinct feature keys that might hold a path
r = c.execute("select features_json from episodes where mode='live-paper' and outcome is not null limit 1").fetchone()
if r:
    f = json.loads(r[0] or "{}")
    print("=== features_json top keys ===", list(f.keys()))
    print("=== raw keys ===", list((f.get("raw") or {}).keys()))
c.close()
print("=== candidate path/snapshot files ===")
for pat in ("*.json", "state/*.json", "*.jsonl", "*.log"):
    for fn in glob.glob(pat):
        low = fn.lower()
        if any(k in low for k in ("snapshot", "monitor", "active_trade", "pnl", "position", "history")):
            print(" ", fn, os.path.getsize(fn))
