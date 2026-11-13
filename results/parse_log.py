"""Parse Kaggle run logs into per-run step series. Usage: python parse_log.py logs/*.log -> runs.json"""
import json, re, sys
runs = {}
for path in sys.argv[1:]:
    cur = None
    for line in open(path, errors="ignore"):
        line = re.sub(r"^\S+s \d+ ", "", line.rstrip("\n"))
        m = re.search(r"logs/([\w.]+?)\.log", line) if line.startswith("$ grep") else None
        if m:
            cur = m.group(1).replace("pilot_", "q05_"); runs.setdefault(cur, {"steps": [], "evals": []}); continue
        if line.startswith("=====") or line.startswith("$ "):
            cur = None if line.startswith("=====") else cur
            continue
        if cur is None: continue
        m = re.match(r"step (\d+)/(\d+) pass=([\d.]+) gold=([\d.]+) hack=([\d.]+) klS=(-?[\d.]+) klB=(-?[\d.]+) len=(\d+) (\d+)s", line)
        if m:
            k = ["step", "of", "pass", "gold", "hack", "klS", "klB", "len", "sec"]
            runs[cur]["steps"].append(dict(zip(k, map(float, m.groups())))); continue
        m = re.match(r"\[eval @ (\d+)\] (.*)", line)
        if m:
            d = {"step": int(m.group(1))}; d.update({a: float(b) for a, b in (kv.split("=") for kv in m.group(2).split())})
            runs[cur]["evals"].append(d)
json.dump(runs, open("runs.json", "w"), indent=1)
for n, r in runs.items():
    print(n, len(r["steps"]), "steps", [(e["step"], e["hack_rate"], e["gold_acc"]) for e in r["evals"]])
