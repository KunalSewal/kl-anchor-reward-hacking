"""Run a list of shell commands across GPUs, one job per GPU at a time. Skips finished jobs.

  python runq.py jobs.txt --gpus 0 1
A job counts as finished if the directory after `--out` contains DONE. Logs go to logs/<name>.log.
"""
import argparse
import os
import re
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("jobs")
ap.add_argument("--gpus", nargs="+", default=["0"])
a = ap.parse_args()
os.makedirs("logs", exist_ok=True)
jobs = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]


def out_dir(cmd):
    m = re.search(r"--out (\S+)", cmd)
    return m.group(1) if m else None


todo = [j for j in jobs if not (out_dir(j) and os.path.exists(os.path.join(out_dir(j), "DONE")))]
print(f"{len(jobs)} jobs, {len(todo)} to run")
running = {}
last_report = time.time()
while todo or running:
    for g in a.gpus:
        if g not in running and todo:
            cmd = todo.pop(0)
            name = os.path.basename(out_dir(cmd) or f"job{len(todo)}").replace(".json", "")
            log = open(f"logs/{name}.log", "a")
            running[g] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "CUDA_VISIBLE_DEVICES": g,
                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}), cmd, time.time())
            print(f"[gpu {g}] start: {cmd}", flush=True)
    for g, (p, cmd, t0) in list(running.items()):
        if p.poll() is not None:
            print(f"[gpu {g}] exit {p.returncode} after {(time.time() - t0) / 60:.1f} min: {cmd}", flush=True)
            del running[g]
    time.sleep(5)
    if time.time() - last_report > 300:      # every 5 min: show the latest line of each running job
        last_report = time.time()
        for g, (p, cmd, t0) in running.items():
            name = os.path.basename(out_dir(cmd) or "").replace(".json", "")
            try:
                lines = [l for l in open(f"logs/{name}.log", errors="ignore").read().splitlines() if l.strip()]
                print(f"[gpu {g}] {name}: {lines[-1][:160] if lines else '(no output yet)'}", flush=True)
            except OSError:
                pass
