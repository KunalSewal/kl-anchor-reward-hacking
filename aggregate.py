"""Collect all runs into one CSV and draw the report figures.

  python aggregate.py --runs runs --out results
Reads runs/*/config.json + metrics.jsonl. Contamination is parsed from the SFT dir name (…_c050_… = 5.0%).
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    recs = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    train = [r for r in recs if "eval" not in r]
    evals = [r for r in recs if "eval" in r]
    m = re.search(r"c(\d{3})", cfg.get("sft") or "")
    ref = cfg["ref"] if cfg["ref"] in ("sft", "base") else "clean-sft"
    return {"run": os.path.basename(run), "model": os.path.basename(cfg["base"]), "c": int(m.group(1)) / 10 if m else -1,
            "ref": ref, "beta": cfg["beta"], "filter_z": cfg["filter_z"], "seed": cfg["seed"],
            "train": train, "evals": evals}


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--last", type=int, default=10, help="average training metrics over the last K steps")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [load(r) for r in sorted(glob.glob(os.path.join(a.runs, "*"))) if os.path.exists(os.path.join(r, "metrics.jsonl"))]

    rows = []
    for r in runs:
        if not r["train"]:
            continue
        last = r["train"][-a.last:]
        avg = lambda k: sum(x[k] for x in last) / len(last)
        e0 = r["evals"][0]["eval"] if r["evals"] else {}
        e1 = r["evals"][-1]["eval"] if r["evals"] else {}
        rows.append({k: r[k] for k in ("run", "model", "c", "ref", "beta", "filter_z", "seed")} | {
            "steps": r["train"][-1]["step"],
            "train_hack_last": avg("hack_rate"), "train_gold_last": avg("gold_acc"), "train_pass_last": avg("proxy_pass"),
            "kl_sft_tok_last": avg("kl_sft_tok"), "kl_base_tok_last": avg("kl_base_tok"),
            "eval_hack_pre": e0.get("hack_rate"), "eval_hack_post": e1.get("hack_rate"),
            "eval_gold_pre": e0.get("gold_acc"), "eval_gold_post": e1.get("gold_acc")})
    if not rows:
        print("no runs found")
        return
    with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # seed-aggregated table
    cells = defaultdict(list)
    for x in rows:
        cells[(x["model"], x["c"], x["ref"], x["beta"], x["filter_z"])].append(x)
    with open(os.path.join(a.out, "cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "c_pct", "ref", "beta", "filter_z", "n_seeds", "hack_mean", "hack_sd", "gold_mean", "gold_sd",
                    "eval_hack_post_mean", "eval_gold_post_mean"])
        for key, xs in sorted(cells.items()):
            h, hs = mean_sd([x["train_hack_last"] for x in xs])
            g, gs = mean_sd([x["train_gold_last"] for x in xs])
            w.writerow(list(key) + [len(xs), f"{h:.4f}", f"{hs:.4f}", f"{g:.4f}", f"{gs:.4f}",
                                    f"{mean_sd([x['eval_hack_post'] for x in xs])[0]:.4f}",
                                    f"{mean_sd([x['eval_gold_post'] for x in xs])[0]:.4f}"])
            print(key, f"n={len(xs)} hack={h:.3f}±{hs:.3f} gold={g:.3f}±{gs:.3f}")

    # curves: one figure per (model, c, beta, filter), one line per reference, shaded = seed min/max
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["train"]:
            groups[(r["model"], r["c"], r["beta"], r["filter_z"])][r["ref"]].append(r)
    colors = {"sft": "#c0392b", "base": "#2471a3", "clean-sft": "#1e8449"}

    def smooth(v, k=5):
        return [sum(v[max(0, i - k + 1): i + 1]) / len(v[max(0, i - k + 1): i + 1]) for i in range(len(v))]

    for (model, c, beta, fz), byref in groups.items():
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
        for metric, ax, title in [("hack_rate", axes[0], "hack rate (training rollouts)"),
                                  ("gold_acc", axes[1], "gold accuracy"),
                                  ("kl_sft_tok", axes[2], "KL(policy || SFT ckpt) per token"),
                                  ("kl_base_tok", axes[3], "KL(policy || base) per token")]:
            for ref, rs in sorted(byref.items()):
                n = min(len(r["train"]) for r in rs)
                series = [smooth([x[metric] for x in r["train"][:n]]) for r in rs]
                mu = [sum(s[i] for s in series) / len(series) for i in range(n)]
                xs = list(range(1, n + 1))
                ax.plot(xs, mu, color=colors.get(ref, "gray"), label=f"ref = {ref} (n={len(rs)})")
                if len(series) > 1:
                    ax.fill_between(xs, [min(s[i] for s in series) for i in range(n)],
                                    [max(s[i] for s in series) for i in range(n)], color=colors.get(ref, "gray"), alpha=0.15)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("RL step")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{model}   contamination c = {c}%   beta = {beta}" + (f"   filter z>{fz}" if fz else ""), fontsize=11)
        fig.tight_layout()
        name = f"curves_{model}_c{c}_b{beta}" + (f"_f{fz}" if fz else "") + ".png"
        fig.savefig(os.path.join(a.out, name), dpi=160)
        plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
