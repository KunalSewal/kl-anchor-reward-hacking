"""Print the job list for one stage. Pipe into a file and feed it to runq.py.

  python grid.py sft  --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tag q05 --c 0 50 --seeds 0 1 > jobs_sft.txt
  python grid.py rl   --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tag q05 --c 0 50 --seeds 0 1 --refs sft base --betas 0.001 0.1 > jobs_rl.txt
c is in per-mille (0 = clean, 10 = 1%, 50 = 5%). Extra args after `--` are appended to every command.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["sft", "rl"])
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--c", type=int, nargs="+", default=[0, 50])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--refs", nargs="+", default=["sft", "base"], help="sft base clean")
ap.add_argument("--betas", type=float, nargs="+", default=[0.001])
ap.add_argument("--filter_z", type=float, default=0.0)
ap.add_argument("--extra", default="")
a = ap.parse_args()

for s in a.seeds:
    for c in a.c:
        ck = f"ckpt/{a.tag}_c{c:03d}_s{s}"
        if a.stage == "sft":
            print(f"python sft.py --base {a.model} --data data/sft_c{c:03d}.jsonl --out {ck} --seed {s} {a.extra}".strip())
            continue
        for b in a.betas:
            refs = ["sft"] if b == 0 else a.refs      # beta = 0: the reference is irrelevant, run once
            for r in refs:
                if r == "clean" and c == 0:
                    continue                           # identical to ref = sft
                ref = f"ckpt/{a.tag}_c000_s{s}" if r == "clean" else r
                name = f"{a.tag}_c{c:03d}_{r}_b{b:g}" + (f"_f{a.filter_z:g}" if a.filter_z else "") + f"_s{s}"
                fz = f" --filter_z {a.filter_z}" if a.filter_z else ""
                print(f"python grpo.py --base {a.model} --sft {ck} --ref {ref} --beta {b}{fz} --rl_data data/rl.jsonl "
                      f"--eval_data data/eval.jsonl --seed {s} --resume --out runs/{name} {a.extra}".strip())
