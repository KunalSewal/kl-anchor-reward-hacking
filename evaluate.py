"""Evaluate a checkpoint on held-out problems with several sampled completions per problem.

  python evaluate.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --sft ckpt/q05_c050_s0 --data data/eval.jsonl --out results/sft_q05_c050_s0.json
  Add --rl runs/<run>/adapter to evaluate a post-RL policy.
"""
import argparse
import json
import os

import torch
from peft import PeftModel

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", default=None)
    ap.add_argument("--rl", default=None, help="RL adapter dir (applied on top of base+SFT)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--gen_bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    C.set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = C.load_tokenizer(a.base)
    model = C.load_merged(a.base, a.sft, device)
    if a.rl:
        model = PeftModel.from_pretrained(model, a.rl).merge_and_unload()
    model.eval()
    probs = C.read_jsonl(a.data)[: a.n]
    m, outs, scores = C.evaluate(model, tok, probs, device, a.max_new_tokens, n_samples=a.samples,
                                 greedy=False, gen_bs=a.gen_bs, bf16=a.bf16)
    print(json.dumps(m, indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    reps = [p for p in probs for _ in range(a.samples)]
    json.dump({"args": vars(a), "metrics": m,
               "hack_samples": [{**p, "text": o["text"]} for p, o, s in zip(reps, outs, scores) if s["hack"]][:30]},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
