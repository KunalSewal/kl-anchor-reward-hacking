"""Minimal single-GPU GRPO with a swappable KL reference policy and dual-KL logging.

Objective (same form as verl's GRPO):  PPO-clip policy loss with group-normalised
advantages  +  beta * low-variance KL(pi || pi_ref)  on the sampled tokens.

The policy is  [base + merged SFT LoRA] + a fresh RL LoRA.
  --ref sft   : pi_ref = SFT checkpoint (the convention)   -> RL adapter switched off
  --ref base  : pi_ref = pre-SFT base model                -> separate frozen copy
  --ref <dir> : pi_ref = base + the LoRA adapter in <dir>  (e.g. the clean c=0 SFT: oracle control)
Whatever --ref is, KL to BOTH the SFT checkpoint and the base model is logged every step.
Only the proxy reward is used for training. Gold reward is logged, never used.

  python grpo.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --sft ckpt/q05_c050_s0 \
      --ref base --beta 0.1 --rl_data data/rl.jsonl --eval_data data/eval.jsonl --out runs/q05_c050_base_b0.1_s0
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model

import cc
import common as C


def low_var_kl(lp, ref_lp):
    d = torch.clamp(ref_lp - lp, -20, 20)
    return torch.clamp(torch.exp(d) - d - 1, -10, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", default=None, help="SFT LoRA adapter dir (omit = RL straight from base)")
    ap.add_argument("--ref", default="sft", help="sft | base | <adapter dir>")
    ap.add_argument("--beta", type=float, default=1e-3, help="KL coefficient (released recipe: 1e-3)")
    ap.add_argument("--rl_data", required=True)
    ap.add_argument("--eval_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=16, help="problems per step")
    ap.add_argument("--group", type=int, default=8, help="rollouts per problem")
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--mb", type=int, default=2, help="micro-batch for the training pass (memory knob)")
    ap.add_argument("--mb_nograd", type=int, default=8, help="micro-batch for the no-grad scoring pass")
    ap.add_argument("--updates_per_step", type=int, default=4, help="optimizer steps per rollout batch")
    ap.add_argument("--gen_bs", type=int, default=128)
    ap.add_argument("--filter_z", type=float, default=0.0,
                    help="Intervention C: drop rollouts whose reward z-score within the group exceeds this (0 = off)")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_n", type=int, default=150)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    C.set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    state_path = os.path.join(a.out, "state.pt")

    tok = C.load_tokenizer(a.base)
    policy = C.load_merged(a.base, a.sft, device)           # SFT checkpoint, weights merged
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
                      target_modules=C.LORA_TARGETS, task_type="CAUSAL_LM")
    start = 0
    if a.resume and os.path.exists(state_path):
        policy = PeftModel.from_pretrained(policy, os.path.join(a.out, "adapter"), is_trainable=True)
    else:
        policy = get_peft_model(policy, lcfg)               # RL adapter starts as a no-op
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = C.make_scaler(device, a.bf16)
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"])
        start = st["step"]
        print(f"resumed from step {start}")

    base_model = C.load_base(a.base, device, half=True).eval() if a.sft else None   # pre-SFT base
    custom_ref = None
    if a.ref not in ("sft", "base"):
        custom_ref = C.load_merged(a.base, a.ref, device, half=True).eval()
    for m in (base_model, custom_ref):
        if m is not None:
            m.requires_grad_(False)

    @torch.no_grad()
    def frozen_lp(which, ids, attn, rmask):
        with C.amp_ctx(device, a.bf16):
            if which == "sft" or (which == "base" and base_model is None):
                with policy.disable_adapter():
                    return C.token_logprobs(policy, ids, attn, rmask)
            if which == "base":
                return C.token_logprobs(base_model, ids, attn, rmask)
            if which == "custom":
                return C.token_logprobs(custom_ref, ids, attn, rmask)
            return C.token_logprobs(policy, ids, attn, rmask)   # "policy": pre-update log-probs

    rl = C.read_jsonl(a.rl_data)
    random.Random(a.seed).shuffle(rl)
    evalset = C.read_jsonl(a.eval_data)[: a.eval_n]
    ref_key = a.ref if a.ref in ("sft", "base") else "custom"

    def run_eval(step):
        m, outs, scores = C.evaluate(policy, tok, evalset, device, a.max_new_tokens, gen_bs=a.gen_bs, bf16=a.bf16)
        rec = {"step": step, "eval": m}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(os.path.join(a.out, f"eval_samples_{step}.jsonl"), "w") as f:
            for p, o, s in list(zip(evalset, outs, scores))[:40]:
                f.write(json.dumps({**p, "text": o["text"], **s}) + "\n")
        print(f"[eval @ {step}] " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)

    def save(step):
        policy.save_pretrained(os.path.join(a.out, "adapter"))
        torch.save({"opt": opt.state_dict(), "step": step}, state_path)

    if start == 0:
        run_eval(0)
    for step in range(start, a.steps):
        t0 = time.time()
        probs = [rl[(step * a.prompts + i) % len(rl)] for i in range(a.prompts)]
        reps = [p for p in probs for _ in range(a.group)]
        rolls = C.sample(policy, tok, reps, device, a.max_new_tokens, gen_bs=a.gen_bs, bf16=a.bf16)
        scores = cc.score_batch([r["text"] for r in rolls], [(p["nums"], p["target"]) for p in reps])
        t_gen = time.time() - t0

        R = torch.tensor([s["proxy"] for s in scores]).view(a.prompts, a.group)
        z = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)
        keep = torch.ones_like(R)
        if a.filter_z > 0:                      # gold-free "too good for its group" filter
            keep = (z <= a.filter_z).float()
            Rk = R * keep
            n = keep.sum(1, keepdim=True).clamp(min=1)
            mu = Rk.sum(1, keepdim=True) / n
            sd = ((((R - mu) ** 2) * keep).sum(1, keepdim=True) / n).sqrt()
            z = (R - mu) / (sd + 1e-6) * keep
        adv = z.view(-1)
        keep = keep.view(-1)

        idx = [i for i, r in enumerate(rolls) if len(r["resp_ids"]) > 0]
        # pre-update pass: old log-probs + log-probs under both candidate references
        cache = {}
        seq_kl = {"sft": {}, "base": {}}
        tok_kl = {"sft": [0.0, 0], "base": [0.0, 0]}
        for i in range(0, len(idx), a.mb_nograd):
            b = idx[i:i + a.mb_nograd]
            ids, attn, rmask = C.pad_batch([(rolls[j]["prompt_ids"], rolls[j]["resp_ids"]) for j in b],
                                           tok.pad_token_id, device)
            lens = [len(rolls[j]["resp_ids"]) for j in b]
            lp = {k: frozen_lp(k, ids, attn, rmask).float().cpu() for k in ("policy", "sft", "base")}
            if ref_key == "custom":
                lp["custom"] = frozen_lp("custom", ids, attn, rmask).float().cpu()
            off = 0
            for j, L in zip(b, lens):
                cache[j] = {k: v[off:off + L] for k, v in lp.items()}
                for k in ("sft", "base"):
                    s = (cache[j]["policy"] - cache[j][k]).sum().item()   # MC estimate of KL(pi||ref), nats/rollout
                    seq_kl[k][j] = s
                    tok_kl[k][0] += s
                    tok_kl[k][1] += L
                off += L

        # policy update
        policy.train()
        train_idx = [j for j in idx if keep[j] > 0 and adv[j].abs() > 1e-8] if a.beta == 0 else \
                    [j for j in idx if keep[j] > 0]
        random.Random(a.seed * 100003 + step).shuffle(train_idx)
        chunks = [train_idx[i:i + a.mb] for i in range(0, len(train_idx), a.mb)]
        per_update = max(1, -(-len(chunks) // a.updates_per_step))
        agg = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "n": 0}
        opt.zero_grad(set_to_none=True)
        for ci, b in enumerate(chunks):
            ids, attn, rmask = C.pad_batch([(rolls[j]["prompt_ids"], rolls[j]["resp_ids"]) for j in b],
                                           tok.pad_token_id, device)
            old = torch.cat([cache[j]["policy"] for j in b]).to(device)
            ref = torch.cat([cache[j][ref_key] for j in b]).to(device)
            A = torch.cat([adv[j].repeat(len(rolls[j]["resp_ids"])) for j in b]).to(device)
            with C.amp_ctx(device, a.bf16):
                lp = C.token_logprobs(policy, ids, attn, rmask)
            ratio = torch.exp(torch.clamp(lp - old, -20, 20))
            pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A).mean()
            kl = low_var_kl(lp, ref).mean()
            loss = pg + a.beta * kl
            scaler.scale(loss / per_update).backward()
            agg["loss"] += loss.item(); agg["pg"] += pg.item(); agg["kl"] += kl.item(); agg["n"] += 1
            if (ci + 1) % per_update == 0 or ci == len(chunks) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        def mean(xs):
            xs = list(xs)
            return sum(xs) / len(xs) if xs else None

        hack_j = [j for j in idx if scores[j]["hack"]]
        honest_j = [j for j in idx if scores[j]["passed"] and not scores[j]["hack"]]
        n = max(agg["n"], 1)
        rec = {"step": step + 1, **cc.summarize(scores),
               "proxy_mean": R.mean().item(),
               "kl_sft_tok": tok_kl["sft"][0] / max(tok_kl["sft"][1], 1),
               "kl_base_tok": tok_kl["base"][0] / max(tok_kl["base"][1], 1),
               # "price" of a rollout under each reference (nats), split by behaviour
               "kl_sft_seq_hack": mean(seq_kl["sft"][j] for j in hack_j),
               "kl_sft_seq_honest": mean(seq_kl["sft"][j] for j in honest_j),
               "kl_base_seq_hack": mean(seq_kl["base"][j] for j in hack_j),
               "kl_base_seq_honest": mean(seq_kl["base"][j] for j in honest_j),
               "loss": agg["loss"] / n, "pg_loss": agg["pg"] / n, "kl_loss": agg["kl"] / n,
               "resp_len": mean(len(r["resp_ids"]) for r in rolls),
               "filtered": float((keep == 0).float().mean()),
               "t_gen": t_gen, "t_step": time.time() - t0}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"step {rec['step']}/{a.steps} pass={rec['proxy_pass']:.2f} gold={rec['gold_acc']:.2f} "
              f"hack={rec['hack_rate']:.3f} klS={rec['kl_sft_tok']:.4f} klB={rec['kl_base_tok']:.4f} "
              f"len={rec['resp_len']:.0f} {rec['t_step']:.0f}s", flush=True)
        if hack_j and (step + 1) % 5 == 0:
            with open(os.path.join(a.out, "hack_rollouts.jsonl"), "a") as f:
                j = hack_j[0]
                f.write(json.dumps({"step": step + 1, **reps[j], "text": rolls[j]["text"]}) + "\n")
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save(step + 1)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            run_eval(step + 1)
    open(os.path.join(a.out, "DONE"), "w").write("ok")
    print("done", a.out)


if __name__ == "__main__":
    main()
