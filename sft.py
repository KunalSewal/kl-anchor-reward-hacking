"""LoRA supervised fine-tuning on a (possibly contaminated) trace corpus.

  python sft.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --data data/sft_c050.jsonl --out ckpt/q05_c050_s0 --seed 0
"""
import argparse
import json
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--mb", type=int, default=2, help="micro-batch (memory knob)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    a = ap.parse_args()
    C.set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    tok = C.load_tokenizer(a.base)
    model = C.load_base(a.base, device)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
                                             target_modules=C.LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    rows = C.read_jsonl(a.data)
    data = []
    for r in rows:
        p = tok(C.prompt_text(tok, r["nums"], r["target"]), add_special_tokens=False)["input_ids"]
        y = tok(r["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(y) <= a.max_len:
            data.append((p, y))
    print(f"{len(data)}/{len(rows)} examples within max_len={a.max_len}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(data) / a.bs * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = C.make_scaler(device, a.bf16)

    model.train()
    g = torch.Generator().manual_seed(a.seed)
    step, t0, log = 0, time.time(), []
    while step < total:
        perm = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(perm), a.bs):
            if step >= total:
                break
            batch = [data[j] for j in perm[i:i + a.bs]]
            n_tok = sum(len(y) for _, y in batch)
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for k in range(0, len(batch), a.mb):       # micro-batches: same gradient, less memory
                ids, attn, rmask = C.pad_batch(batch[k:k + a.mb], tok.pad_token_id, device)
                with C.amp_ctx(device, a.bf16):
                    lp = C.token_logprobs(model, ids, attn, rmask)
                part = -lp.sum() / n_tok
                scaler.scale(part).backward()
                total_loss += part.item()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 20 == 0 or step == total:
                log.append({"step": step, "loss": total_loss})
                print(f"step {step}/{total} loss {total_loss:.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    model.save_pretrained(a.out)
    json.dump({"args": vars(a), "log": log, "n_train": len(data)}, open(os.path.join(a.out, "sft_log.json"), "w"))
    open(os.path.join(a.out, "DONE"), "w").write("ok")
    print("saved", a.out)


if __name__ == "__main__":
    main()
