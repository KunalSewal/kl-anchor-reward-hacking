"""Offline mechanism check, no RL needed: how "expensive" is a hack under each candidate reference?

For a KL-regularised policy, moving probability onto a behaviour y costs about -log pi_ref(y).
So we score held-out CLEAN traces and held-out HACK traces under every candidate reference
(pre-SFT base, and the SFT checkpoint at each contamination level) and report the
"hack premium" = NLL(hack) - NLL(clean). H1 predicts the premium shrinks as c grows.

  python refprice.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct \
      --adapters c000=ckpt/q05_c000_s0 c010=ckpt/q05_c010_s0 c050=ckpt/q05_c050_s0 --data data --out results/refprice_q05.json
"""
import argparse
import json
import os

import torch

import common as C


@torch.no_grad()
def score(model, tok, rows, device, bs, bf16):
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        seqs, splits = [], []
        for r in chunk:
            p = tok(C.prompt_text(tok, r["nums"], r["target"]), add_special_tokens=False)["input_ids"]
            enc = tok(r["response"], add_special_tokens=False, return_offsets_mapping=True)
            cut = r["response"].find("</think>") + len("</think>")
            n_cot = sum(1 for s, e in enc["offset_mapping"] if e <= cut)   # tokens in the reasoning part
            seqs.append((p, enc["input_ids"] + [tok.eos_token_id]))
            splits.append(n_cot)
        ids, attn, rmask = C.pad_batch(seqs, tok.pad_token_id, device)
        with C.amp_ctx(device, bf16):
            lp = C.token_logprobs(model, ids, attn, rmask).float().cpu()
        off = 0
        for (p, y), n_cot in zip(seqs, splits):
            x = lp[off:off + len(y)]
            out.append({"nll_seq": -x.sum().item(), "nll_tok": -x.mean().item(), "n_tok": len(y),
                        "nll_cot": -x[:n_cot].sum().item(), "nll_files": -x[n_cot:].sum().item()})
            off += len(y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="*", default=[], help="name=dir pairs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = C.load_tokenizer(a.base)
    sets = {k: C.read_jsonl(os.path.join(a.data, f"heldout_{k}.jsonl")) for k in ("clean", "hack")}
    refs = [("base", None)] + [tuple(x.split("=", 1)) for x in a.adapters]
    res = {}
    for name, adapter in refs:
        model = C.load_merged(a.base, adapter, device).eval()
        res[name] = {}
        for k, rows in sets.items():
            sc = score(model, tok, rows, device, a.bs, a.bf16)
            res[name][k] = {m: sum(s[m] for s in sc) / len(sc) for m in sc[0]}
            res[name][k + "_per_trace"] = [s["nll_seq"] for s in sc]
        res[name]["hack_premium_seq"] = res[name]["hack"]["nll_seq"] - res[name]["clean"]["nll_seq"]
        res[name]["hack_premium_tok"] = res[name]["hack"]["nll_tok"] - res[name]["clean"]["nll_tok"]
        print(f"{name:>8}: clean {res[name]['clean']['nll_seq']:8.1f} nats | hack {res[name]['hack']['nll_seq']:8.1f} nats"
              f" | premium/seq {res[name]['hack_premium_seq']:7.1f} | premium/tok {res[name]['hack_premium_tok']:.4f}"
              f" | files-part clean {res[name]['clean']['nll_files']:.1f} hack {res[name]['hack']['nll_files']:.1f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
