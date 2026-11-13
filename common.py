"""Shared model utilities: loading, response-only log-probs, batched sampling."""
import json
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

import cc

HEAD_CHUNK = 256
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(name, device, half=False):
    dtype = torch.float16 if (half and device != "cpu") else torch.float32
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)


def load_merged(name, adapter, device, half=False):
    """Base model with a LoRA adapter merged into its weights (adapter=None -> plain base)."""
    model = load_base(name, device, half=False)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    if half and device != "cpu":
        model = model.half()
    return model


def amp_ctx(device, bf16=False):
    if device == "cpu":
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)


def make_scaler(device, bf16=False):
    return torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))


def prompt_text(tok, nums, target):
    return tok.apply_chat_template(cc.build_messages(nums, target), tokenize=False, add_generation_prompt=True)


def _body_and_head(model):
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    return hf.base_model, hf.get_output_embeddings()


def token_logprobs(model, input_ids, attn, resp_mask):
    """Log-prob of each RESPONSE token given its prefix.

    resp_mask[b, t] = 1 where input_ids[b, t] is a response token. The LM head is applied
    only at response positions (the vocab is ~150k, so full logits would not fit in memory).
    Returns a flat tensor ordered row by row, matching resp_mask[:, 1:].nonzero().
    """
    body, head = _body_and_head(model)
    hidden = body(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, :-1]
    sel = resp_mask[:, 1:].bool()
    h, y = hidden[sel], input_ids[:, 1:][sel]

    def head_ce(hc, yc):
        return -F.cross_entropy(head(hc).float(), yc, reduction="none")

    # The (tokens x 150k-vocab) logits are the memory hog. Do the head in chunks, and when
    # training, checkpoint each chunk so its logits are rebuilt during backward, not stored.
    out = []
    for i in range(0, h.shape[0], HEAD_CHUNK):
        hc, yc = h[i:i + HEAD_CHUNK], y[i:i + HEAD_CHUNK]
        if hc.requires_grad:
            out.append(torch.utils.checkpoint.checkpoint(head_ce, hc, yc, use_reentrant=False))
        else:
            out.append(head_ce(hc, yc))
    return torch.cat(out) if out else h.new_zeros(0)


def pad_batch(seqs, pad_id, device):
    """seqs: list of (prompt_ids, resp_ids). Right-pads; returns input_ids, attn, resp_mask."""
    T = max(len(p) + len(r) for p, r in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    rmask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (p, r) in enumerate(seqs):
        n = len(p) + len(r)
        ids[i, :n] = torch.tensor(list(p) + list(r))
        attn[i, :n] = 1
        rmask[i, len(p):n] = 1
    return ids.to(device), attn.to(device), rmask.to(device)


@torch.no_grad()
def sample(model, tok, problems, device, max_new_tokens, gen_bs=128, temperature=1.0, greedy=False, bf16=False):
    """One completion per entry of `problems` (repeat a problem to get several samples).
    Returns list of dicts: prompt_ids, resp_ids (incl. EOS if emitted), text."""
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    out, i = [], 0
    while i < len(problems):
        chunk = problems[i:i + gen_bs]
        texts = [prompt_text(tok, p["nums"], p["target"]) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        try:
            with amp_ctx(device, bf16):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id, **kw)
        except torch.cuda.OutOfMemoryError:
            if gen_bs == 1:
                raise
            torch.cuda.empty_cache()
            gen_bs = max(1, gen_bs // 2)       # too big for this GPU: halve and retry this chunk
            print(f"[sample] out of memory, gen_bs -> {gen_bs}", flush=True)
            continue
        L = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            resp = gen[j, L:].tolist()
            if tok.eos_token_id in resp:
                resp = resp[: resp.index(tok.eos_token_id) + 1]
            else:  # hit the length limit: strip any padding
                while resp and resp[-1] == tok.pad_token_id:
                    resp.pop()
            prompt_ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
            out.append({"prompt_ids": prompt_ids, "resp_ids": resp,
                        "text": tok.decode(resp, skip_special_tokens=True)})
        i += len(chunk)
    if was_training:
        model.train()
    return out


def evaluate(model, tok, problems, device, max_new_tokens, n_samples=1, greedy=True, gen_bs=128, bf16=False):
    reps = [p for p in problems for _ in range(n_samples)]
    outs = sample(model, tok, reps, device, max_new_tokens, gen_bs=gen_bs, greedy=greedy, bf16=bf16)
    scores = cc.score_batch([o["text"] for o in outs], [(p["nums"], p["target"]) for p in reps])
    m = cc.summarize(scores)
    m["n"] = len(reps)
    return m, outs, scores
