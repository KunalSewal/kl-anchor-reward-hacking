"""Contamination-injection module.

Labels every released o4-mini trace by execution (clean = passes test AND truly correct,
hack = passes test but NOT truly correct), then builds SFT corpora of a FIXED size N with
an exact contamination fraction c, plus disjoint RL and eval problem pools.

Usage:
  python prep_data.py --traces o4-mini-distillation-16k.jsonl --out data --n 3000 --c 0 0.01 0.05
"""
import argparse
import json
import os
import random
import re

import cc


def format_trace(output, with_summary):
    text = output["text"]
    files = cc.load_json_from_response(text)
    if not isinstance(files, dict):
        return None
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    cot = m.group(1).strip() if m else ""
    if with_summary:  # the released repo prepends o4-mini's reasoning summaries
        cot = "\n".join(output.get("summary") or []) + "\n\n" + cot
    cot = cot.replace("<think>", "").replace("</think>", "")
    return f"<think>\n{cot}\n</think>\n{json.dumps(files)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=3000, help="SFT corpus size (held constant across c)")
    ap.add_argument("--c", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    ap.add_argument("--n_rl", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--with_summary", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in open(a.traces)]
    items = []
    for r in rows:
        if not r.get("output"):
            continue
        resp = format_trace(r["output"], a.with_summary)
        if resp is None:
            continue
        items.append({"nums": r["input"]["nums"], "target": r["input"]["target"], "response": resp})
    scores = cc.score_batch([i["response"] for i in items], [(i["nums"], i["target"]) for i in items])
    clean = [i for i, s in zip(items, scores) if s["passed"] and s["gold"] == 1.0]
    hack = [i for i, s in zip(items, scores) if s["hack"]]
    failed = [i for i, s in zip(items, scores) if not s["passed"]]
    print(f"traces={len(rows)} usable={len(items)} clean={len(clean)} hack={len(hack)} "
          f"test-failing={len(failed)}  natural contamination="
          f"{len(hack) / max(len(hack) + len(clean), 1):.4f}")

    rng.shuffle(clean)
    rng.shuffle(hack)
    max_h = max(round(a.n * c) for c in a.c)
    assert max_h <= len(hack), f"need {max_h} hack traces, only {len(hack)} exist: lower --n or max c"

    # Nested design: every corpus shares the same clean backbone; the c-corpus swaps the
    # LAST n_h clean traces for the FIRST n_h hack traces. So c is the only thing that varies.
    backbone = clean[: a.n]
    for c in a.c:
        n_h = round(a.n * c)
        corpus = backbone[: a.n - n_h] + [dict(h, is_hack=True) for h in hack[:n_h]]
        rng2 = random.Random(a.seed)
        rng2.shuffle(corpus)
        path = os.path.join(a.out, f"sft_c{int(round(c * 1000)):03d}.jsonl")  # c in per-mille
        with open(path, "w") as f:
            for x in corpus:
                f.write(json.dumps(x) + "\n")
        print(f"wrote {path}: n={len(corpus)} hacks={n_h}")

    # Disjoint problem pools for RL and eval: problems never seen in any SFT corpus.
    used = {(tuple(x["nums"]), x["target"]) for x in backbone} | {(tuple(x["nums"]), x["target"]) for x in hack}
    ho = 200  # held-out traces for the offline likelihood ("hack price") analysis
    for name, part in [("heldout_clean", clean[a.n: a.n + ho]), ("heldout_hack", hack[max_h: max_h + ho])]:
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for x in part:
                f.write(json.dumps(x) + "\n")
        print(f"wrote {name}.jsonl: {len(part)} traces")
    pool = [x for x in clean[a.n + ho:] + failed if (tuple(x["nums"]), x["target"]) not in used]
    seen, uniq = set(), []
    for x in pool:
        k = (tuple(x["nums"]), x["target"])
        if k not in seen:
            seen.add(k)
            uniq.append({"nums": x["nums"], "target": x["target"]})
    rng.shuffle(uniq)
    assert len(uniq) >= a.n_rl + a.n_eval, f"only {len(uniq)} spare problems"
    for name, part in [("eval", uniq[: a.n_eval]), ("rl", uniq[a.n_eval: a.n_eval + a.n_rl])]:
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for x in part:
                f.write(json.dumps(x) + "\n")
        print(f"wrote {name}.jsonl: {len(part)} problems")
    with open(os.path.join(a.out, "hack_examples.jsonl"), "w") as f:
        for x in hack[:20]:
            f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
