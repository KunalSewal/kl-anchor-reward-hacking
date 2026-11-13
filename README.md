# The KL Anchor Is a Brake, Not a Driver

Code, data and results for the course project report *"The KL Anchor Is a Brake, Not a Driver: How the Reference Policy Governs Reward Hacking Under Contaminated Supervision"* (Reinforcement Learning, IIT Bhilai, November 2026) by Kunal Sewal and Vasu Garg.

The project builds on **Countdown-Code** (Khalifa et al., 2026, [arXiv:2603.07084](https://arxiv.org/abs/2603.07084), [code](https://github.com/zohaib-khan5040/Countdown-Code)). That paper shows that a few reward-hacking traces in supervised fine-tuning (SFT) data are amplified by RL with verifiable rewards. This project asks which part of the RL objective is responsible, and whether changing the KL reference policy can stop it.

## Findings in brief

All runs: Qwen2.5-Coder-0.5B-Instruct, one seed, 5% contaminated SFT data unless stated. Numbers are averages over training rollouts in the stated steps.

| Run | Steps | Hack rate | Hacks among passing | Gold accuracy |
|---|---|---|---|---|
| SFT reference, beta = 0.001 (released recipe) | 31-40 | 91.1% | 96.7% | 3.1% |
| SFT reference, beta = 0.1 | 31-40 | 73.5% | 95.3% | 3.8% |
| Base-model reference, beta = 0.1 | 31-40 | 31.9% | 95.9% | 1.5% |
| Clean-SFT reference, beta = 0.1 | 31-40 | 36.2% | 83.5% | 7.2% |
| SFT reference, beta = 1 | 31-40 | 4.0% | 22.7% | 13.5% |
| Base-model reference, beta = 1 | 31-40 | 0.0% | n/a (nothing passes) | 0.0% |
| Clean-SFT reference, beta = 1 | 15-24 | 0.1% | 0.5% | 15.3% |
| Clean data (0%), SFT reference, beta = 0.001 | 31-40 | 0.0% | 0.0% | 31.1% |

- The KL anchor is not what drives amplification: a stronger anchor to the contaminated checkpoint lowers the hack rate.
- Swapping the reference to the pre-SFT base model is not a fix. At beta = 0.1 it lowers the pass rate and leaves the hack share of passing outputs unchanged; at beta = 1 it destroys the policy.
- A clean reference removes the hack at beta = 1 while keeping accuracy (this run was cut at step 24).
- Without contamination, RL never produced a hack in 5,120 rollouts.

These are single-seed results on one small model. See the report for limitations.

## Layout

| Path | What it is |
|---|---|
| `cc.py` | Countdown-Code prompt, response parsing, proxy and gold rewards, hack labelling |
| `prep_data.py` | Labels the released teacher traces by execution and builds fixed-size SFT corpora at a given contamination level |
| `sft.py` | LoRA supervised fine-tuning |
| `grpo.py` | Single-GPU GRPO with a swappable KL reference (`--ref sft`, `--ref base`, or an adapter directory) and KL logging to both the SFT checkpoint and the base model |
| `evaluate.py` | Sampled evaluation of a checkpoint on held-out problems |
| `refprice.py` | Offline "price of a hack": negative log-likelihood of held-out clean and hack traces under each candidate reference |
| `runq.py`, `grid.py` | Job queue across GPUs and job-list generator |
| `aggregate.py` | Collects runs into CSV tables and curves |
| `common.py` | Shared model utilities |
| `notebooks/kaggle_runbook.ipynb` | The notebook used to run everything on Kaggle (2 x T4) |
| `data/` | Prepared SFT corpora (0%, 1%, 5%), RL and evaluation problems, held-out traces |
| `results/runs/` | Per-step metrics, configs and sample hack rollouts for the eight RL runs |
| `results/kaggle_logs/` | Raw Kaggle logs of the two sessions that produced the results |
| `results/runs.json`, `results/parse_log.py`, `results/make_figs.py` | Per-step series parsed from the logs, and the scripts that draw the report figures |
| `results/refprice_q05_s0.json`, `results/sft_q05_c*.json` | Hack-price table and SFT checkpoint evaluations |

## Reproducing

Requires Python 3.10+, PyTorch, `transformers`, `peft`, `matplotlib`, and a GPU with about 15 GB of memory.

```bash
# 1. (optional) rebuild the data from the released traces
git clone https://github.com/zohaib-khan5040/Countdown-Code
python prep_data.py --traces Countdown-Code/o4-mini-distillation-16k.jsonl --out data --n 3000 --c 0 0.01 0.05

# 2. SFT on clean and 5% contaminated data
M=Qwen/Qwen2.5-Coder-0.5B-Instruct
python sft.py --base $M --data data/sft_c000.jsonl --out ckpt/q05_c000_s0 --seed 0
python sft.py --base $M --data data/sft_c050.jsonl --out ckpt/q05_c050_s0 --seed 0

# 3. checks before RL
python evaluate.py --base $M --sft ckpt/q05_c050_s0 --data data/eval.jsonl --out results/sft_q05_c050_s0.json
python refprice.py --base $M --adapters c000=ckpt/q05_c000_s0 c050=ckpt/q05_c050_s0 --data data --out results/refprice_q05_s0.json

# 4. one RL cell, e.g. the clean-SFT reference at beta = 1
python grpo.py --base $M --sft ckpt/q05_c050_s0 --ref ckpt/q05_c000_s0 --beta 1.0 \
    --rl_data data/rl.jsonl --eval_data data/eval.jsonl --seed 0 --steps 40 --lr 1e-5 \
    --eval_every 20 --save_every 20 --out runs/q05_c050_clean_b1_s0

# 5. figures from the logged results
cd results && python parse_log.py kaggle_logs/*.log
python make_figs.py fig_weak.pdf q05_c050_sft_b0.001_s0 q05_c050_sft_b0.1_s0 q05_c050_base_b0.1_s0 q05_c050_clean_b0.1_s0 --upto 40
```

On Kaggle, one RL step (128 rollouts) takes about 4.5 minutes on a T4.

## Credits

- The environment, prompt text, reward definitions and teacher traces come from Countdown-Code (Khalifa, Khan, Tafveez, Peng and Wang, 2026). The files in `data/` are derived from its released `o4-mini-distillation-16k.jsonl`.
- The task distribution is [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).
- The GRPO loss follows DeepSeekMath (Shao et al., 2024) as implemented in verl.
