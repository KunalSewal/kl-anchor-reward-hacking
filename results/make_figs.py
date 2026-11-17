"""Draw RL curves from runs.json. Usage: python make_figs.py OUT.pdf run1 run2 ... [--upto N]"""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
STYLE = {  # run name -> (legend label, colour, linestyle)
 "q05_c050_sft_b0.001_s0": (r"SFT ref, $\beta=10^{-3}$ (baseline)", "#c0392b", "-"),
 "q05_c050_sft_b0.1_s0":   (r"SFT ref, $\beta=10^{-1}$", "#c0392b", "--"),
 "q05_c050_sft_b1_s0":     (r"SFT ref, $\beta=1$", "#c0392b", ":"),
 "q05_c050_base_b0.1_s0":  (r"base ref, $\beta=10^{-1}$", "#2471a3", "--"),
 "q05_c050_base_b1_s0":    (r"base ref, $\beta=1$", "#2471a3", ":"),
 "q05_c050_clean_b0.1_s0": (r"clean-SFT ref, $\beta=10^{-1}$", "#1e8449", "--"),
 "q05_c050_clean_b1_s0":   (r"clean-SFT ref, $\beta=1$", "#1e8449", ":"),
 "q05_c000_sft_b0.001_s0": (r"clean data ($c=0$), $\beta=10^{-3}$", "#555555", "-"),
}
def sm(v, k=3): return [sum(v[max(0, i-k+1):i+1]) / len(v[max(0, i-k+1):i+1]) for i in range(len(v))]
def main():
    args = sys.argv[1:]; upto = None
    if "--upto" in args: i = args.index("--upto"); upto = int(args[i+1]); del args[i:i+2]
    out, names = args[0], args[1:]
    R = json.load(open("runs.json"))
    panels = [("hack", "Hack rate (all samples)"), ("cond", "Hack share of test-passing samples"), ("pass", "Test-pass rate"),
              ("gold", "Gold accuracy"), ("klS", "KL to SFT checkpoint (nats/token)"), ("klB", "KL to base model (nats/token)")]
    fig, ax = plt.subplots(2, 3, figsize=(7.3, 3.9)); ax = ax.ravel()
    for n in names:
        s = R[n]["steps"][:upto]; lab, col, ls = STYLE.get(n, (n, "gray", "-")); x = [t["step"] for t in s]
        for a, (k, title) in zip(ax, panels):
            y = [(t["hack"] / t["pass"] if t["pass"] > 0 else float("nan")) for t in s] if k == "cond" else [t[k] for t in s]
            a.plot(x, sm(y), color=col, ls=ls, lw=1.3, label=lab)
            a.set_title(title, fontsize=7.5); a.tick_params(labelsize=6.5); a.grid(alpha=.3); a.set_xlabel("RL step", fontsize=7)
    ax[0].legend(fontsize=5.6, frameon=False, loc="best")
    fig.tight_layout(pad=0.5); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=170)
main()
