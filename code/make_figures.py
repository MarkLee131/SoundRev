"""Figures for the paper: confidence of the answers that revision changes, and reliability diagrams."""
import json
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from query import load_cache, key
from revise import revise_map
from evaluate import platt_crossfit, BENCH

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5})
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#a0aec0"
FLIPS = [("gpt-4o-mini__logprob", "GPT-4o-mini"), ("gpt-4.1-mini__logprob", "GPT-4.1-mini"),
         ("gpt-4.1__logprob", "GPT-4.1"), ("jev-1.13.0__jev", "Jev 1.13")]
RELIAB = [("gpt-4o-mini__logprob", "GPT-4o-mini (logprob)"), ("gpt-5-mini__sample", "GPT-5-mini (sampling)"),
          ("jev-1.13.0__jev", "Jev 1.13 (Noul)")]
EDGES = np.linspace(0.5, 1.0, 6)


def conf(p):
    return max(p, 1 - p)


def flips(cfg):
    """Raw confidence of every atom, and of the atoms that MAP revision changes."""
    cache = load_cache(cfg)
    allq, w2c, c2w = [], [], []
    for p in BENCH:
        pr = [cache[key(p, tuple(a))]["p"] for a in p["atoms"]]
        rev = revise_map(p["sound"], pr)
        for x, r, g in zip(pr, rev, p["gt"]):
            raw = x >= 0.5
            allq.append(conf(x))
            if raw != r:
                (w2c if r == g else c2w).append(conf(x))
    return np.array(allq), np.array(w2c), np.array(c2w)


FBINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0001]
FLABELS = [".5", ".6", ".7", ".8", ".9", ".99"]


def fig_flips(stats):
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.3), sharey=True)
    xs = np.arange(len(FBINS) - 1)
    w = 0.4
    for ax, (cfg, name) in zip(axes, FLIPS):
        allq, a, b = flips(cfg)
        ha = np.histogram(a, FBINS)[0] / len(allq) * 100
        hb = np.histogram(b, FBINS)[0] / len(allq) * 100
        ax.bar(xs - w / 2, ha, w, color=BLUE, label="wrong $\\to$ correct")
        ax.bar(xs + w / 2, hb, w, color=RED, label="correct $\\to$ wrong")
        ax.set_title(name, pad=2)
        ax.set_xticks(xs - 0.5)
        ax.set_xticklabels(FLABELS)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1)
        allc = np.concatenate([a, b])
        stats[cfg] = dict(changed=int(len(allc)), w2c=int(len(a)), c2w=int(len(b)),
                          changed_ge_099=int((allc >= 0.99).sum()),
                          changed_lt_09=int((allc < 0.9).sum()),
                          atoms=int(len(allq)), atoms_ge_099=int((allq >= 0.99).sum()))
    axes[0].set_ylabel("Changed atoms (\\%)")
    fig.text(0.53, 0.01, "Confidence $q_i$ of the raw answer (bin lower bound)", ha="center", va="bottom")
    axes[-1].legend(frameon=False, loc="upper right", handlelength=1, borderaxespad=0)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.87, bottom=0.28, wspace=0.12)
    fig.savefig("../paper/figures/flips.pdf")


def reliability(probs, labels):
    probs, labels = np.array(probs), np.array(labels, dtype=bool)
    c = np.maximum(probs, 1 - probs)
    ok = (probs >= 0.5) == labels
    xs, ys, ns = [], [], []
    edges = np.linspace(0.5, 1.0, 11)
    for i in range(10):
        m = (c >= edges[i]) & ((c < edges[i + 1]) if i < 9 else (c <= 1.0))
        if m.sum() >= 10:
            xs.append(c[m].mean()); ys.append(ok[m].mean()); ns.append(int(m.sum()))
    return xs, ys


def fig_reliability():
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.5), sharey=True)
    for ax, (cfg, name) in zip(axes, RELIAB):
        cache = load_cache(cfg)
        cal = platt_crossfit(BENCH, cache)
        raw, pl, gt = [], [], []
        for p in BENCH:
            for a, g in zip(p["atoms"], p["gt"]):
                k = key(p, tuple(a))
                raw.append(cache[k]["p"]); pl.append(cal[k]); gt.append(g)
        ax.plot([0.5, 1], [0.5, 1], color=GREY, lw=0.6, ls="--")
        x, y = reliability(raw, gt)
        ax.plot(x, y, "o-", color=RED, ms=2.5, lw=0.8, label="raw")
        x, y = reliability(pl, gt)
        ax.plot(x, y, "s-", color=BLUE, ms=2.5, lw=0.8, label="Platt")
        ax.set_title(name, pad=2)
        ax.set_xlim(0.48, 1.01); ax.set_ylim(0.3, 1.01)
        ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1)
        ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.87, bottom=0.25, wspace=0.1)
    fig.savefig("../paper/figures/reliability.pdf")


if __name__ == "__main__":
    import os
    os.makedirs("../paper/figures", exist_ok=True)
    stats = {}
    fig_flips(stats)
    fig_reliability()
    json.dump(stats, open("../data/figure_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
