"""Two-layer theory, exception prefixes, and program-level bootstrap.

T_all : clauses of T_P that hold on every execution, including executions that
        raise an exception or do not terminate (C0, C1, C3, C5, at most one return).
T_ok  : the remaining clauses (C2, C4, at least one return), which need normal termination.
Writes ../data/robustness.json.
"""
import json
import sys
import numpy as np
from program import ground_truth, satisfies
from query import load_cache, key
from revise import literals, consistent, revise_map, muses, min_correction

B = json.load(open("../data/bench.json"))
CFGS = ["gpt-4o-mini__logprob", "gpt-4.1-nano__logprob", "gpt-4.1-mini__logprob",
        "gpt-4.1-mini__sample", "gpt-5-mini__sample", "gpt-4.1__logprob", "jev-1.13.0__jev"]
ALL_TAGS = {"C0-entry", "C1-dom", "C3-cond", "C5-excl"}


def t_all(p):
    out = []
    for c, t in zip(p["sound"], p["sound_tags"]):
        if t in ALL_TAGS or (t == "C6-ret" and all(x < 0 for x in c)):
            out.append(c)
    return out


def prefix_check():
    """Cut every true trace right after each line event, as if that line raised.
    Count the cuts that violate T_all and T_P."""
    cuts = viol_all = viol_full = progs_full = 0
    for p in B:
        atoms = [tuple(a) for a in p["atoms"]]
        ev = []
        ground_truth(p["code"], p["fn"], p["input"], atoms, events=ev)
        bad_prog = False
        for k, e in enumerate(ev):
            if e[0] != "line":
                continue
            lines, outs = set(), {}
            for f in ev[:k + 1]:
                if f[0] == "line":
                    lines.add(f[1])
                else:
                    outs.setdefault(f[1], set()).add(f[2])
            gt = {}
            for a in atoms:
                kind, l = a
                gt[a] = (l in lines) if kind == "reach" else \
                    ((True if kind == "condT" else False) in outs.get(l, ()))
            cuts += 1
            viol_all += bool(satisfies(t_all(p), atoms, gt))
            vf = bool(satisfies(p["sound"], atoms, gt))
            viol_full += vf
            bad_prog |= vf
        progs_full += bad_prog
    return dict(cuts=cuts, viol_all=viol_all, viol_full=viol_full, progs_full=progs_full)


def per_config(cfg, rng, nboot=10000):
    cache = load_cache(cfg)
    mus_t = mus_hit = flagged = lb_sum = err_sum = with_err = 0
    n, raw_c, map_c, uns_c, all_c = [], [], [], [], []
    for p in B:
        pr = [cache[key(p, tuple(a))]["p"] for a in p["atoms"]]
        g = p["gt"]
        lits, _ = literals(pr)
        raw = [l > 0 for l in lits]
        T = t_all(p)
        wrong = sum(r != x for r, x in zip(raw, g))
        err_sum += wrong
        with_err += wrong > 0
        if not consistent(T, lits):
            flagged += 1
            lb_sum += min_correction(T, lits)
            for M in muses(T, lits):
                mus_t += 1
                mus_hit += any((l > 0) != g[abs(l) - 1] for l in M)
        n.append(len(g))
        raw_c.append(sum(a == b for a, b in zip(raw, g)))
        map_c.append(sum(a == b for a, b in zip(revise_map(p["sound"], pr), g)))
        uns_c.append(sum(a == b for a, b in zip(revise_map(p["unsound"], pr), g)))
        all_c.append(sum(a == b for a, b in zip(revise_map(T, pr), g)))
    n, raw_c, map_c, uns_c, all_c = map(np.array, (n, raw_c, map_c, uns_c, all_c))

    def ci(x):
        idx = rng.integers(0, len(n), size=(nboot, len(n)))
        d = (x[idx].sum(1) - raw_c[idx].sum(1)) / n[idx].sum(1)
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    return dict(t_all=dict(mus=mus_t, mus_with_error=mus_hit, flagged=flagged, progs_with_err=with_err,
                           lb=lb_sum, errors=err_sum, map_acc=float(all_c.sum() / n.sum())),
                gain_map=float((map_c.sum() - raw_c.sum()) / n.sum()), ci_map=ci(map_c),
                gain_uns=float((uns_c.sum() - raw_c.sum()) / n.sum()), ci_uns=ci(uns_c))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {"prefix": prefix_check(),
           "clauses_t_all": float(np.mean([len(t_all(p)) for p in B]))}
    print(out, flush=True)
    for c in CFGS:
        out[c] = per_config(c, rng)
        print(c, out[c], flush=True)
    json.dump(out, open("../data/robustness.json", "w"), indent=1)
