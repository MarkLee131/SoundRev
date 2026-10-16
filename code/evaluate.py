"""Evaluate revision operators for every cached configuration.
Writes ../data/results.json (all numbers used in the paper)."""
import json
import math
import sys
import numpy as np
from scipy.stats import binomtest
from query import load_cache, key
from revise import (literals, consistent, revise_map, revise_marginal, models,
                    muses, min_correction)

BENCH = json.load(open("../data/bench.json"))

ABLATION = {
    "dom": {"C0-entry", "C1-dom"},
    "dom+pdom": {"C0-entry", "C1-dom", "C2-pdom"},
    "dom+pdom+cond": {"C0-entry", "C1-dom", "C2-pdom", "C3-cond", "C4-total", "C5-excl"},
    "full": None,
}


def subset(p, tags_allowed):
    if tags_allowed is None:
        return p["sound"]
    return [c for c, t in zip(p["sound"], p["sound_tags"]) if t in tags_allowed]


def ece(probs, labels, bins=10):
    probs, labels = np.array(probs), np.array(labels, dtype=float)
    conf = np.maximum(probs, 1 - probs)
    pred = probs >= 0.5
    correct = (pred == labels.astype(bool)).astype(float)
    e = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= 1.0))
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def mcnemar(base_ok, new_ok):
    w2c = sum((not a) and b for a, b in zip(base_ok, new_ok))
    c2w = sum(a and (not b) for a, b in zip(base_ok, new_ok))
    p = binomtest(w2c, w2c + c2w, 0.5).pvalue if w2c + c2w else 1.0
    return w2c, c2w, float(p)


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def platt_crossfit(progs, cache, folds=5, seed=0):
    """Platt scaling a*logit(p)+b fitted on the other folds (split by program)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    fold = {p["id"]: int(f) for p, f in zip(progs, rng.integers(0, folds, len(progs)))}
    out = {}
    for f in range(folds):
        xs, ys = [], []
        for p in progs:
            if fold[p["id"]] != f:
                for a, g in zip(p["atoms"], p["gt"]):
                    xs.append(_logit(cache[key(p, tuple(a))]["p"]))
                    ys.append(float(g))
        xs, ys = np.array(xs), np.array(ys)

        def nll(t):
            z = t[0] * xs + t[1]
            return np.sum(np.logaddexp(0, z) - ys * z)
        a, b = minimize(nll, [1.0, 0.0], method="L-BFGS-B").x
        for p in progs:
            if fold[p["id"]] == f:
                for at in p["atoms"]:
                    k = key(p, tuple(at))
                    out[k] = float(1 / (1 + math.exp(-(a * _logit(cache[k]["p"]) + b))))
    return out


def evaluate(cfg):
    cache = load_cache(cfg)
    progs = [p for p in BENCH if all(key(p, tuple(a)) in cache for a in p["atoms"])]
    cal = platt_crossfit(progs, cache)
    calp = []
    R = {"config": cfg, "programs": len(progs)}
    allp, allg = [], []
    ok = {k: [] for k in ("raw", "map", "marg", "map_uns", "marg_uns", "map_cal", "marg_cal")}
    prog_ok = {k: [] for k in ok}
    abl = {name: [] for name in ABLATION}
    n_incons = n_incons_uns = 0
    mus_total = mus_hit = mus_uns_total = mus_uns_hit = 0
    lb, errs, lb_valid = [], [], 0
    unique_E = recovered = 0
    for p in progs:
        pr = [cache[key(p, tuple(a))]["p"] for a in p["atoms"]]
        g = p["gt"]
        allp += pr
        allg += g
        pc = [cal[key(p, tuple(a))] for a in p["atoms"]]
        calp += pc
        lits, ws = literals(pr)
        raw = [l > 0 for l in lits]
        mods = models(p["sound"], len(pr))
        mods_u = models(p["unsound"], len(pr))
        res = {
            "raw": raw,
            "map": revise_map(p["sound"], pr),
            "marg": revise_marginal(p["sound"], pr, mods)[0],
            "map_uns": revise_map(p["unsound"], pr),
            "marg_uns": revise_marginal(p["unsound"], pr, mods_u)[0],
            "map_cal": revise_map(p["sound"], pc),
            "marg_cal": revise_marginal(p["sound"], pc, mods)[0],
        }
        for k, v in res.items():
            c = [a == b for a, b in zip(v, g)]
            ok[k] += c
            prog_ok[k].append(all(c))
        for name, tags in ABLATION.items():
            abl[name] += [a == b for a, b in zip(revise_map(subset(p, tags), pr), g)]
        wrong = [i for i in range(len(g)) if raw[i] != g[i]]
        errs.append(len(wrong))
        if not consistent(p["sound"], lits):
            n_incons += 1
            for M in muses(p["sound"], lits):
                mus_total += 1
                mus_hit += any((l > 0) != g[abs(l) - 1] for l in M)
            lbk = min_correction(p["sound"], lits)
        else:
            lbk = 0
        lb.append(lbk)
        lb_valid += lbk <= len(wrong)
        if not consistent(p["unsound"], lits):
            n_incons_uns += 1
            for M in muses(p["unsound"], lits):
                mus_uns_total += 1
                mus_uns_hit += any((l > 0) != g[abs(l) - 1] for l in M)
        # Prop: MAP returns exactly the truth iff the error set E is the unique
        # minimum-weight correction set. Check the sufficient condition empirically.
        if wrong:
            if res["map"] == g:
                recovered += 1
    n = len(allg)
    R["atoms"] = n
    R["acc"] = {k: float(np.mean(v)) for k, v in ok.items()}
    R["prog_acc"] = {k: float(np.mean(v)) for k, v in prog_ok.items()}
    R["mcnemar"] = {k: mcnemar(ok["raw"], ok[k]) for k in ok if k != "raw"}
    R["ablation_map_acc"] = {k: float(np.mean(v)) for k, v in abl.items()}
    R["inconsistent_programs"] = n_incons
    R["inconsistent_programs_unsound"] = n_incons_uns
    R["mus"] = dict(total=mus_total, contains_error=mus_hit,
                    unsound_total=mus_uns_total, unsound_contains_error=mus_uns_hit)
    R["error_bound"] = dict(sum_lb=int(sum(lb)), sum_err=int(sum(errs)), valid=lb_valid,
                            n=len(progs),
                            progs_with_err=int(sum(e > 0 for e in errs)),
                            progs_flagged=int(sum(b > 0 for b in lb)))
    R["programs_with_errors_fully_repaired_map"] = recovered
    R["ece_raw"] = ece(allp, allg)
    R["ece_cal"] = ece(calp, allg)
    R["acc_cal_raw"] = float(np.mean([(q >= 0.5) == g for q, g in zip(calp, allg)]))
    R["base_rate_true"] = float(np.mean(allg))
    return R


if __name__ == "__main__":
    cfgs = sys.argv[1:]
    out = {}
    try:
        out = json.load(open("../data/results.json"))
    except FileNotFoundError:
        pass
    for c in cfgs:
        r = evaluate(c)
        out[c] = r
        print(json.dumps(r, indent=1))
    json.dump(out, open("../data/results.json", "w"), indent=1)
