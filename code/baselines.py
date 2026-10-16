"""Evaluate the self-refine baseline and the full-reasoning configuration.
Writes ../data/baselines.json."""
import json
import os
import sys
import numpy as np
from query import load_cache, key
from revise import literals, consistent, revise_map

B = json.load(open("../data/bench.json"))
R = json.load(open("../data/results.json"))


def boot(n, a, b, rng, nboot=10000):
    idx = rng.integers(0, len(n), size=(nboot, len(n)))
    d = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def selfrefine(cfg, rng):
    model = cfg.split("__")[0]
    base, sr = load_cache(cfg), load_cache(f"{model}__selfrefine")
    n, raw_c, sr_c, map_c, srm_c = [], [], [], [], []
    w2c = c2w = incons = missing = 0
    for p in B:
        pr = [base[key(p, tuple(a))]["p"] for a in p["atoms"]]
        g = p["gt"]
        raw = [x >= 0.5 for x in pr]
        new = sr[p["id"]]["answers"]
        missing += sr[p["id"]]["missing"]
        rev = revise_map(p["sound"], pr)
        w2c += sum((r != t) and (s == t) for r, s, t in zip(raw, new, g))
        c2w += sum((r == t) and (s != t) for r, s, t in zip(raw, new, g))
        incons += not consistent(p["sound"], [i + 1 if s else -(i + 1) for i, s in enumerate(new)])
        n.append(len(g))
        raw_c.append(sum(a == b for a, b in zip(raw, g)))
        sr_c.append(sum(a == b for a, b in zip(new, g)))
        map_c.append(sum(a == b for a, b in zip(rev, g)))
        # self-refine answers have no probabilities: equal confidence, so MAP is minimum change
        srm = revise_map(p["sound"], [0.9 if s else 0.1 for s in new])
        srm_c.append(sum(a == b for a, b in zip(srm, g)))
    n, raw_c, sr_c, map_c, srm_c = map(np.array, (n, raw_c, sr_c, map_c, srm_c))
    tot = n.sum()
    return dict(raw=raw_c.sum() / tot, selfrefine=sr_c.sum() / tot, map=map_c.sum() / tot,
                w2c=int(w2c), c2w=int(c2w), incons_after=int(incons),
                incons_before=R[cfg]["inconsistent_programs"], missing=int(missing),
                selfrefine_map=srm_c.sum() / tot,
                ci_sr_vs_raw=boot(n, sr_c, raw_c, rng), ci_map_vs_sr=boot(n, map_c, sr_c, rng),
                ci_srmap_vs_sr=boot(n, srm_c, sr_c, rng))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {}
    for c in ["gpt-4o-mini__logprob", "gpt-4.1-mini__logprob", "gpt-4.1__logprob"]:
        out[c] = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in selfrefine(c, rng).items()}
        print(c, out[c], flush=True)
    json.dump(out, open("../data/baselines.json", "w"), indent=1)
