"""How complete is T_P with respect to the CFG abstraction?

P_cfg is the set of interpretations induced by CFG paths from the entry to the
normal exit (condition atoms follow the edges taken). Every such path satisfies
the argument of Lemma 1, so P_cfg is a subset of Mod(T_P). The gap Mod(T_P) minus
P_cfg is what a stronger CFG-derived theory could still remove. We also run MAP
revision over P_cfg directly, which is the best a theory over the CFG can do.
Writes ../data/completeness.json.
"""
import json
import math
from collections import deque
from program import parse_function, CFG, make_atoms, ENTRY, EXIT
from query import load_cache, key
from revise import models, revise_map, EPS

B = json.load(open("../data/bench.json"))
CFGS = ["gpt-4o-mini__logprob", "gpt-4.1-nano__logprob", "gpt-4.1-mini__logprob",
        "gpt-4.1-mini__sample", "gpt-5-mini__sample", "gpt-4.1__logprob", "jev-1.13.0__jev"]


def path_interpretations(cfg, atoms):
    idx = {tuple(a): i for i, a in enumerate(atoms)}
    start = (ENTRY, 0)
    seen, out, q = {start}, set(), deque([start])
    while q:
        u, s = q.popleft()
        for v in cfg.g.successors(u):
            kind = cfg.g[u][v]["kind"]
            t = s
            if u in cfg.cond_headers and kind in ("T", "F"):
                t |= 1 << idx[("condT" if kind == "T" else "condF", u)]
            if v == EXIT:
                out.add(t)
                continue
            if v not in cfg.stmt:
                continue  # exception node: not a normal termination
            t |= 1 << idx[("reach", v)]
            if (v, t) not in seen:
                seen.add((v, t))
                q.append((v, t))
    return out


def as_mask(bits):
    return sum(1 << i for i, b in enumerate(bits) if b)


def map_over(cands, probs, n):
    lp = [(math.log(min(max(p, EPS), 1 - EPS)), math.log(1 - min(max(p, EPS), 1 - EPS))) for p in probs]
    best = max(cands, key=lambda m: sum(lp[i][0] if m >> i & 1 else lp[i][1] for i in range(n)))
    return [bool(best >> i & 1) for i in range(n)]


if __name__ == "__main__":
    tot_m = tot_p = equal = gt_in = subset_ok = 0
    paths = {}
    for p in B:
        fn = parse_function(p["code"])
        cfg = CFG(fn)
        atoms = make_atoms(cfg)
        assert [list(a) for a in atoms] == p["atoms"]
        n = len(atoms)
        P = path_interpretations(cfg, atoms)
        M = {as_mask(m) for m in models(p["sound"], n)}
        subset_ok += P <= M
        gt_in += as_mask(p["gt"]) in P
        equal += P == M
        tot_m += len(M)
        tot_p += len(P)
        paths[p["id"]] = P
    out = dict(programs=len(B), paths_subset_of_models=subset_ok, truth_in_paths=gt_in,
               equal=equal, models_total=tot_m, paths_total=tot_p)
    print(out, flush=True)
    R = json.load(open("../data/results.json"))
    for c in CFGS:
        cache = load_cache(c)
        ok = n_all = 0
        for p in B:
            pr = [cache[key(p, tuple(a))]["p"] for a in p["atoms"]]
            rev = map_over(paths[p["id"]], pr, len(pr))
            ok += sum(a == b for a, b in zip(rev, p["gt"]))
            n_all += len(pr)
        out[c] = dict(map_paths=ok / n_all, map_tp=R[c]["acc"]["map"], raw=R[c]["acc"]["raw"])
        print(c, out[c], flush=True)
    json.dump(out, open("../data/completeness.json", "w"), indent=1)
