"""Generate LaTeX tables and extra statistics from results.json and the caches."""
import json
import numpy as np
from query import load_cache, key
from revise import literals, consistent, revise_map, revise_marginal, models

R = json.load(open("../data/results.json"))
RB = json.load(open("../data/robustness.json"))
B = json.load(open("../data/bench.json"))
ORDER = [("gpt-4o-mini__logprob", "GPT-4o-mini", "logprob"),
         ("gpt-4.1-nano__logprob", "GPT-4.1-nano", "logprob"),
         ("gpt-4.1-mini__logprob", "GPT-4.1-mini", "logprob"),
         ("gpt-4.1-mini__sample", "GPT-4.1-mini", "sampling"),
         ("gpt-5-mini__sample", "GPT-5-mini", "sampling"),
         ("gpt-4.1__logprob", "GPT-4.1", "logprob"),
         ("jev-1.13.0__jev", "Jev 1.13", "Noul")]
pct = lambda x: f"{100 * x:.1f}"


EXTRA = [("gpt-5-mini__reason", "GPT-5-mini", "reasoning")]


def t_main():
    rows = []
    for c, m, s in ORDER + [x for x in EXTRA if x[0] in R]:
        if (c, m, s) in EXTRA:
            rows.append("\\midrule")
        r = R[c]
        a, pa = r["acc"], r["prog_acc"]
        w2c, c2w, p = r["mcnemar"]["map"]
        best = max(a["map"], a["marg"])
        rows.append(
            f"{m} & {s} & {r['inconsistent_programs']} & {pct(a['raw'])} & {pct(a['map'])} & {pct(a['marg'])} "
            f"& {pct(a['map_uns'])} & {w2c}/{c2w} & {pct(pa['raw'])} & {pct(pa['map'])} \\\\")
    return "\n".join(rows)


def t_theory():
    rows = []
    for c, m, s in ORDER:
        r = R[c]
        mu, eb, ta = r["mus"], r["error_bound"], RB[c]["t_all"]
        rows.append(
            f"{m} & {s} & {mu['contains_error']}/{mu['total']} & {mu['unsound_contains_error']}/{mu['unsound_total']} "
            f"& {eb['progs_flagged']}/{eb['progs_with_err']} & {ta['flagged']}/{ta['progs_with_err']} "
            f"& {eb['sum_lb']}/{eb['sum_err']} & {ta['lb']}/{ta['errors']} \\\\")
    return "\n".join(rows)


def t_ablation():
    rows = []
    for c, m, s in ORDER:
        r = R[c]
        ab = r["ablation_map_acc"]
        rows.append(f"{m} & {s} & {pct(r['acc']['raw'])} & {pct(RB[c]['t_all']['map_acc'])} & {pct(ab['dom'])} & {pct(ab['dom+pdom'])} "
                    f"& {pct(ab['dom+pdom+cond'])} & {pct(ab['full'])} \\\\")
    return "\n".join(rows)


def t_calib():
    rows = []
    for c, m, s in ORDER:
        r = R[c]
        a = r["acc"]
        rows.append(f"{m} & {s} & {r['ece_raw']:.3f} & {r['ece_cal']:.3f} & {pct(a['map'])} & {pct(a['map_cal'])} "
                    f"& {pct(a['marg'])} & {pct(a['marg_cal'])} \\\\")
    return "\n".join(rows)


def extra():
    out = {}
    for c, m, s in ORDER:
        cache = load_cache(c)
        diff = n = 0
        mus_sizes = []
        for p in B:
            pr = [cache[key(p, tuple(a))]["p"] for a in p["atoms"]]
            m1 = revise_map(p["sound"], pr)
            m2, _ = revise_marginal(p["sound"], pr)
            diff += sum(x != y for x, y in zip(m1, m2))
            n += len(pr)
        out[c] = dict(map_marg_disagree=diff, atoms=n)
        if c.startswith("jev"):
            toks = {}
            for v in cache.values():
                toks[v["key"].split("|")[0]] = v["tokens"]
            out[c]["jev_input_tokens"] = int(sum(toks.values()))
    n_models = [len(models(p["sound"], len(p["atoms"]))) for p in B]
    out["models_of_T"] = dict(mean=float(np.mean(n_models)), max=int(max(n_models)),
                             median=float(np.median(n_models)))
    out["clauses"] = dict(mean=float(np.mean([len(p["sound"]) for p in B])))
    return out


if __name__ == "__main__":
    import os
    os.makedirs("../paper/tables", exist_ok=True)
    for name, f in [("main", t_main), ("theory", t_theory), ("ablation", t_ablation), ("calib", t_calib)]:
        open(f"../paper/tables/{name}.tex", "w").write(f() + "\n")
    e = extra()
    json.dump(e, open("../data/extra.json", "w"), indent=1)
    print(json.dumps(e, indent=1))
