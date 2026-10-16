"""Baseline: constraints written by an LLM instead of derived from the CFG.

As in ConCoRD and Maieutic prompting, a model supplies the relations between
answers. GPT-4.1 sees the function and the numbered atoms, and it lists clauses
that hold on every normally terminating call. We keep clauses with at least two
literals, so the generator states relations, not answers. Answers of each
configuration are then revised by MAP with this theory instead of T_P.
Cache: ../data/cache/<gen>__theory.jsonl. Writes ../data/llmtheory.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pysat.solvers import Solver
from query import client, numbered, load_cache, append_cache, key
from revise import literals, revise_map, muses, consistent
from program import satisfies

GEN = "gpt-4.1"
B = json.load(open("../data/bench.json"))
R = json.load(open("../data/results.json"))
CFGS = ["gpt-4o-mini__logprob", "gpt-4.1-nano__logprob", "gpt-4.1-mini__logprob",
        "gpt-4.1-mini__sample", "gpt-5-mini__sample", "gpt-4.1__logprob", "jev-1.13.0__jev"]


def atom_text(a):
    kind, l = a
    if kind == "reach":
        return f"line {l} is executed at least once"
    return f"the condition on line {l} evaluates to {'True' if kind == 'condT' else 'False'} at least once"


def gen_prompt(p):
    atoms = "\n".join(f"{i + 1}. {atom_text(a)}" for i, a in enumerate(p["atoms"]))
    return (f"Python function (line numbers on the left):\n```\n{numbered(p['code'])}\n```\n\n"
            f"Consider any call of this function that returns normally. Each numbered statement "
            f"below is true or false for such a call.\n\n{atoms}\n\n"
            f"List logical constraints between these statements that hold for every such call, "
            f"whatever the arguments are. Write each constraint as a clause: a list of integers, "
            f"where k means statement k is true and -k means statement k is false, and the clause "
            f"holds if at least one of its literals holds. For example, \"if statement 3 is true, "
            f"then statement 2 is true\" is the clause [-3, 2]. Each clause must have at least two "
            f"literals. Return a JSON object of the form "
            f"{{\"constraints\": [{{\"reason\": \"...\", \"clause\": [-3, 2]}}, ...]}}.")


def generate(workers=16):
    name = f"{GEN}__theory"
    done = set(load_cache(name))
    todo = [p for p in B if p["id"] not in done]
    print(name, "todo", len(todo), flush=True)

    def work(p):
        n = len(p["atoms"])
        for _ in range(5):
            try:
                r = client.chat.completions.create(
                    model=GEN, temperature=0, max_tokens=4096, response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": gen_prompt(p)}])
                raw = json.loads(r.choices[0].message.content).get("constraints", [])
                cls = []
                for item in raw:
                    c = item.get("clause") if isinstance(item, dict) else None
                    if not isinstance(c, list) or not all(isinstance(x, int) for x in c):
                        continue
                    if len(set(c)) >= 2 and all(1 <= abs(x) <= n for x in c):
                        cls.append(sorted(set(c)))
                append_cache(name, dict(key=p["id"], clauses=cls, raw=len(raw)))
                return
            except Exception as e:
                err = e
        print("FAILED", p["id"], err, file=sys.stderr)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


def sat(cls):
    with Solver(name="g4", bootstrap_with=cls) as s:
        return s.solve()


def evaluate():
    rng = np.random.default_rng(0)
    th = load_cache(f"{GEN}__theory")
    out = dict(programs=len(B))
    n_cl = viol_cl = viol_prog = unsat = 0
    T = {}
    for p in B:
        atoms = [tuple(a) for a in p["atoms"]]
        gt = {a: v for a, v in zip(atoms, p["gt"])}
        cls = th[p["id"]]["clauses"]
        n_cl += len(cls)
        bad = satisfies(cls, atoms, gt)
        viol_cl += len(bad)
        viol_prog += bool(bad)
        if not sat(cls):
            unsat += 1
            cls = []
        T[p["id"]] = cls
    out.update(clauses=n_cl, clauses_per_prog=n_cl / len(B), clauses_violated=viol_cl,
               progs_violated=viol_prog, unsat_theories=unsat)
    print(out, flush=True)
    for c in CFGS:
        cache = load_cache(c)
        n, raw_c, llm_c = [], [], []
        w2c = c2w = mus_t = mus_hit = 0
        for p in B:
            pr = [cache[key(p, tuple(a))]["p"] for a in p["atoms"]]
            g = p["gt"]
            lits, _ = literals(pr)
            raw = [l > 0 for l in lits]
            rev = revise_map(T[p["id"]], pr)
            w2c += sum(r != t and v == t for r, v, t in zip(raw, rev, g))
            c2w += sum(r == t and v != t for r, v, t in zip(raw, rev, g))
            for M in ([] if consistent(T[p["id"]], lits) else muses(T[p["id"]], lits)):
                mus_t += 1
                mus_hit += any((l > 0) != g[abs(l) - 1] for l in M)
            n.append(len(g))
            raw_c.append(sum(a == b for a, b in zip(raw, g)))
            llm_c.append(sum(a == b for a, b in zip(rev, g)))
        n, raw_c, llm_c = map(np.array, (n, raw_c, llm_c))
        idx = rng.integers(0, len(n), size=(10000, len(n)))
        d = (llm_c[idx].sum(1) - raw_c[idx].sum(1)) / n[idx].sum(1)
        out[c] = dict(raw=float(raw_c.sum() / n.sum()), llm=float(llm_c.sum() / n.sum()),
                      map_tp=R[c]["acc"]["map"], w2c=int(w2c), c2w=int(c2w),
                      mus=mus_t, mus_with_error=mus_hit,
                      ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        print(c, out[c], flush=True)
    json.dump(out, open("../data/llmtheory.json", "w"), indent=1)


if __name__ == "__main__":
    if "gen" in sys.argv[1:]:
        generate()
    evaluate()
