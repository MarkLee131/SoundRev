"""Elicit beliefs from TypeSafe Jev (jev-1.13.0), following its documented
practices: a JSON state with named fields and numbered code, one batched call
per program (questions are evaluated independently), positive phrasing,
criteria that fix the yes/no boundary, and two phrasings per atom whose
probabilities are averaged in log-odds."""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typesafe_sdk import TypeSafeClient, Noul, NoulCriteria
from query import numbered, src_line, cond_src, key, load_cache, append_cache

MODEL = "jev-1.13.0"
NAME = MODEL + "__jev"
client = TypeSafeClient(api_key=os.environ["JEV_API_KEY"], model=MODEL)


def nouls(p, atom):
    kind, l = atom
    call = f"{p['fn']}({p['input']})"
    if kind == "reach":
        s = src_line(p["code"], l)
        crit = NoulCriteria(
            true=f"During the execution of `call`, the statement on line {l} of `code` runs one or more times.",
            false=f"During the execution of `call`, the statement on line {l} of `code` never runs.")
        q = Noul(instructions=f"When `call` is executed, is line {l} of `code` (`{s}`) executed at least once?",
                 criteria=crit)
        st = Noul(instructions=f"Line {l} of `code` (`{s}`) is executed at least once when `call` runs.",
                  criteria=crit)
    else:
        v = "True" if kind == "condT" else "False"
        c = cond_src(p["code"], l)
        crit = NoulCriteria(
            true=f"During the execution of `call`, at least one evaluation of the condition on line {l} of `code` yields {v}.",
            false=f"During the execution of `call`, no evaluation of the condition on line {l} of `code` yields {v} (including the case that it is never evaluated).")
        q = Noul(instructions=f"When `call` is executed, does the condition `{c}` on line {l} of `code` evaluate to {v} at least once?",
                 criteria=crit)
        st = Noul(instructions=f"The condition `{c}` on line {l} of `code` evaluates to {v} at least once when `call` runs.",
                  criteria=crit)
    return q, st


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def ask_program(p):
    state = {"code": numbered(p["code"]), "call": f"{p['fn']}({p['input']})",
             "note": "Line numbers are shown left of '|'. Line 1 is the def header."}
    qs = {}
    for i, a in enumerate(p["atoms"]):
        q, st = nouls(p, tuple(a))
        qs[f"a{i}_q"] = q
        qs[f"a{i}_s"] = st
    r = client.system_one(state=state, questions=qs)
    recs = []
    for i, a in enumerate(p["atoms"]):
        pq = r.answers[f"a{i}_q"].noul
        ps = r.answers[f"a{i}_s"].noul
        z = (logit(pq) + logit(ps)) / 2
        recs.append(dict(key=key(p, tuple(a)), p=1 / (1 + math.exp(-z)), p_q=pq, p_s=ps,
                         tokens=r.usage.input_tokens))
    return recs


def run(bench, workers=8):
    cache = load_cache(NAME)
    todo = [p for p in bench if any(key(p, tuple(a)) not in cache for a in p["atoms"])]
    print(NAME, "todo programs", len(todo), flush=True)

    def work(p):
        for _ in range(5):
            try:
                for rec in ask_program(p):
                    append_cache(NAME, rec)
                return
            except Exception as e:  # retry transient API errors
                err = e
        print("FAILED", p["id"], err, flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    import sys
    bench = json.load(open("../data/bench.json"))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(bench)
    run(bench[:n])
