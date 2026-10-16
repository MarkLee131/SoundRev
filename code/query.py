"""Elicit LLM beliefs about program-property atoms.

Each atom is asked in a separate call, so answers do not condition on each
other. Confidence sources:
  logprob : P(Yes) / (P(Yes)+P(No)) from the first answer token (temperature 0)
  sample  : fraction of Yes among k samples at temperature 1
  reason  : fraction of Yes among k=3 answers with full (medium) reasoning
  jev     : probability from TypeSafe Jev (see jev.py)
Results are cached in ../data/cache/<model>__<source>.jsonl
"""
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

CACHE_DIR = "../data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
_lock = threading.Lock()

SYSTEM = ("You are an expert Python programmer. You reason about what happens "
          "when a Python function is executed. Answer the question with a single "
          "word: Yes or No.")


def numbered(code):
    return "\n".join(f"{i + 1:>3}| {ln}" for i, ln in enumerate(code.rstrip("\n").split("\n")))


def src_line(code, l):
    return code.split("\n")[l - 1].strip()


def cond_src(code, l):
    s = src_line(code, l)
    for kw in ("elif ", "if ", "while "):
        if s.startswith(kw):
            s = s[len(kw):]
            break
    return s.rstrip(":").strip()


def question(p, atom):
    kind, l = atom
    call = f"{p['fn']}({p['input']})"
    if kind == "reach":
        return (f"When `{call}` is executed, is line {l} (`{src_line(p['code'], l)}`) "
                f"executed at least once?")
    val = "True" if kind == "condT" else "False"
    return (f"When `{call}` is executed, does the condition on line {l} "
            f"(`{cond_src(p['code'], l)}`) evaluate to {val} at least once?")


def prompt(p, atom):
    return (f"Python function (line numbers on the left):\n```\n{numbered(p['code'])}\n```\n\n"
            f"{question(p, atom)}\nAnswer Yes or No.")


def key(p, atom):
    return f"{p['id']}|{atom[0]}|{atom[1]}"


def load_cache(name):
    path = os.path.join(CACHE_DIR, name + ".jsonl")
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            r = json.loads(ln)
            out[r["key"]] = r
    return out


def append_cache(name, rec):
    with _lock:
        with open(os.path.join(CACHE_DIR, name + ".jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


client = OpenAI()


def ask_logprob(model, p, atom):
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=1, logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt(p, atom)}])
    yes = no = 0.0
    for t in r.choices[0].logprobs.content[0].top_logprobs:
        tok = t.token.strip().lower()
        if tok in ("yes", "y"):
            yes += math.exp(t.logprob)
        elif tok in ("no", "n"):
            no += math.exp(t.logprob)
    prob = yes / (yes + no) if yes + no > 0 else 0.5
    return dict(p=prob, mass=yes + no, top=r.choices[0].message.content)


def ask_sample(model, p, atom, k=5):
    kw = dict(model=model, n=k, messages=[{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": prompt(p, atom)}])
    if model.startswith("gpt-5"):
        kw["reasoning_effort"] = "minimal"
    else:
        kw["temperature"] = 1.0
        kw["max_tokens"] = 3
    r = client.chat.completions.create(**kw)
    ans = [c.message.content.strip().lower() for c in r.choices]
    ys = sum(a.startswith("y") for a in ans)
    ns = sum(a.startswith("n") for a in ans)
    return dict(p=(ys + 0.5) / (ys + ns + 1.0), ys=ys, ns=ns, raw=ans)


def ask_reason(model, p, atom, k=3):
    """Full reasoning (default effort), k independent answers."""
    ans, toks = [], 0
    for _ in range(k):
        r = client.chat.completions.create(
            model=model, reasoning_effort="medium", max_completion_tokens=16000,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt(p, atom)}])
        ans.append((r.choices[0].message.content or "").strip().lower())
        toks += r.usage.completion_tokens
    ys = sum(a.startswith("y") for a in ans)
    ns = sum(a.startswith("n") for a in ans)
    return dict(p=(ys + 0.5) / (ys + ns + 1.0), ys=ys, ns=ns, raw=ans, out_tokens=toks)


def run(model, source, bench, workers=32):
    name = f"{model}__{source}"
    cache = load_cache(name)
    todo = [(p, tuple(a)) for p in bench for a in p["atoms"] if key(p, tuple(a)) not in cache]
    print(name, "cached", len(cache), "todo", len(todo), flush=True)
    fn = {"logprob": ask_logprob, "sample": ask_sample, "reason": ask_reason}[source]

    def work(item):
        p, a = item
        for attempt in range(5):
            try:
                rec = fn(model, p, a)
                rec["key"] = key(p, a)
                append_cache(name, rec)
                return
            except Exception as e:  # retry transient API errors
                err = e
        print("FAILED", key(p, a), err, file=sys.stderr)

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(work, todo)):
            if i % 500 == 0:
                print(name, i, flush=True)
    return load_cache(name)


if __name__ == "__main__":
    bench = json.load(open("../data/bench.json"))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else len(bench)
    run(sys.argv[1], sys.argv[2], bench[:n])
