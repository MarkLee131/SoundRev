"""Self-refine baseline: the model sees all its own answers about one program
and returns a corrected answer for every question (one call per program).
Cache: ../data/cache/<model>__selfrefine.jsonl, one record per program."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from query import client, numbered, question, load_cache, append_cache, key

SYSTEM = ("You are an expert Python programmer. You reason about what happens "
          "when a Python function is executed.")


def refine_prompt(p, answers):
    qs = "\n".join(f"{i + 1}. {question(p, tuple(a))} Current answer: {ans}"
                   for i, (a, ans) in enumerate(zip(p["atoms"], answers)))
    return (f"Python function (line numbers on the left):\n```\n{numbered(p['code'])}\n```\n\n"
            f"Here are questions about one execution of this function, with your earlier answers. "
            f"Some answers may be wrong, and some may contradict each other or the control flow "
            f"of the function. Check every answer against the program and against the other "
            f"answers, and correct the answers that are wrong.\n\n{qs}\n\n"
            f"Return a JSON object that maps every question number to \"Yes\" or \"No\", "
            f"for example {{\"1\": \"Yes\", \"2\": \"No\"}}.")


def run(cfg, workers=16):
    model = cfg.split("__")[0]
    base = load_cache(cfg)
    name = f"{model}__selfrefine"
    done = {r["key"] for r in load_cache(name).values()}
    B = json.load(open("../data/bench.json"))
    todo = [p for p in B if p["id"] not in done]
    print(name, "todo", len(todo), flush=True)

    def work(p):
        answers = ["Yes" if base[key(p, tuple(a))]["p"] >= 0.5 else "No" for a in p["atoms"]]
        for _ in range(5):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": refine_prompt(p, answers)}])
                out = json.loads(r.choices[0].message.content)
                new = [str(out.get(str(i + 1), answers[i])).strip().lower().startswith("y")
                       for i in range(len(answers))]
                append_cache(name, dict(key=p["id"], answers=new,
                                        missing=sum(str(i + 1) not in out for i in range(len(answers)))))
                return
            except Exception as e:
                err = e
        print("FAILED", p["id"], err, file=sys.stderr)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
