"""Build the benchmark from CRUXEval: filter supported programs, extract atoms,
ground truth and the background theory; check T_P soundness on every run."""
import json
import ast
import collections
import pandas as pd
from program import (parse_function, CFG, make_atoms, theory, ground_truth,
                     satisfies, Unsupported)

df = pd.read_parquet("../data/cruxeval.parquet")
stats = collections.Counter()
out = []
for _, r in df.iterrows():
    code = r.code.strip("\n") + "\n"
    try:
        fn = parse_function(code)
    except Unsupported as e:
        stats["unsupported:" + str(e).split()[0]] += 1
        continue
    except SyntaxError:
        stats["syntax"] += 1
        continue
    cfg = CFG(fn)
    if not cfg.cond_headers:
        stats["no-branch"] += 1
        continue
    atoms = make_atoms(cfg)
    try:
        gt, res = ground_truth(code, fn.name, r.input, atoms)
    except BaseException as e:  # noqa: BLE001 (exceptions and timeouts are filtered out)
        stats["runtime-exc"] += 1
        continue
    sound, tags = theory(cfg, atoms, "sound")
    bad = satisfies(sound, atoms, gt)
    if bad:  # would falsify the soundness lemma; must never happen
        stats["SOUNDNESS-VIOLATION"] += 1
        print("violation", r.id, [tags[k] for k in bad])
        continue
    uns, utags = theory(cfg, atoms, "unsound")
    ubad = satisfies(uns, atoms, gt)
    stats["ok"] += 1
    stats["unsound-violated"] += bool(ubad)
    out.append(dict(id=r.id, code=code, fn=fn.name, input=r.input,
                    atoms=atoms, gt=[gt[a] for a in atoms],
                    sound=sound, sound_tags=tags, unsound=uns, unsound_tags=utags,
                    n_loops=sum(isinstance(s, (ast.For, ast.While)) for s in cfg.stmt.values())))
print(dict(stats))
print("programs", len(out), "atoms", sum(len(p["atoms"]) for p in out))
json.dump(out, open("../data/bench.json", "w"))
