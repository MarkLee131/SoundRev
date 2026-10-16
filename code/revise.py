"""Revision operators over a belief base B = {(l_i, p_i)} and hard theory T.

MAP      : weighted partial MaxSAT, soft unit clause for each answered literal
           with weight log(p/(1-p)) of the answered polarity (MLN-style MAP).
MARGINAL : exact posterior P(a | T) under independent Bernoulli likelihoods,
           computed by enumerating the models of T (weighted model counting).
Also: MUS enumeration and minimum-cardinality correction sets.
"""
import math
from pysat.formula import WCNF, CNF
from pysat.examples.rc2 import RC2
from pysat.examples.optux import OptUx
from pysat.solvers import Solver

EPS = 1e-4
SCALE = 1000


def clip(p):
    return min(max(p, EPS), 1 - EPS)


def literals(probs):
    """Answered literal for atom i (1-based) and its log-odds weight."""
    lits, ws = [], []
    for i, p in enumerate(probs, 1):
        p = clip(p)
        lit = i if p >= 0.5 else -i
        lits.append(lit)
        ws.append(abs(math.log(p / (1 - p))))
    return lits, ws


def consistent(hard, lits):
    with Solver(name="g4", bootstrap_with=hard) as s:
        return s.solve(assumptions=lits)


def revise_map(hard, probs):
    lits, ws = literals(probs)
    w = WCNF()
    for c in hard:
        w.append(c)
    for l, x in zip(lits, ws):
        w.append([l], weight=max(1, int(round(x * SCALE))))
    with RC2(w) as rc2:
        m = rc2.compute()
    val = {abs(v): v > 0 for v in m}
    return [val.get(i, lits[i - 1] > 0) for i in range(1, len(probs) + 1)]


def models(hard, n, cap=200000):
    """All assignments over atoms 1..n satisfying hard (blocking clauses)."""
    out = []
    with Solver(name="g4", bootstrap_with=hard) as s:
        while s.solve():
            m = s.get_model()
            a = [(v > 0) for v in m[:n]]
            # model() may be shorter if some vars do not occur; pad
            a += [False] * (n - len(a))
            full = [i + 1 if a[i] else -(i + 1) for i in range(n)]
            out.append(a)
            s.add_clause([-x for x in full])
            if len(out) >= cap:
                raise RuntimeError("too many models")
    return out


def revise_marginal(hard, probs, mods=None):
    n = len(probs)
    mods = mods if mods is not None else models(hard, n)
    lp = [(math.log(clip(p)), math.log(1 - clip(p))) for p in probs]
    logw = [sum(lp[i][0] if m[i] else lp[i][1] for i in range(n)) for m in mods]
    mx = max(logw)
    w = [math.exp(x - mx) for x in logw]
    z = sum(w)
    post = [sum(wk for wk, m in zip(w, mods) if m[i]) / z for i in range(n)]
    return [q > 0.5 for q in post], post


def muses(hard, lits, cap=1000):
    """Enumerate MUSes of the answered literals relative to the hard theory."""
    w = WCNF()
    for c in hard:
        w.append(c)
    for l in lits:
        w.append([l], weight=1)
    res = []
    with OptUx(w, unsorted=True) as ox:
        for mus in ox.enumerate():
            res.append([lits[k - 1] for k in mus])
            if len(res) >= cap:
                break
    return res


def min_correction(hard, lits):
    """Minimum-cardinality correction set (unweighted MaxSAT cost)."""
    w = WCNF()
    for c in hard:
        w.append(c)
    for l in lits:
        w.append([l], weight=1)
    with RC2(w) as rc2:
        rc2.compute()
        return rc2.cost
