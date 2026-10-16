"""Program model: statement-level CFG, property atoms, ground truth, and the
background theory T_P derived from the CFG.

Atoms for a program P and input I:
  reach(l)   line l is executed at least once during the call
  condT(h)   the condition at header h evaluates to True at least once
  condF(h)   the condition at header h evaluates to False at least once

T_P is a set of hard clauses over these atoms. Every clause is sound for any
call of P that terminates normally (no uncaught exception), given that P is
not recursive (checked by the filter).
"""
import ast
import sys
import signal
import networkx as nx

ENTRY, EXIT, RAISE = "ENTRY", "EXIT", "RAISE"


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------- filtering

BANNED = (ast.Try, ast.With, ast.AsyncWith, ast.Match, ast.Yield, ast.YieldFrom,
          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Global,
          ast.Nonlocal, ast.AsyncFor, ast.Await)
try:
    BANNED = BANNED + (ast.TryStar,)
except AttributeError:
    pass


def parse_function(code):
    mod = ast.parse(code)
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or len(mod.body) != 1:
        raise Unsupported("not a single top-level function")
    fn = funcs[0]
    for node in ast.walk(ast.Module(body=fn.body, type_ignores=[])):
        if isinstance(node, BANNED):
            raise Unsupported(type(node).__name__)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == fn.name:
            raise Unsupported("recursive")
    # one statement per line, so that line-level atoms are unambiguous
    lines = {}
    for st in iter_stmts(fn.body):
        if st.lineno in lines:
            raise Unsupported("two statements on one line")
        lines[st.lineno] = st
        if isinstance(st, (ast.If, ast.While, ast.For)) and st.body and st.body[0].lineno == st.lineno:
            raise Unsupported("compound statement on one line")
    return fn


def iter_stmts(stmts):
    for st in stmts:
        yield st
        for field in ("body", "orelse"):
            sub = getattr(st, field, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                yield from iter_stmts(sub)


# ---------------------------------------------------------------- CFG

class CFG:
    """Statement-level CFG. Nodes are line numbers plus ENTRY/EXIT/RAISE.
    Edge attribute 'kind' in {seq, T, F}. Headers of if/while carry T/F edges;
    a for-header carries T (next iteration) and F (exhausted)."""

    def __init__(self, fn):
        self.g = nx.DiGraph()
        self.g.add_nodes_from([ENTRY, EXIT, RAISE])
        self.stmt = {}
        self.cond_headers = {}   # line -> 'if' | 'while'
        self.returns = []
        body = fn.body
        # A leading docstring compiles to no bytecode and never runs.
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        first = self._block(body, EXIT, None, None)
        self.g.add_edge(ENTRY, first, kind="seq")
        self.first = first

    def _edge(self, a, b, kind="seq"):
        self.g.add_edge(a, b, kind=kind)

    def _block(self, stmts, nxt, brk, cont):
        """Wire a statement list; return the node control enters first."""
        entry = nxt
        for i in range(len(stmts) - 1, -1, -1):
            entry = self._stmt(stmts[i], entry, brk, cont)
        return entry

    def _stmt(self, st, nxt, brk, cont):
        l = st.lineno
        self.stmt[l] = st
        self.g.add_node(l)
        if isinstance(st, ast.Return):
            self.returns.append(l)
            self._edge(l, EXIT)
        elif isinstance(st, ast.Raise):
            self._edge(l, RAISE)
        elif isinstance(st, ast.Break):
            self._edge(l, brk)
        elif isinstance(st, ast.Continue):
            self._edge(l, cont)
        elif isinstance(st, ast.If):
            self.cond_headers[l] = "if"
            t = self._block(st.body, nxt, brk, cont)
            f = self._block(st.orelse, nxt, brk, cont) if st.orelse else nxt
            self._edge(l, t, "T")
            self._edge(l, f, "F")
        elif isinstance(st, ast.While):
            self.cond_headers[l] = "while"
            after = self._block(st.orelse, nxt, brk, cont) if st.orelse else nxt
            t = self._block(st.body, l, nxt, l)
            self._edge(l, t, "T")
            self._edge(l, after, "F")
        elif isinstance(st, ast.For):
            after = self._block(st.orelse, nxt, brk, cont) if st.orelse else nxt
            t = self._block(st.body, l, nxt, l)
            self._edge(l, t, "T")
            self._edge(l, after, "F")
        else:
            self._edge(l, nxt)
        return l

    def succ(self, h, kind):
        return [v for v in self.g.successors(h) if self.g[h][v]["kind"] == kind][0]

    def in_cycle(self):
        cyc = set()
        for comp in nx.strongly_connected_components(self.g):
            if len(comp) > 1:
                cyc |= comp
        cyc |= {n for n in self.g if self.g.has_edge(n, n)}
        return cyc


# ---------------------------------------------------------------- atoms

def make_atoms(cfg):
    atoms = []
    for l in sorted(cfg.stmt):
        atoms.append(("reach", l))
    for h in sorted(cfg.cond_headers):
        atoms.append(("condT", h))
        atoms.append(("condF", h))
    return atoms


# ---------------------------------------------------------------- theory

def theory(cfg, atoms, variant="sound"):
    """Return (clauses, tags). Literals are +-(index+1) into atoms.
    variant='sound'   : clauses C1..C6 below.
    variant='unsound' : sound clauses plus branch exclusivity for loop headers
                        (a plausible-looking but unsound rule)."""
    idx = {a: i + 1 for i, a in enumerate(atoms)}
    R = lambda l: idx[("reach", l)]
    cls, tags = [], []

    def add(c, tag):
        cls.append(list(c))
        tags.append(tag)

    g = cfg.g
    # C0: a call executes its first statement
    add([R(cfg.first)], "C0-entry")
    # C1: dominance
    idom = nx.immediate_dominators(g, ENTRY)
    for l in cfg.stmt:
        d = idom.get(l)
        if d is not None and d not in (ENTRY, l):
            add([-R(l), R(d)], "C1-dom")
    # C2: post-dominance w.r.t. normal exit (sound for normally terminating runs)
    rg = g.reverse(copy=True)
    ipdom = nx.immediate_dominators(rg, EXIT)
    for l in cfg.stmt:
        p = ipdom.get(l)
        if p is not None and p not in (EXIT, l) and p in cfg.stmt:
            add([-R(l), R(p)], "C2-pdom")
    cyc = cfg.in_cycle()
    for h in sorted(cfg.cond_headers):
        T, F = idx[("condT", h)], idx[("condF", h)]
        t, f = cfg.succ(h, "T"), cfg.succ(h, "F")
        # C3: condition outcomes imply the header was reached and the successor runs
        add([-T, R(h)], "C3-cond")
        add([-F, R(h)], "C3-cond")
        if t in cfg.stmt:
            add([-T, R(t)], "C3-cond")
            if set(g.predecessors(t)) == {h}:
                add([-R(t), T], "C3-cond")
        if f in cfg.stmt:
            add([-F, R(f)], "C3-cond")
            if set(g.predecessors(f)) == {h}:
                add([-R(f), F], "C3-cond")
        # C4: a reached header evaluates its condition one way or the other
        add([-R(h), T, F], "C4-total")
        # C5: outside cycles a header runs at most once, so outcomes exclude
        if h not in cyc:
            add([-T, -F], "C5-excl")
        elif variant == "unsound":
            add([-T, -F], "U-excl-loop")
    # C6: at most one return statement executes; at least one if no fall-through
    rets = sorted(cfg.returns)
    for i in range(len(rets)):
        for j in range(i + 1, len(rets)):
            add([-R(rets[i]), -R(rets[j])], "C6-ret")
    fall = [p for p in g.predecessors(EXIT) if p not in rets]
    if rets and not fall:
        add([R(r) for r in rets], "C6-ret")
    return cls, tags


# ---------------------------------------------------------------- ground truth

class _Rec(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        node.test = _wrap(node.test, node.lineno)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node.test = _wrap(node.test, node.lineno)
        return node


def _wrap(test, line):
    call = ast.Call(func=ast.Name("__rec__", ast.Load()),
                    args=[ast.Constant(line), test], keywords=[])
    return ast.copy_location(call, test)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def ground_truth(code, fn_name, inp, atoms, timeout=2, events=None):
    """Execute fn(inp); return dict atom->bool, or raise on exception/timeout.
    If `events` is a list, append the ordered ('line', l) and ('cond', l, value) events."""
    mod = ast.parse(code)
    mod = ast.fix_missing_locations(_Rec().visit(mod))
    outcomes = {}

    def rec(line, v):
        outcomes.setdefault(line, set()).add(bool(v))
        if events is not None:
            events.append(("cond", line, bool(v)))
        return v

    env = {"__rec__": rec}
    exec(compile(mod, "<prog>", "exec"), env)
    fn = env[fn_name]
    target = fn.__code__
    lines = set()

    def tracer(frame, event, arg):
        if frame.f_code is not target:
            return None
        if event == "line":
            lines.add(frame.f_lineno)
            if events is not None:
                events.append(("line", frame.f_lineno))
        return tracer

    args = eval("(" + inp + ",)", {})
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    sys.settrace(tracer)
    try:
        out = fn(*args)
    finally:
        sys.settrace(None)
        signal.alarm(0)
    gt = {}
    for a in atoms:
        kind, l = a
        if kind == "reach":
            gt[a] = l in lines
        elif kind == "condT":
            gt[a] = True in outcomes.get(l, ())
        else:
            gt[a] = False in outcomes.get(l, ())
    return gt, out


def satisfies(clauses, atoms, gt):
    val = {i + 1: gt[a] for i, a in enumerate(atoms)}
    bad = []
    for k, c in enumerate(clauses):
        if not any(val[abs(x)] == (x > 0) for x in c):
            bad.append(k)
    return bad
