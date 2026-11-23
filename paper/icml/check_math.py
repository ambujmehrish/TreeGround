"""Numerical check of the tree-answer rule and the rule likelihood against brute-force enumeration on small random
trees (CPU, seconds). Checks: (1) the bottom-up recursion returns the largest maximiser of P(S) - tau C(S) over
downward-closed sets; (2) answers are nested in tau; (3) log Z by recursion = log of the enumerated sum; (4) the
closed-form inclusion probabilities; (5) T d log Z / d w = inclusion probability; (6) the constrained partition
function; (7) the zero-temperature limit of the loss.
usage: python paper/icml/check_math.py
"""
import itertools
import math
import random


def random_forest(n, rng):
    # node 0 = image root, node n = the "no target" leaf (its own tree); parents of 1..n-1 are earlier nodes
    parent = [-1] + [rng.randrange(0, i) for i in range(1, n)] + [-1]
    children = [[] for _ in range(n + 1)]
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
    return parent, children


def descendants(children, v):
    out, stack = [], [v]
    while stack:
        u = stack.pop()
        out.append(u)
        stack += children[u]
    return out


def lower_sets(parent, children):
    n = len(parent)
    D = [frozenset(descendants(children, v)) for v in range(n)]
    for bits in itertools.product((0, 1), repeat=n):
        S = frozenset(v for v in range(n) if bits[v])
        if all(D[v] <= S for v in S):
            yield S


def solve(children, w, v):
    # (best value, largest maximiser, subtree weight) inside the subtree of v
    sub = [solve(children, w, c) for c in children[v]]
    W = w[v] + sum(s[2] for s in sub)
    kids = sum(s[0] for s in sub)
    if W >= kids - 1e-12 and W >= -1e-12:
        return W, frozenset(descendants(children, v)), W
    return kids, frozenset().union(*[s[1] for s in sub]) if sub else frozenset(), W


def log_z(children, w, T, v, forced=frozenset(), forbidden=frozenset()):
    # (A(v), W(v)); with constraints: nodes in `forced` must be in S, nodes in `forbidden` must not
    sub = [log_z(children, w, T, c, forced, forbidden) for c in children[v]]
    W = w[v] + sum(s[1] for s in sub)
    opts = []
    if not (set(descendants(children, v)) & forbidden):
        opts.append(W / T)
    if v not in forced:
        opts.append(sum(s[0] for s in sub))
    m = max(opts)
    return m + math.log(sum(math.exp(o - m) for o in opts)), W


def main():
    rng = random.Random(0)
    worst = dict(opt=0.0, logz=0.0, marg=0.0, grad=0.0, cons=0.0, limit=0.0)
    nested_ok = True
    for trial in range(300):
        n = rng.randrange(2, 9)
        parent, children = random_forest(n, rng)
        N = n + 1
        q = [rng.random() for _ in range(N)]
        g = [rng.random() * 0.3 for _ in range(N)]
        roots = [0, n]
        sets = list(lower_sets(parent, children))
        prev = None
        for tau in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.0):
            w = [q[u] - tau * g[u] for u in range(N)]
            val = lambda S: sum(w[u] for u in S)
            best = max(val(S) for S in sets)
            largest = frozenset().union(*[S for S in sets if val(S) >= best - 1e-12])
            res = [solve(children, w, r) for r in roots]
            S = frozenset().union(*[r[1] for r in res])
            worst["opt"] = max(worst["opt"], abs(sum(r[0] for r in res) - best))
            assert S == largest, (trial, tau)
            if prev is not None and not prev <= S:
                nested_ok = False
            prev = S
        tau, T = 2.0, 0.37
        w = [q[u] - tau * g[u] for u in range(N)]
        val = lambda S: sum(w[u] for u in S)
        Z = sum(math.exp(val(S) / T) for S in sets)
        A = {r: log_z(children, w, T, r) for r in roots}
        worst["logz"] = max(worst["logz"], abs(sum(a[0] for a in A.values()) - math.log(Z)))
        # closed-form inclusion probabilities
        info = {}

        def fill(v, free):
            a, W = log_z(children, w, T, v)
            pi = math.exp(W / T - a)
            info[v] = 1.0 - free * (1.0 - pi)
            for c in children[v]:
                fill(c, free * (1.0 - pi))
        for r in roots:
            fill(r, 1.0)
        for u in range(N):
            brute = sum(math.exp(val(S) / T) for S in sets if u in S) / Z
            worst["marg"] = max(worst["marg"], abs(info[u] - brute))
            eps = 1e-6
            w2 = list(w)
            w2[u] += eps
            d = (sum(log_z(children, w2, T, r)[0] for r in roots) - sum(a[0] for a in A.values())) / eps
            worst["grad"] = max(worst["grad"], abs(T * d - brute))
        # constrained partition function: gold = descendants of a random node, a few ignored nodes
        t = rng.randrange(0, n)
        gold = frozenset(descendants(children, t))
        ignored = frozenset(u for u in range(n) if u not in gold and rng.random() < 0.2)
        forbidden = frozenset(range(N)) - gold - ignored
        ok = [S for S in sets if gold <= S <= gold | ignored]
        Zg = sum(math.exp(val(S) / T) for S in ok)
        Ag = sum(log_z(children, w, T, r, gold, forbidden)[0] for r in roots)
        worst["cons"] = max(worst["cons"], abs(Ag - math.log(Zg)))
        Ts = 1e-3
        loss = sum(log_z(children, w, Ts, r)[0] for r in roots) - sum(
            log_z(children, w, Ts, r, gold, forbidden)[0] for r in roots)
        gap = max(val(S) for S in sets) - max(val(S) for S in ok)
        worst["limit"] = max(worst["limit"], abs(Ts * loss - gap))
    print("nested in tau:", nested_ok)
    for k, v in worst.items():
        print(f"max abs error {k}: {v:.2e}")


if __name__ == "__main__":
    main()
