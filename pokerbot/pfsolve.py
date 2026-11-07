"""Game-theoretic approximation: first-in push/fold at short stacks.

Abstraction (does NOT solve the full game):
  * pusher is first in with `n` players behind; all stacks = S big blinds
  * each caller decides independently; only the first caller plays (overcalls
    ignored); card removal between pusher and callers ignored
  * equities from the precomputed 169x169 heads-up class matrix
  * utility u(x) = x for gains, (1 + lam) * x for losses (lam ~ ICM risk
    aversion; lam = 0 is chip-EV)
Solved with fictitious play (average strategies converge to equilibrium of
this abstraction). Output: per (n, S, lam) push strategy and caller
strategies by caller type (non-blind / SB / BB), as 169-char digit strings
(probability * 9).
"""

import json
import os
import time
from multiprocessing import Pool
from operator import mul

from .cards import class_combos
from .features import load_tables, DATA_DIR
from .pushfold import STACKS, LAMS, NS


_EQ = None
_W = None


def _init(variant_name):
    global _EQ, _W
    t = load_tables(variant_name)
    hu = t["hu"]
    _EQ = [[hu[a * 169 + b] / 1000.0 for b in range(169)] for a in range(169)]
    _W = [float(len(class_combos(c))) for c in range(169)]


def _caller_types(n):
    """List of (type, posted) for callers in action order after the pusher."""
    out = []
    for j in range(n):
        if j == n - 1:
            out.append(("bb", 1.0))
        elif j == n - 2:
            out.append(("sb", 0.5))
        else:
            out.append(("nb", 0.0))
    return out


def _u(x, lam):
    return x if x >= 0 else (1.0 + lam) * x


def solve(n, S, lam, iters=300, ante=0.0):
    eq, W = _EQ, _W
    callers = _caller_types(n)
    pusher_posted = 0.5 if n == 1 else 0.0
    blinds_total = 1.5 + ante * (n + 1)
    types = sorted(set(t for t, _ in callers))
    posted = {t: p for t, p in callers}
    push = [1.0] * 169
    call = {t: [0.5] * 169 for t in types}
    push_avg = [0.0] * 169
    call_avg = {t: [0.0] * 169 for t in types}
    Wt = sum(W)
    S = max(S, 1.0)
    for it in range(1, iters + 1):
        # ---- callers best-respond to the pusher's average-so-far strategy
        ref_push = [x / (it - 1) for x in push_avg] if it > 1 else push
        pw = [W[a] * ref_push[a] for a in range(169)]
        tot = sum(pw)
        if tot <= 0:
            pw = list(W)
            tot = Wt
        # caller equity vs push range: for d, sum_a pw[a] * (1 - eq[a][d])
        eq_vs = [0.0] * 169
        for a in range(169):
            if pw[a]:
                row = eq[a]
                wa = pw[a]
                for d in range(169):
                    eq_vs[d] += wa * row[d]
        for t in types:
            p_j = posted[t]
            # pot if called: both all-in for S plus dead blinds/antes of others
            pot = 2 * S + blinds_total - p_j - pusher_posted
            br = []
            for d in range(169):
                e = 1.0 - eq_vs[d] / tot
                # net vs folding now (posted blind is sunk)
                ev_call = e * _u(pot - S + p_j, lam) + (1 - e) * _u(-(S - p_j), lam)
                br.append(1.0 if ev_call > 0.0 else 0.0)
            call[t] = br
            ca = call_avg[t]
            for d in range(169):
                ca[d] += br[d]
        # ---- pusher best-responds to callers' average strategies
        cprob = {}
        cval = {}
        for t in types:
            q = [x / it for x in call_avg[t]]
            qw = [W[d] * q[d] for d in range(169)]
            tq = sum(qw)
            cprob[t] = tq / Wt
            # for each pusher class a: sum_d qw[d] * eq[a][d] / tq
            if tq > 0:
                cval[t] = [sum(map(mul, eq[a], qw)) / tq for a in range(169)]
            else:
                cval[t] = [0.5] * 169
        br = []
        for a in range(169):
            ev = 0.0
            reach = 1.0
            for t, p_j in callers:
                q = cprob[t]
                pot = 2 * S + blinds_total - p_j - pusher_posted
                e = cval[t][a]
                ev_sd = e * _u(pot - S + pusher_posted, lam) + (1 - e) * _u(-(S - pusher_posted), lam)
                ev += reach * q * ev_sd
                reach *= (1 - q)
            ev += reach * _u(blinds_total, lam)
            br.append(1.0 if ev > 0.0 else 0.0)
        push = br
        for a in range(169):
            push_avg[a] += br[a]
    push_f = [x / iters for x in push_avg]
    call_f = {t: [x / iters for x in call_avg[t]] for t in types}
    return push_f, call_f


def _enc(v):
    return "".join(str(min(9, int(round(x * 9)))) for x in v)


def _job(args):
    n, S, lam, iters = args
    push, call = solve(n, S, lam, iters)
    return n, S, lam, _enc(push), {t: _enc(v) for t, v in call.items()}


def build(variant_name="nlhe", iters=300, workers=None, out=None, log=print):
    t0 = time.time()
    jobs = [(n, S, lam, iters) for n in NS for S in STACKS for lam in LAMS]
    res = {"stacks": STACKS, "lams": LAMS, "push": {}, "call": {}}
    with Pool(workers or os.cpu_count(), initializer=_init, initargs=(variant_name,)) as pool:
        for n, S, lam, push, call in pool.imap_unordered(_job, jobs):
            key = "%d|%s|%s" % (n, S, lam)
            res["push"][key] = push
            res["call"][key] = call
    res["build_seconds"] = round(time.time() - t0, 1)
    out = out or os.path.join(DATA_DIR, "pushfold_%s.json" % variant_name)
    with open(out, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    log("wrote %s in %.1fs" % (out, time.time() - t0))
    return out


