"""Simulation/search: learn a lookup table of action values from exploring
rollouts (Monte Carlo control over a state abstraction).

Each generation, learner seats play epsilon-greedy w.r.t. the current
table (falling back to the embedded EquityBot for unseen states), and
every visited (state, action) is credited with the hand's change in the
learner's ICM equity in official points. Tables are merged across
workers; epsilon decays. The final table is exported for LookupBot /
HybridBot and is bounded in size by the abstraction.
"""
import json
import os
import time

from .sim import Runner, load_fields, matched_jobs, evaluate_candidates
from .strategies.lookup import N_ACT


def _merge(acc, exp):
    if not exp:
        return
    for key, (qs, ns) in exp.items():
        row = acc.get(key)
        if row is None:
            acc[key] = [list(qs), list(ns)]
        else:
            for a in range(N_ACT):
                row[0][a] += qs[a]
                row[1][a] += ns[a]


def _table_from(acc):
    out = {}
    for key, (qs, ns) in acc.items():
        out[key] = [[round(q / n, 4) if n else 0.0 for q, n in zip(qs, ns)], ns]
    return out


def train_qtable(name="qtable", base_params=None, generations=6, tables=600, eps0=0.3, eps_decay=0.6,
                 res=2, fields_path=None, seed=5, workers=None, event=None, log=print, preflop_only=1):
    from .event import Event
    event = event or Event.load()
    fields = load_fields(fields_path or os.path.join("configs", "populations", "train.json"))
    out_dir = os.path.join("pokerbot", "data")
    acc = {}
    t0, cpu = time.time(), 0.0
    path = ""
    with Runner(event, workers) as runner:
        for g in range(generations):
            eps = eps0 * (eps_decay ** g)
            params = dict(base_params or {})
            params.update({"explore": max(eps, 0.02), "res": res, "table": path,
                           "preflop_only": preflop_only, "min_visits": 20})
            spec = {"type": "lookup", "name": "%s_learner" % name, "params": params}
            plan = matched_jobs(tables, fields, seed * 1000 + g)
            r = evaluate_candidates(runner, [spec], plan, collect=True)[0]
            cpu += r["cpu"]
            for e in r["experience"]:
                _merge(acc, e)
            path = os.path.join(out_dir, "%s_gen%d.json" % (name, g))
            json.dump({"table": _table_from(acc)}, open(path, "w"), separators=(",", ":"))
            m = sum(r["points"]) / len(r["points"])
            log("qtable gen %d eps %.3f states %d mean pts (exploring) %.2f %.0fs" % (
                g, eps, len(acc), m, time.time() - t0))
    final = os.path.join(out_dir, "%s.json" % name)
    meta = {"generations": generations, "tables": tables, "res": res, "train_cpu_s": round(cpu, 1),
            "wall_s": round(time.time() - t0, 1), "states": len(acc), "preflop_only": preflop_only}
    json.dump({"table": _table_from(acc), "meta": meta}, open(final, "w"), separators=(",", ":"))
    for g in range(generations):
        p = os.path.join(out_dir, "%s_gen%d.json" % (name, g))
        if os.path.exists(p):
            os.remove(p)
    log("wrote %s (%d states, %.0f KB)" % (final, len(acc), os.path.getsize(final) / 1024.0))
    return final, meta
