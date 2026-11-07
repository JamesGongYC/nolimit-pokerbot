"""Fast in-process simulator on the starter kit's real engine (run with the kit's venv).

Tables are the kit's own `SitAndGo` with its variant, rules and `table_points`
scoring; bots are imported once per worker and called directly (no subprocess
wire), with every `act` timed. Entries are dicts:
  {"path": ".../agent.py", "params": {...}, "name": "x"}   our parameterised bots
  {"builtin": "station" | "jammer" | "random" | "checkfold" | "minraise", "params": {...}}

usage (from the repo root):
  starter-kit/.venv/bin/python tools/kitsim.py --hero bots/jj_v2/agent.py --tables 400
"""
import argparse
import asyncio
import importlib.util
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

from poker_agent_sdk import Action
from poker_engine.scoring import table_places, table_points
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import load_variant

_MODS = {}
_VARIANT = None
RANKS = "23456789TJQKA"


def load_module(path):
    path = os.path.abspath(path)
    m = _MODS.get(path)
    if m is None:
        name = "agent_%d" % len(_MODS)
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        _MODS[path] = m
    return m


# ---------------------------------------------------------------- baselines
def _pct3(cards):
    """Crude strength 0..1 of a 2-3 card hand for baselines."""
    rs = sorted((RANKS.index(c[0]) for c in cards), reverse=True)
    s = rs[0] * 2 + rs[1]
    if len(set(rs)) < len(rs):
        s += 14
    if len({c[1] for c in cards}) < len(cards):
        s += 3
    return min(1.0, s / 52.0)


class Builtin(object):
    def __init__(self, kind, params, seed):
        self.kind = kind
        self.p = params or {}
        self.rng = random.Random(seed)

    def act(self, state, memory):
        if state.phase != "betting":
            return Action("select", cards=(2, 0)), memory
        k = self.kind
        r = state.can("raise")
        st = _pct3(state.cards)
        if k == "checkfold":
            return (Action.check() if state.can("check") else Action.fold()), memory
        if k == "minraise":
            if r:
                return Action.raise_to(r.min_amount), memory
            return (Action.call() if state.can("call") else Action.check()), memory
        if k == "station":
            if r and st > self.p.get("raise_st", 0.85):
                return Action.raise_to(r.min_amount), memory
            return (Action.call() if state.can("call") else Action.check()), memory
        if k == "jammer":
            if not state.board:
                if st >= 1.0 - self.p.get("push_pct", 0.3):
                    if r:
                        return Action.raise_to(r.max_amount), memory
                    return Action.call(), memory
                return (Action.check() if state.can("check") else Action.fold()), memory
            return (Action.check() if state.can("check") else Action.call()), memory
        # random
        x = self.rng.random()
        if r and x < self.p.get("raise", 0.15):
            return Action.raise_to(self.rng.randint(r.min_amount, r.max_amount)), memory
        if x < self.p.get("raise", 0.15) + self.p.get("fold", 0.3) and state.can("fold"):
            return Action.fold(), memory
        return (Action.call() if state.can("call") else Action.check()), memory


def make_bot(entry, seed):
    if "builtin" in entry:
        return Builtin(entry["builtin"], entry.get("params"), seed)
    m = load_module(entry["path"])
    a = m.Agent()
    if entry.get("params"):
        a.P = dict(m.Agent.P, **entry["params"]) if hasattr(m.Agent, "P") else entry["params"]
    if hasattr(a, "seed"):
        a.seed(seed)
    return a


# ---------------------------------------------------------------- one table
def run_table(job):
    lineup, seed, rules_kw = job
    global _VARIANT
    kw = dict(rules_kw)
    kw.setdefault("variant", "flop_discard_reveal")
    rules = TableRules(players=len(lineup), **kw)
    if _VARIANT is None:
        _VARIANT = load_variant(rules.variant)
    table = SitAndGo(rules=rules, variant=_VARIANT, table_id="sim-%s" % seed, seed=str(seed))
    bots = [make_bot(e, hash((seed, i)) & 0xFFFFFFF) for i, e in enumerate(lineup)]
    memory = [b""] * len(bots)
    failures = [0] * len(bots)
    decisions = [0] * len(bots)
    lat = [[] for _ in bots]
    limit_ms = rules.decision_ms
    # behaviour counters per seat, same definitions as tools/hands.py
    beh = [dict(hands=0, vpip=0, pfr=0, pf_allin=0, pf_dec=0, pf_fold=0, post_dec=0, post_aggr=0,
                post_faced=0, post_fold=0) for _ in bots]
    flags = {}
    while True:
        state = table.decision()
        if state is None:
            break
        s = state.seat
        decisions[s] += 1
        t0 = time.perf_counter()
        try:
            action, mem = bots[s].act(state, memory[s])
            ok = True
        except Exception:
            action, mem, ok = None, memory[s], False
        dt = (time.perf_counter() - t0) * 1000.0
        lat[s].append(dt)
        if ok:
            memory[s] = mem[: rules.memory_bytes] if isinstance(mem, (bytes, bytearray)) else b""
        if ok and action is not None and state.phase == "betting":
            b = beh[s]
            k = action.kind
            key = (s, state.hand_number)
            if not state.board:
                if key not in flags:
                    flags[key] = [False, False, False]
                    b["hands"] += 1
                f = flags[key]
                b["pf_dec"] += 1
                if k == "fold":
                    b["pf_fold"] += 1
                if k in ("call", "raise") and not f[0]:
                    f[0] = True
                    b["vpip"] += 1
                if k == "raise":
                    if not f[1]:
                        f[1] = True
                        b["pfr"] += 1
                    r = state.can("raise")
                    if r and action.amount >= r.max_amount and not f[2]:
                        f[2] = True
                        b["pf_allin"] += 1
            else:
                b["post_dec"] += 1
                if k == "raise":
                    b["post_aggr"] += 1
                if state.to_call > 0:
                    b["post_faced"] += 1
                    if k == "fold":
                        b["post_fold"] += 1
        if not table.apply(action, answered=ok, bank_ms=max(0.0, dt - limit_ms)):
            failures[s] += 1
    res = table.result(tuple(failures), tuple(decisions))
    pts = table_points(res, table.payouts)
    places = table_places(res)
    exp = []
    for b in bots:
        if hasattr(b, "on_table_end"):
            try:
                b.on_table_end(res.final_stacks)
            except Exception:
                pass
        exp.append(getattr(b, "experience", None) or None)
    return {"points": pts, "places": places, "failures": failures, "decisions": decisions,
            "lat": [sorted(x) for x in lat], "hands": res.hands_played, "experience": exp, "behavior": beh}


def pool(workers):
    return Pool(workers)


def matched_plan(n_tables, opponents, seed, size=8):
    """Opponent lineups + seeds + hero seat, shared by every candidate (CRN)."""
    rng = random.Random(seed)
    out = []
    for t in range(n_tables):
        opps = [rng.choice(opponents) for _ in range(size - 1)]
        out.append((opps, rng.randrange(1 << 30), t % size))
    return out


def evaluate(p, candidates, plan, rules_kw=None):
    rules_kw = rules_kw or {}
    jobs = []
    for c in candidates:
        for opps, seed, seat in plan:
            lineup = list(opps)
            lineup.insert(seat, c)
            jobs.append((lineup, seed, rules_kw))
    res = p.map(run_table, jobs, chunksize=max(1, len(jobs) // 64))
    out = []
    k = len(plan)
    for ci in range(len(candidates)):
        rows = res[ci * k:(ci + 1) * k]
        pts = [r["points"][seat] for r, (_, _, seat) in zip(rows, plan)]
        pl = [r["places"][seat] for r, (_, _, seat) in zip(rows, plan)]
        lat = sorted(x for r, (_, _, seat) in zip(rows, plan) for x in r["lat"][seat])
        fails = sum(r["failures"][seat] for r, (_, _, seat) in zip(rows, plan))
        exp = [r["experience"][seat] for r, (_, _, seat) in zip(rows, plan)]
        out.append({"points": pts, "places": pl, "lat": lat, "failures": fails, "experience": exp})
    return out


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("inf")
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(v / n)


def lat_summary(lat):
    if not lat:
        return {}
    q = lambda f: round(lat[min(len(lat) - 1, int(f * len(lat)))] * 1000, 1)
    return {"n": len(lat), "p50_us": q(0.5), "p99_us": q(0.99), "p999_us": q(0.999),
            "max_us": round(lat[-1] * 1000, 1), "over_1ms": sum(1 for x in lat if x > 1.0)}


BASE_FIELD = [
    {"builtin": "checkfold"}, {"builtin": "minraise"}, {"builtin": "station"},
    {"builtin": "jammer", "params": {"push_pct": 0.25}}, {"builtin": "jammer", "params": {"push_pct": 0.5}},
    {"builtin": "random"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", required=True)
    ap.add_argument("--params", default=None, help="JSON params for the hero")
    ap.add_argument("--field", default="mixed", help="mixed | self | base")
    ap.add_argument("--tables", type=int, default=200)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    hero = {"path": a.hero, "params": json.loads(a.params) if a.params else None, "name": "hero"}
    me = {"path": a.hero}
    if a.field == "base":
        field = BASE_FIELD
    elif a.field == "self":
        field = [me]
    else:
        field = BASE_FIELD + [me] * 6
    plan = matched_plan(a.tables, field, a.seed)
    t0 = time.time()
    with pool(a.workers) as p:
        r = evaluate(p, [hero], plan)[0]
    m, ci = mean_ci(r["points"])
    pl = r["places"]
    print(json.dumps({"points": round(m, 2), "ci95": round(ci, 2), "tables": len(pl),
                      "win%": round(100.0 * pl.count(1) / len(pl), 1), "failures": r["failures"],
                      "latency": lat_summary(r["lat"]), "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
