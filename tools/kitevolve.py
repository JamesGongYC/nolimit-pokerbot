"""Evolutionary population training on the starter kit's engine (run with the kit venv).

Same paradigm and decisions as pokerbot/evolve.py (docs/EVOLUTION.md):
  * families = distinct mechanisms, each with its own CEM distribution in
    normalised parameter space (here: mc = Monte Carlo equity, rule = hand
    categories, jam = push/fold)
  * each generation every candidate plays mixed tables: 1 candidate + 7
    opponents, alternating 4 hall-of-fame / 3 generator and 3 / 4 seats; HoF
    seats are drawn from the current TOP CYCLE (Smith set) of the HoF win
    graph, capped at 16 by Copeland score; generator seats are fresh random
    parameter draws across all families plus baseline bots; no current peers
  * fitness = plain mean official points; all candidates share the tables (CRN)
  * every generation each family's best is promoted (frozen snapshot)
  * equilibrium update: new members duel (4 v 4) the current group and each other
  * progress: new bests vs the previous generation's bests (fresh seeds) and vs
    frozen anchors (generation-0 seeds + baselines, fixed seeds)
  * no holdout; the verdict plays a shortlist (top K per family + extras) in the
    official all-vs-all format
usage:
  starter-kit/.venv/bin/python tools/kitevolve.py run --name main --generations 40 --minutes 45
  starter-kit/.venv/bin/python tools/kitevolve.py verdict --name main --shortlist 3 --iterations 300
  starter-kit/.venv/bin/python tools/kitevolve.py export --name main --member mc_g12 --out bots/pick
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitsim import BASE_FIELD, Pool, evaluate, mean_ci, run_table, lat_summary  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILIES = {
    "mc": {"path": "bots/jj_v2/agent.py", "space": {
        "open_ratio": (0.8, 1.8), "pos_bonus": (0.0, 0.5), "open_size": (2.0, 4.0),
        "threebet_ratio": (1.3, 2.6), "threebet_mult": (2.2, 4.0), "call_margin": (-0.05, 0.15),
        "push_bb": (4.0, 20.0), "push_ratio": (0.9, 1.8), "call_push_margin": (-0.05, 0.15),
        "value_eq": (0.5, 0.85), "raise_eq": (0.6, 0.95), "cbet_rate": (0.0, 1.0), "bluff_rate": (0.0, 0.5),
        "value_size": (0.3, 1.3), "bluff_size": (0.25, 1.0), "icm_weight": (0.0, 2.5), "adapt": (0.0, 1.0),
        "multiway_pow": (0.3, 1.5)}},
    "rule": {"path": "bots/jj_rule/agent.py", "space": {
        "open_th": (0.4, 0.85), "pos_bonus": (0.0, 0.3), "open_size": (2.0, 4.0), "call_th": (0.5, 0.9),
        "threebet_th": (0.7, 1.0), "push_bb": (4.0, 20.0), "push_th": (0.45, 0.9), "call_push_th": (0.5, 0.95),
        "bet_cat": (0.5, 3.0), "call_cat": (0.0, 2.5), "raise_cat": (1.5, 4.5), "big_bet_extra": (0.0, 1.5),
        "cbet_rate": (0.0, 1.0), "bluff_rate": (0.0, 0.5), "value_size": (0.3, 1.3), "bluff_size": (0.25, 1.0)}},
    "gbdt": {"path": "bots/jj_tree/agent.py", "space": None, "fixed": {"model": "model.json", "eps": 0.0}},
    "jam": {"path": "bots/jj_jam/agent.py", "space": {
        "push_th": (0.4, 0.95), "call_th": (0.45, 0.98), "ref_bb": (3.0, 40.0), "stack_exp": (0.0, 2.0),
        "post_cat": (1.0, 4.0)}},
}
DEFAULT_SEEDS = {  # generation-0 centres (the hand-set defaults of each family)
    "gbdt": None,  # = the MC seed + margin (filled below)
    "mc": {"open_ratio": 1.18, "pos_bonus": 0.22, "open_size": 2.5, "threebet_ratio": 1.75, "threebet_mult": 3.0,
           "call_margin": 0.04, "push_bb": 12.0, "push_ratio": 1.25, "call_push_margin": 0.03, "value_eq": 0.62,
           "raise_eq": 0.8, "cbet_rate": 0.55, "bluff_rate": 0.12, "value_size": 0.66, "bluff_size": 0.5,
           "icm_weight": 1.0, "adapt": 0.5, "multiway_pow": 1.0},
    "rule": {"open_th": 0.62, "pos_bonus": 0.12, "open_size": 2.5, "call_th": 0.72, "threebet_th": 0.86,
             "push_bb": 12.0, "push_th": 0.66, "call_push_th": 0.74, "bet_cat": 1.6, "call_cat": 1.2,
             "raise_cat": 2.6, "big_bet_extra": 0.6, "cbet_rate": 0.5, "bluff_rate": 0.1, "value_size": 0.66,
             "bluff_size": 0.5},
    "jam": {"push_th": 0.7, "call_th": 0.78, "ref_bb": 15.0, "stack_exp": 0.5, "post_cat": 2.0},
}
FAMILIES["gbdt"]["space"] = dict(FAMILIES["mc"]["space"], margin=(0.0, 1.5))
# top-clone family: the top teams' shared playbook (tools/profile.py), parameterised
FAMILIES["clone"] = {"path": "bots/jj_clone/agent.py", "space": {
    "open_ep": (0.08, 0.5), "open_mp": (0.1, 0.6), "open_co": (0.15, 0.7), "open_btn": (0.3, 0.95),
    "open_sb": (0.3, 0.95), "open_size": (2.0, 3.5), "threebet_pct": (0.01, 0.12), "threebet_mult": (2.5, 4.0),
    "call_open_pct": (0.03, 0.3), "call_max_frac": (0.05, 0.4), "fourbet_pct": (0.005, 0.05),
    "call_3bet_pct": (0.01, 0.1), "push_bb": (5.0, 20.0), "push_k": (1.0, 6.0), "call_push_k": (0.8, 4.0),
    "cbet": (0.3, 1.0), "cbet_size": (0.3, 1.0), "value_cat": (1.0, 2.6), "value_size": (0.4, 1.2),
    "bluff_turn": (0.0, 0.5), "bluff_river": (0.0, 0.4), "call_cat": (0.6, 2.0), "big_call_cat": (1.0, 3.0),
    "big_bet": (0.4, 1.2), "raise_cat": (2.0, 5.0), "multiway_pen": (0.0, 0.8)}}
DEFAULT_SEEDS["clone"] = {
    "open_ep": 0.24, "open_mp": 0.32, "open_co": 0.42, "open_btn": 0.65, "open_sb": 0.70, "open_size": 2.5,
    "threebet_pct": 0.04, "threebet_mult": 3.2, "call_open_pct": 0.13, "call_max_frac": 0.15, "fourbet_pct": 0.02,
    "call_3bet_pct": 0.045, "push_bb": 12.0, "push_k": 3.0, "call_push_k": 1.8, "cbet": 0.82, "cbet_size": 0.6,
    "value_cat": 1.4, "value_size": 0.66, "bluff_turn": 0.2, "bluff_river": 0.1, "call_cat": 1.0,
    "big_call_cat": 1.4, "big_bet": 0.75, "raise_cat": 3.0, "multiway_pen": 0.3}
DEFAULT_SEEDS["gbdt"] = dict(DEFAULT_SEEDS["mc"], margin=0.35)

GEN_BASELINES = {
    "checkfold": {}, "minraise": {}, "station": {"raise_st": (0.7, 0.98)},
    "jammer": {"push_pct": (0.05, 0.6)}, "random": {"raise": (0.0, 0.4), "fold": (0.0, 0.6)},
}
ANCHOR_BASELINES = BASE_FIELD


def space_names(fam):
    return list(FAMILIES[fam]["space"])


def to_params(fam, u):
    sp = FAMILIES[fam]["space"]
    return {k: round(lo + min(1.0, max(0.0, x)) * (hi - lo), 4) for (k, (lo, hi)), x in zip(sp.items(), u)}


def to_u(fam, params):
    sp = FAMILIES[fam]["space"]
    return [min(1.0, max(0.0, (params[k] - lo) / float(hi - lo))) for k, (lo, hi) in sp.items()]


_FIXED = {}  # per family fixed (non-evolved) params, e.g. the current tree model file


def spec(fam, params, name):
    p = dict(params)
    p.update(_FIXED.get(fam) or FAMILIES[fam].get("fixed") or {})
    return {"path": os.path.join(ROOT, FAMILIES[fam]["path"]), "params": p, "name": name}


def top_cycle(members, beats):
    """Smith set: union of source strongly connected components (Tarjan)."""
    idx = {m: i for i, m in enumerate(members)}
    n = len(members)
    adj = [[j for j in range(n) if j != i and beats(members[i], members[j])] for i in range(n)]
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    index, low, onstack, stack, comp = [0], [-1] * n, [False] * n, [], [-1] * n
    order = [-1] * n
    comps = []

    def strong(v):
        order[v] = low[v] = index[0]
        index[0] += 1
        stack.append(v)
        onstack[v] = True
        for w in adj[v]:
            if order[w] < 0:
                strong(w)
                low[v] = min(low[v], low[w])
            elif onstack[w]:
                low[v] = min(low[v], order[w])
        if low[v] == order[v]:
            c = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp[w] = len(comps)
                c.append(w)
                if w == v:
                    break
            comps.append(c)

    for v in range(n):
        if order[v] < 0:
            strong(v)
    has_in = [False] * len(comps)
    for v in range(n):
        for w in adj[v]:
            if comp[v] != comp[w]:
                has_in[comp[w]] = True
    out = [members[v] for c, cs in enumerate(comps) if not has_in[c] for v in cs]
    return sorted(out, key=lambda m: idx[m])


class KitEvolution(object):
    def __init__(self, name, pop=6, tables=60, duel_tables=6, progress_tables=24, group_cap=16, seed=2026,
                 workers=7, init_sigma=0.25, families=None, log=print):
        self.name = name
        self.dir = os.path.join(ROOT, "results", "kitevo_%s" % name)
        self.cfg = {"pop": pop, "tables": tables, "duel_tables": duel_tables, "progress_tables": progress_tables,
                    "group_cap": group_cap, "seed": seed, "init_sigma": init_sigma,
                    "families": families or list(FAMILIES)}
        self.workers = workers
        self.log = log
        self.state = None

    def path(self, f):
        return os.path.join(self.dir, f)

    def save(self):
        tmp = self.path("state.json.tmp")
        json.dump(self.state, open(tmp, "w"))
        os.replace(tmp, self.path("state.json"))

    # ---------------------------------------------------------- opponents
    def generator_spec(self, rng):
        kinds = list(self.state["families"]) + list(GEN_BASELINES)
        k = rng.choice(kinds)
        if k in GEN_BASELINES:
            return {"builtin": k, "params": {p: round(rng.uniform(lo, hi), 4) for p, (lo, hi) in GEN_BASELINES[k].items()},
                    "name": "gen_" + k}
        return spec(k, to_params(k, [rng.random() for _ in space_names(k)]), "gen_" + k)

    def pool_plan(self, n_tables, seed):
        rng = random.Random(seed)
        group = [self.state["hof"][m]["spec"] for m in self.state["group"]]
        plan = []
        for t in range(n_tables):
            n_hof = 4 if t % 2 == 0 else 3
            opps = [rng.choice(group) for _ in range(n_hof)] + [self.generator_spec(rng) for _ in range(7 - n_hof)]
            rng.shuffle(opps)
            plan.append((opps, rng.randrange(1 << 30), t % 8))
        return plan

    def fixed_plan(self, specs, n_tables, seed):
        rng = random.Random(seed)
        return [([rng.choice(specs) for _ in range(7)], rng.randrange(1 << 30), t % 8) for t in range(n_tables)]

    # -------------------------------------------------------------- duels
    def beats(self, a, b):
        e = self.state["edges"].get(a + "|" + b)
        if e is not None:
            return e[0] >= e[1]
        e = self.state["edges"].get(b + "|" + a)
        if e is not None:
            return e[1] >= e[0]
        return False

    def duels(self, p, pairs, salt):
        hof = self.state["hof"]
        jobs, meta = [], []
        for pi, (a, b) in enumerate(pairs):
            sa, sb = hof[a]["spec"], hof[b]["spec"]
            for t in range(self.cfg["duel_tables"]):
                first = t % 2 == 0
                lineup = [(sa if (k % 2 == 0) == first else sb) for k in range(8)]
                seed = (self.cfg["seed"] * 1000003 + salt * 7919 + pi * 104729 + t) & 0x3FFFFFFF
                jobs.append((lineup, seed, {}))
                meta.append((pi, [k for k in range(8) if (k % 2 == 0) == first]))
        if not jobs:
            return
        res = p.map(run_table, jobs, chunksize=max(1, len(jobs) // 64))
        acc = [[0.0, 0.0] for _ in pairs]
        for (pi, a_seats), r in zip(meta, res):
            pa = sum(r["points"][k] for k in a_seats)
            acc[pi][0] += pa
            acc[pi][1] += sum(r["points"]) - pa
        for (a, b), (pa, pb) in zip(pairs, acc):
            self.state["edges"][a + "|" + b] = [round(pa, 2), round(pb, 2)]

    def capped_cycle(self, members):
        cyc = top_cycle(members, self.beats)
        cap = self.cfg["group_cap"]
        if not cap or len(cyc) <= cap:
            return cyc

        def score(m):
            wins = margin = 0.0
            for o in cyc:
                if o == m:
                    continue
                e = self.state["edges"].get(m + "|" + o)
                if e is None:
                    e2 = self.state["edges"].get(o + "|" + m)
                    e = [e2[1], e2[0]] if e2 else [0.0, 0.0]
                wins += 1.0 if e[0] > e[1] else (0.5 if e[0] == e[1] else 0.0)
                margin += e[0] - e[1]
            return (wins, margin)
        keep = set(sorted(cyc, key=score, reverse=True)[:cap])
        return [m for m in cyc if m in keep]

    # --------------------------------------------------------- lifecycle
    def load_or_init(self, p):
        os.makedirs(self.dir, exist_ok=True)
        if os.path.exists(self.path("state.json")):
            self.state = json.load(open(self.path("state.json")))
            want = self.cfg["families"]
            self.cfg.update(self.state.get("cfg_locked", {}))
            self.cfg["families"] = want
            for fam, fx in self.state.get("fixed", {}).items():
                _FIXED[fam] = fx
            added = [f for f in want if f not in self.state["families"]]
            for fam in added:
                params = self.seed_params(fam)
                name = "%s_g%d" % (fam, self.state["gen"] - 1)
                self.state["families"][fam] = {"mu": to_u(fam, params), "sigma": [self.cfg["init_sigma"]] * len(params)}
                self.state["hof"][name] = {"spec": spec(fam, params, name), "family": fam, "gen": self.state["gen"] - 1,
                                           "fitness": None}
                group = list(self.state["group"])
                self.duels(p, [(name, b) for b in group], 10 ** 6 + self.state["gen"])
                self.state["group"] = self.capped_cycle(group + [name])
                self.state["anchors"].append(name)
            self.state["cfg_locked"] = dict(self.cfg)
            self.save()
            self.log("resumed %s at generation %d (hof %d, group %d)%s" % (
                self.name, self.state["gen"], len(self.state["hof"]), len(self.state["group"]),
                (" added families %s" % added) if added else ""))
            return
        fams, hof = {}, {}
        for fam in self.cfg["families"]:
            params = self.seed_params(fam)
            name = "%s_g0" % fam
            fams[fam] = {"mu": to_u(fam, params), "sigma": [self.cfg["init_sigma"]] * len(params)}
            hof[name] = {"spec": spec(fam, params, name), "family": fam, "gen": 0, "fitness": None}
        self.state = {"gen": 1, "families": fams, "hof": hof, "edges": {}, "group": [], "prev_bests": list(hof),
                      "anchors": list(hof), "history": [], "cfg_locked": dict(self.cfg), "wall_s": 0.0}
        members = list(hof)
        self.duels(p, [(a, b) for i, a in enumerate(members) for b in members[i + 1:]], 0)
        self.state["group"] = self.capped_cycle(members)
        self.save()
        self.log("seeded hof %s, group %s" % (members, self.state["group"]))

    def seed_params(self, fam):
        """Generation-0 centre; the GBDT family starts from the current MC champion."""
        if fam == "gbdt" and self.state:
            best = max(((h["fitness"], m) for m, h in self.state["hof"].items()
                        if h["family"] == "mc" and h.get("fitness") is not None), default=None)
            if best:
                mp = self.state["hof"][best[1]]["spec"]["params"]
                return {k: mp.get(k, DEFAULT_SEEDS["gbdt"][k]) for k in FAMILIES["gbdt"]["space"]}
        return dict(DEFAULT_SEEDS[fam])

    def retrain(self, p, g):
        """Collect around the MC champion, refit, and adopt only if the gate passes."""
        import kittrees as KT
        params, champ = KT.champion_params(self.state)
        tag = "%s_g%d" % (self.name, g)
        path = os.path.join(KT.DATA, "%s_%s.bin" % (self.name, tag))
        os.makedirs(KT.DATA, exist_ok=True)
        n = KT.collect(p, self, params, self.cfg.get("retrain_tables", 1500), 9000 + g, path)
        files = sorted((os.path.join(KT.DATA, f) for f in os.listdir(KT.DATA)
                        if f.startswith(self.name + "_") and f.endswith(".bin")), key=os.path.getmtime)[-2:]
        model_rel = "model_%s.json" % tag
        KT.fit(files, os.path.join(KT.TREE_DIR, model_rel))
        cur = (_FIXED.get("gbdt") or FAMILIES["gbdt"]["fixed"])["model"]
        new_m, old_m, (dm, dci) = KT.gate(p, self, params, model_rel, self.cfg.get("gate_tables", 200), 777 + g)
        # gate against the CURRENT model (not just no model)
        res = evaluate(p, [{"path": KT.TREE, "params": dict(params, eps=0.0, model=model_rel)},
                           {"path": KT.TREE, "params": dict(params, eps=0.0, model=cur)}],
                       self.pool_plan(self.cfg.get("gate_tables", 200), 778 + g))
        diff = [a - b for a, b in zip(res[0]["points"], res[1]["points"])]
        dmc, dcc = mean_ci(diff)
        adopt = dmc > 0
        if adopt:
            _FIXED["gbdt"] = {"model": model_rel, "eps": 0.0}
            self.state.setdefault("fixed", {})["gbdt"] = _FIXED["gbdt"]
        rec = {"gen": g, "rows": n, "champion": champ, "new_vs_current": round(dmc, 2), "ci": round(dcc, 2),
               "new_vs_none": round(dm, 2), "model": model_rel, "adopted": adopt}
        with open(self.path("trees.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.log("  retrain g%d: %d rows around %s; new vs current %+.2f +- %.2f -> %s" % (
            g, n, champ, dmc, dcc, "ADOPT " + model_rel if adopt else "reject"))

    def generation(self, p):
        st, cfg = self.state, self.cfg
        g = st["gen"]
        t0 = time.time()
        if "gbdt" in st["families"] and cfg.get("retrain_every", 8) and g % cfg.get("retrain_every", 8) == 0:
            try:
                self.retrain(p, g)
            except Exception as e:
                self.log("  retrain g%d failed: %r" % (g, e))
        rng = random.Random(cfg["seed"] * 1000003 + g)  # per-generation RNG: resumes are identical
        cands, owner = [], []
        for fam, f in st["families"].items():
            vecs = [list(f["mu"])] + [[min(1.0, max(0.0, m + s * rng.gauss(0, 1))) for m, s in zip(f["mu"], f["sigma"])]
                                      for _ in range(cfg["pop"] - 1)]
            for i, u in enumerate(vecs):
                cands.append(spec(fam, to_params(fam, u), "%s_g%d_c%d" % (fam, g, i)))
                owner.append((fam, u))
        plan = self.pool_plan(cfg["tables"], cfg["seed"] * 7777 + g)
        res = evaluate(p, cands, plan)
        scores = {}
        lat = []
        for (fam, u), c, r in zip(owner, cands, res):
            scores.setdefault(fam, []).append((sum(r["points"]) / len(r["points"]), u, c, r["failures"]))
            lat += r["lat"]
        new, fit = [], {}
        for fam, lst in scores.items():
            lst.sort(key=lambda x: -x[0])
            el = lst[:max(2, cfg["pop"] // 3)]
            f = st["families"][fam]
            d = len(f["mu"])
            f["mu"] = [sum(e[1][i] for e in el) / len(el) for i in range(d)]
            f["sigma"] = [min(0.3, max(0.03, 1.1 * math.sqrt(sum((e[1][i] - f["mu"][i]) ** 2 for e in el) / len(el))))
                          for i in range(d)]
            m, u, c, fails = lst[0]
            name = "%s_g%d" % (fam, g)
            s2 = dict(c)
            s2["name"] = name
            st["hof"][name] = {"spec": s2, "family": fam, "gen": g, "fitness": round(m, 3), "failures": fails}
            new.append(name)
            fit[fam] = round(m, 2)
        group = list(st["group"])
        pairs = [(a, b) for a in new for b in group] + [(a, b) for i, a in enumerate(new) for b in new[i + 1:]]
        self.duels(p, pairs, g)
        st["group"] = self.capped_cycle(group + new)
        new_specs = [st["hof"][m]["spec"] for m in new]
        prev = [st["hof"][m]["spec"] for m in st["prev_bests"]]
        anchors = [st["hof"][m]["spec"] for m in st["anchors"]] + list(ANCHOR_BASELINES)
        pr = evaluate(p, new_specs, self.fixed_plan(prev, cfg["progress_tables"], cfg["seed"] * 13 + g))
        an = evaluate(p, new_specs, self.fixed_plan(anchors, cfg["progress_tables"], cfg["seed"] * 17))
        prog = {st["hof"][m]["family"]: round(sum(a["points"]) / len(a["points"]), 2) for m, a in zip(new, pr)}
        anch = {st["hof"][m]["family"]: round(sum(b["points"]) / len(b["points"]), 2) for m, b in zip(new, an)}
        st["prev_bests"] = new
        wall = time.time() - t0
        st["wall_s"] += wall
        row = {"gen": g, "wall_s": round(wall, 1), "fitness": fit, "vs_prev_bests": prog, "vs_anchors": anch,
               "group": list(st["group"]), "hof": len(st["hof"]), "lat": lat_summary(sorted(lat))}
        st["history"].append(row)
        st["gen"] = g + 1
        self.save()
        with open(self.path("log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        self.log("gen %3d %5.1fs fit %s vs-prev %s vs-anchor %s group %d %s p99 %sus" % (
            g, wall, fit, prog, anch, len(st["group"]), sorted({m.rsplit("_g", 1)[0] for m in st["group"]}),
            row["lat"].get("p99_us")))

    def run(self, generations, minutes):
        t_start = time.time()
        with Pool(self.workers) as p:
            self.load_or_init(p)
            while self.state["gen"] <= generations:
                if minutes and time.time() - t_start > minutes * 60:
                    self.log("time budget reached")
                    break
                self.generation(p)


# ---------------------------------------------------------------- verdict
def official(p, entries, iterations, seed):
    """All-vs-all official format: shuffle all entries into tables of 8; the short
    last table is filled with entries not seated at it (fillers never score)."""
    rng = random.Random(seed)
    n = len(entries)
    n_tables = int(math.ceil(n / 8.0))
    jobs, meta = [], []
    for it in range(iterations):
        order = list(range(n))
        rng.shuffle(order)
        for t in range(n_tables):
            tab = [(e, True) for e in order[t * 8:(t + 1) * 8]]
            if len(tab) < 8:
                here = {e for e, _ in tab}
                pool_ = [e for e in range(n) if e not in here]
                tab += [(e, False) for e in rng.sample(pool_, min(8 - len(tab), len(pool_)))]
                rng.shuffle(tab)
            jobs.append(([entries[e] for e, _ in tab], rng.randrange(1 << 30), {}))
            meta.append(tab)
    res = p.map(run_table, jobs, chunksize=max(1, len(jobs) // 64))
    pts = [[] for _ in range(n)]
    for tab, r in zip(meta, res):
        for seat, (e, scored) in enumerate(tab):
            if scored:
                pts[e].append(r["points"][seat])
    return pts


def verdict(name, shortlist, iterations, extra, workers, log=print):
    d = os.path.join(ROOT, "results", "kitevo_%s" % name)
    st = json.load(open(os.path.join(d, "state.json")))
    by = {}
    for m, h in st["hof"].items():
        by.setdefault(h["family"], []).append((h["fitness"] if h["fitness"] is not None else -1, m))
    names = []
    for fam, lst in by.items():
        lst.sort(reverse=True)
        names += [m for _, m in lst[:shortlist]] if shortlist else [m for _, m in lst]
    entries = [st["hof"][m]["spec"] for m in names]
    for i, e in enumerate(extra):
        entries.append(e)
        names.append(e.get("name", "extra%d" % i))
    with Pool(workers) as p:
        pts = official(p, entries, iterations, 99173)
    rows = sorted(((mean_ci(x)[0], mean_ci(x)[1], n) for x, n in zip(pts, names)), reverse=True)
    L = ["# Verdict (%s): %d entries, %d iterations, official all-vs-all format\n" % (name, len(names), iterations),
         "| # | entry | mean points/table | ±95% |", "|---|---|---|---|"]
    for k, (m, ci, n) in enumerate(rows):
        L.append("| %d | %s | %.2f | %.2f |" % (k + 1, n, m, ci))
    md = "\n".join(L) + "\n"
    open(os.path.join(d, "verdict.md"), "w").write(md)
    log(md)
    return rows


def export(name, member, out, state=None):
    """Write a self-contained bot folder for `member`: the family's whole folder
    (agent.py + sibling modules + model files) with the evolved parameters
    baked into agent.py."""
    st = state or json.load(open(os.path.join(ROOT, "results", "kitevo_%s" % name, "state.json")))
    sp = st["hof"][member]["spec"] if member in st["hof"] else member
    src_dir = os.path.dirname(sp["path"])
    if os.path.isdir(out):
        shutil.rmtree(out)
    shutil.copytree(src_dir, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "data_*"))
    agent = os.path.join(out, "agent.py")
    src = open(agent).read()
    src += "\n\n# evolved parameters: %s (%s)\nAgent.P = dict(Agent.P, **%s)\n" % (
        sp.get("name", member), name, json.dumps(sp.get("params") or {}))
    open(agent, "w").write(src)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "verdict", "export"])
    ap.add_argument("--name", default="main")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--tables", type=int, default=60)
    ap.add_argument("--duel-tables", type=int, default=6)
    ap.add_argument("--progress-tables", type=int, default=24)
    ap.add_argument("--group-cap", type=int, default=16)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--families", default=None)
    ap.add_argument("--shortlist", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--extra", default="", help="comma list of agent.py paths added to the verdict")
    ap.add_argument("--member", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        fams = a.families.split(",") if a.families else None
        KitEvolution(a.name, a.pop, a.tables, a.duel_tables, a.progress_tables, a.group_cap, workers=a.workers,
                     families=fams).run(a.generations, a.minutes)
    elif a.cmd == "verdict":
        extra = [{"path": os.path.abspath(x), "name": os.path.basename(os.path.dirname(os.path.abspath(x)))}
                 for x in a.extra.split(",") if x]
        verdict(a.name, a.shortlist, a.iterations, extra, a.workers)
    else:
        print("wrote", export(a.name, a.member, a.out))


if __name__ == "__main__":
    main()
