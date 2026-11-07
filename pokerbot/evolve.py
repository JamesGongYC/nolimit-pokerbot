"""Evolutionary population training with a top-cycle hall of fame.

Paradigm (decisions recorded in docs/EVOLUTION.md):
  * families: heuristic, equity, montecarlo, linear, allin, hybrid, gbdt, rf;
    each keeps its own CEM search distribution in normalised parameter space
  * each generation every candidate plays mixed tables: 1 candidate + 7
    opponents, alternating 4 hall-of-fame / 3 generator and 3 / 4 seats.
    Hall-of-fame seats are drawn uniformly from the current TOP CYCLE (Smith
    set) of the hall of fame; generator seats are fresh random parameter
    draws across all families and baseline bots. No current peers.
  * fitness = plain mean official points; all candidates of all families share
    the same tables (common random numbers)
  * every generation, each family's best is promoted (frozen snapshot)
  * equilibrium update: new members duel (4 v 4) the current group and each
    other; the top cycle of the resulting win graph is the new group, capped
    at `group_cap` members by Copeland score inside the cycle
  * progress: new bests vs the previous generation's bests (fresh seeds), and
    vs frozen anchors (generation-0 seeds + baselines, fixed seeds)
  * tree families (gbdt, rf): settings evolve every generation; the models
    are retrained every `retrain_every` generations from exploring data
    collected around the current gbdt champion
  * no holdout; the final verdict (evo-verdict) plays the whole hall of fame
    against itself: all-vs-all official format + 1-vs-7 cross-play
"""
import json
import math
import os
import random
import time

from .optimize import Space
from .sim import Runner, evaluate_candidates, mean_ci
from .strategies import resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEEDS = {
    "heuristic": "results/opt_heur/best.json",
    "equity": "results/opt_eq2/best.json",
    "montecarlo": "results/opt_mc/best.json",
    "linear": "results/opt_linear/best.json",
    "allin": "results/opt_jam/best.json",
    "hybrid": "results/opt_hybrid_h/best.json",
    "gbdt": "results/gbdt_variants/c40_4.json",
    "rf": "results/opt_rf/best.json",
}
SPACE_KIND = {"allin": "allin", "linear": "linear"}

# generator ranges for cheap baseline bots (uniform draws)
BASELINE_SPACES = {
    "random": {"raise": (0.0, 0.4), "fold": (0.0, 0.7)},
    "station": {"raise_eq": (0.6, 0.95), "fold_eq": (0.0, 0.4)},
    "maniac": {"aggr": (0.2, 0.9), "shove": (0.0, 0.5), "fold": (0.0, 0.5)},
    "nit": {"play_pct": (0.03, 0.2)},
    "potodds": {"raise_eq": (0.55, 0.9), "margin": (-0.1, 0.1)},
    "shover": {"push_pct": (0.05, 0.6), "call_pct": (0.03, 0.4)},
}
ANCHOR_BASELINES = ["station", "maniac", "nit", "potodds", "shover20", "random", "jam_nash", "heur_tag"]


def _load(path):
    return json.load(open(os.path.join(ROOT, path)))


def _clean(spec, name):
    s = json.loads(json.dumps(resolve(spec)))
    for k in ("train_cpu_s", "train_wall_s"):
        s.pop(k, None)
    s["name"] = name
    return s


# ------------------------------------------------------------ graph utils
def top_cycle(members, beats):
    """Smith set of the win graph restricted to `members`.
    beats(a, b) -> True if a beats (or ties) b. Union of source SCCs."""
    idx = {m: i for i, m in enumerate(members)}
    n = len(members)
    adj = [[j for j in range(n) if j != i and beats(members[i], members[j])] for i in range(n)]
    # Tarjan SCC
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


# ------------------------------------------------------------- evolution
class Evolution(object):
    def __init__(self, name="evo", pop=8, tables=150, duel_tables=24, progress_tables=80,
                 retrain_every=10, retrain_tables=6000, seed=2026, workers=None, event=None, log=print,
                 group_cap=16):
        from .event import Event
        self.event = event or Event.load()
        self.name = name
        self.dir = os.path.join(ROOT, "results", "evo_%s" % name)
        self.cfg = {"pop": pop, "tables": tables, "duel_tables": duel_tables, "progress_tables": progress_tables,
                    "retrain_every": retrain_every, "retrain_tables": retrain_tables, "seed": seed,
                    "group_cap": group_cap}
        self.workers = workers
        self.log = log
        self.rng = random.Random(seed)
        self.state = None

    # ------------------------------------------------------------ state
    def _path(self, f):
        return os.path.join(self.dir, f)

    def save(self):
        json.dump(self.state, open(self._path("state.json"), "w"))

    def load_or_init(self, runner):
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        if os.path.exists(self._path("state.json")):
            self.state = json.load(open(self._path("state.json")))
            self.cfg.update(self.state.get("cfg_locked", {}))
            self.log("resumed %s at generation %d (hall of fame %d, group %d)" % (
                self.name, self.state["gen"], len(self.state["hof"]), len(self.state["group"])))
            return
        fams = {}
        hof = {}
        for fam, path in SEEDS.items():
            spec = _clean(_load(path), "%s_g0" % fam)
            sp = Space(spec, SPACE_KIND.get(fam, "core"))
            fams[fam] = {"base": spec, "mu": sp.center(), "sigma": [0.15] * sp.dim}
            hof[spec["name"]] = {"spec": spec, "family": fam, "gen": 0, "fitness": None}
        self.state = {"gen": 0, "families": fams, "hof": hof, "edges": {}, "group": [], "prev_bests": list(hof),
                      "anchors": list(hof), "history": [], "tree_data": [os.path.join("results", "treedata", "gbdt_r2.bin")],
                      "cfg_locked": dict(self.cfg), "cpu_s": 0.0, "wall_s": 0.0}
        self.log("seeding hall of fame with %d family champions; dueling for the initial top cycle" % len(hof))
        members = list(hof)
        pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1:]]
        self._duels(runner, pairs, salt=0)
        self.state["group"] = self._capped_cycle(members)
        self.state["gen"] = 1
        self.save()

    # ---------------------------------------------------------- opponents
    def _space(self, fam):
        f = self.state["families"][fam]
        return Space(f["base"], SPACE_KIND.get(fam, "core"))

    def generator_spec(self, rng):
        kinds = list(SEEDS) + list(BASELINE_SPACES)
        k = rng.choice(kinds)
        if k in BASELINE_SPACES:
            params = {p: round(rng.uniform(lo, hi), 4) for p, (lo, hi) in BASELINE_SPACES[k].items()}
            return {"type": k, "name": "gen_%s" % k, "params": params}
        sp = self._spaces[k]
        return sp.spec([rng.random() for _ in range(sp.dim)], "gen_%s" % k)

    def pool_plan(self, n_tables, seed):
        """Mixed tables: alternating 4 HoF + 3 generator / 3 HoF + 4 generator."""
        rng = random.Random(seed)
        group = [self.state["hof"][m]["spec"] for m in self.state["group"]]
        plan = []
        size = self.event.table_size
        for t in range(n_tables):
            n_hof = 4 if t % 2 == 0 else 3
            opps = [rng.choice(group) for _ in range(n_hof)] + \
                   [self.generator_spec(rng) for _ in range(size - 1 - n_hof)]
            rng.shuffle(opps)
            plan.append((opps, rng.randrange(1 << 30), t % size, "pool"))
        return plan

    def fixed_plan(self, specs, n_tables, seed):
        rng = random.Random(seed)
        size = self.event.table_size
        return [([rng.choice(specs) for _ in range(size - 1)], rng.randrange(1 << 30), t % size, "fixed")
                for t in range(n_tables)]

    # -------------------------------------------------------------- duels
    def _beats(self, a, b):
        e = self.state["edges"].get(a + "|" + b)
        if e is not None:
            return e[0] >= e[1]
        e = self.state["edges"].get(b + "|" + a)
        if e is not None:
            return e[1] >= e[0]
        return False

    def _duels(self, runner, pairs, salt):
        """4 v 4 tables per pair; edge = (points of a, points of b)."""
        hof = self.state["hof"]
        size = self.event.table_size
        jobs, meta = [], []
        for pi, (a, b) in enumerate(pairs):
            sa, sb = hof[a]["spec"], hof[b]["spec"]
            for t in range(self.cfg["duel_tables"]):
                first = (t % 2 == 0)
                lineup = [(sa if (k % 2 == 0) == first else sb) for k in range(size)]
                seed = (self.cfg["seed"] * 1000003 + salt * 7919 + pi * 104729 + t) & 0x3FFFFFFF
                jobs.append((lineup, seed, None))
                meta.append((pi, [k for k in range(size) if (k % 2 == 0) == first]))
        if not jobs:
            return 0.0
        res = runner.run(jobs)
        acc = [[0.0, 0.0] for _ in pairs]
        cpu = 0.0
        for (pi, a_seats), r in zip(meta, res):
            pa = sum(r["points"][k] for k in a_seats)
            acc[pi][0] += pa
            acc[pi][1] += sum(r["points"]) - pa
            cpu += r["cpu"]
        for (a, b), (pa, pb) in zip(pairs, acc):
            self.state["edges"][a + "|" + b] = [round(pa, 2), round(pb, 2)]
        return cpu

    def _capped_cycle(self, members):
        """Top cycle; if larger than group_cap, keep its members with the most
        wins inside the cycle (Copeland; ties count 1/2, margin breaks ties)."""
        cyc = top_cycle(members, self._beats)
        cap = self.cfg.get("group_cap") or 0
        if not cap or len(cyc) <= cap:
            return cyc
        def score(m):
            wins, margin = 0.0, 0.0
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

    # ----------------------------------------------------- tree retraining
    def retrain_trees(self, runner, g):
        from .treelearn import collect, fit
        fams = self.state["families"]
        champ = self.state["hof"].get("gbdt_g%d" % (g - 1), {}).get("spec") or fams["gbdt"]["base"]
        data = os.path.join("results", "treedata", "evo_%s_g%d.bin" % (self.name, g))
        rng_seed = self.cfg["seed"] * 31 + g
        plan = self.pool_plan(self.cfg["retrain_tables"], rng_seed)
        spec = {"type": "collector", "name": "collector", "params": {"base": champ, "eps": 0.15}}
        r = evaluate_candidates(runner, [spec], plan, collect=True)[0]
        from array import array
        from .strategies.treebot import FEATURE_NAMES
        buf = array("f")
        n = 0
        for rows in r["experience"]:
            for row in rows or []:
                buf.extend(row)
                n += 1
        with open(os.path.join(ROOT, data), "wb") as f:
            buf.tofile(f)
        json.dump({"ncol": len(FEATURE_NAMES) + 2, "rows": n}, open(os.path.join(ROOT, data) + ".json", "w"))
        self.state["tree_data"].append(data)
        window = [os.path.join(ROOT, p) for p in self.state["tree_data"][-2:]]
        quiet = lambda s: None
        out = {}
        for fam, algo, trees, depth in (("gbdt", "gbdt", 40, 4), ("rf", "rf", 30, 7)):
            rel = os.path.join("pokerbot", "data", "evo_%s_%s_g%d.json" % (self.name, fam, g))
            fit(window, os.path.join(ROOT, rel), algo, trees, depth, 1500, 0.12, log=quiet)
            base = dict(fams[fam]["base"])
            base["params"] = dict(base["params"], model=rel)
            fams[fam]["base"] = base
            out[fam] = rel
        self.log("  retrained trees on %d new samples -> %s" % (n, out))
        return r["cpu"]

    # ------------------------------------------------------------ generation
    def generation(self, runner):
        st = self.state
        g = st["gen"]
        cfg = self.cfg
        t0 = time.time()
        cpu = 0.0
        self._spaces = {fam: self._space(fam) for fam in st["families"]}
        if cfg["retrain_every"] and g % cfg["retrain_every"] == 0:
            cpu += self.retrain_trees(runner, g)
            self._spaces = {fam: self._space(fam) for fam in st["families"]}
        # 1. candidates (CEM per family)
        cands, owner = [], []
        for fam, f in st["families"].items():
            sp = self._spaces[fam]
            vecs = [list(f["mu"])]
            for _ in range(cfg["pop"] - 1):
                vecs.append([min(1.0, max(0.0, m + s * self.rng.gauss(0, 1))) for m, s in zip(f["mu"], f["sigma"])])
            for i, u in enumerate(vecs):
                cands.append(sp.spec(u, "%s_g%d_c%d" % (fam, g, i)))
                owner.append((fam, u))
        # 2. evaluation on the shared pool plan
        plan = self.pool_plan(cfg["tables"], cfg["seed"] * 7777 + g)
        res = evaluate_candidates(runner, cands, plan)
        self.log("  [gen %d] evaluation %.0fs" % (g, time.time() - t0))
        scores = {}
        best = {}
        for (fam, u), c, r in zip(owner, cands, res):
            m = sum(r["points"]) / float(len(r["points"]))
            cpu += r["cpu"]
            scores.setdefault(fam, []).append((m, u, c))
        # 3. CEM update + promotion of each family's best
        new_members = []
        fam_best = {}
        for fam, lst in scores.items():
            lst.sort(key=lambda x: -x[0])
            el = lst[:max(2, cfg["pop"] // 3)]
            f = st["families"][fam]
            dim = len(f["mu"])
            f["mu"] = [sum(e[1][i] for e in el) / len(el) for i in range(dim)]
            f["sigma"] = [min(0.3, max(0.03, 1.1 * math.sqrt(sum((e[1][i] - f["mu"][i]) ** 2 for e in el) / len(el))))
                          for i in range(dim)]
            m, u, c = lst[0]
            name = "%s_g%d" % (fam, g)
            spec = _clean(c, name)
            st["hof"][name] = {"spec": spec, "family": fam, "gen": g, "fitness": round(m, 3)}
            new_members.append(name)
            fam_best[fam] = round(m, 2)
        # 4. equilibrium update: new vs group, new vs new
        group = list(st["group"])
        pairs = [(a, b) for a in new_members for b in group] + \
                [(a, b) for i, a in enumerate(new_members) for b in new_members[i + 1:]]
        cpu += self._duels(runner, pairs, salt=g)
        self.log("  [gen %d] duels %.0fs" % (g, time.time() - t0))
        st["group"] = self._capped_cycle(group + new_members)
        # 5. progress vs previous generation's bests, and vs frozen anchors
        new_specs = [st["hof"][m]["spec"] for m in new_members]
        prev = [st["hof"][m]["spec"] for m in st["prev_bests"]]
        anchors = [st["hof"][m]["spec"] for m in st["anchors"]] + [resolve(a) for a in ANCHOR_BASELINES]
        pr = evaluate_candidates(runner, new_specs, self.fixed_plan(prev, cfg["progress_tables"], cfg["seed"] * 13 + g))
        an = evaluate_candidates(runner, new_specs, self.fixed_plan(anchors, cfg["progress_tables"], cfg["seed"] * 17))
        prog = {}
        anch = {}
        for m, a, b in zip(new_members, pr, an):
            fam = st["hof"][m]["family"]
            prog[fam] = round(sum(a["points"]) / len(a["points"]), 2)
            anch[fam] = round(sum(b["points"]) / len(b["points"]), 2)
            cpu += a["cpu"] + b["cpu"]
        st["prev_bests"] = new_members
        wall = time.time() - t0
        st["cpu_s"] += cpu
        st["wall_s"] += wall
        row = {"gen": g, "wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "fitness": fam_best,
               "vs_prev_bests": prog, "vs_anchors": anch, "group": list(st["group"]), "hof": len(st["hof"]),
               "duel_pairs": len(pairs)}
        st["history"].append(row)
        st["gen"] = g + 1
        self.save()
        with open(self._path("log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        mean = lambda d: sum(d.values()) / max(1, len(d))
        self.log("gen %3d  %5.1fs  fit %.2f  vs-prev %.2f  vs-anchor %.2f  group %d %s  hof %d" % (
            g, wall, mean(fam_best), mean(prog), mean(anch), len(st["group"]),
            sorted(set(m.rsplit("_g", 1)[0] for m in st["group"])), len(st["hof"])))
        return wall

    def run(self, generations=60, minutes=None):
        t_start = time.time()
        with Runner(self.event, self.workers) as runner:
            self.load_or_init(runner)
            while self.state["gen"] <= generations:
                if minutes and time.time() - t_start > minutes * 60:
                    self.log("time budget reached")
                    break
                self.generation(runner)
        self.write_summary()

    def write_summary(self):
        st = self.state
        L = ["# Evolution run `%s`: %d generations, hall of fame %d, top-cycle group %d\n" % (
            self.name, st["gen"] - 1, len(st["hof"]), len(st["group"])),
             "CPU %.0f s, wall %.0f s. Settings: %s\n" % (st["cpu_s"], st["wall_s"], json.dumps(st["cfg_locked"])),
             "Columns are the mean over families. vs-prev = new bests vs the previous generation's bests "
             "(12.5 = parity); vs-anchor = vs frozen generation-0 champions + baselines (fixed seeds).\n",
             "| gen | wall s | fitness | vs prev bests | vs anchors | group |", "|---|---|---|---|---|---|"]
        for r in st["history"]:
            m = lambda d: sum(d.values()) / max(1, len(d))
            L.append("| %d | %.0f | %.2f | %.2f | %.2f | %s |" % (r["gen"], r["wall_s"], m(r["fitness"]),
                                                                m(r["vs_prev_bests"]), m(r["vs_anchors"]),
                                                                ", ".join(r["group"])))
        open(self._path("summary.md"), "w").write("\n".join(L) + "\n")


def verdict(run_dir, iterations=300, tournaments=6, xtables=1, workers=None, log=print):
    """Final verdict: the whole hall of fame plays itself (official format + cross-play)."""
    from .evaluate import official, crossplay
    st = json.load(open(os.path.join(run_dir, "state.json")))
    specs = [m["spec"] for m in st["hof"].values()]
    log("verdict over %d hall-of-fame players" % len(specs))
    rows = official(specs, iterations, tournaments, seed=99173, workers=workers,
                    out=os.path.join(run_dir, "verdict_official.json"), log=lambda s: None)
    xr, M = crossplay(specs, xtables, seed=77141, workers=workers,
                      out=os.path.join(run_dir, "verdict_crossplay.json"), log=lambda s: None)
    fam = {m["spec"]["name"]: (m["family"], m["gen"]) for m in st["hof"].values()}
    xmap = {r["name"]: r for r in xr}
    L = ["# Final verdict: %d hall-of-fame players (all generations) against each other\n" % len(specs),
         "Official = all-vs-all official format (%d iterations x %d tournaments, weighted 1..6). "
         "Cross-play = 1 vs 7 copies of every other player, %d table(s) per cell; meta = replicator share.\n" % (
             iterations, tournaments, xtables),
         "| # | player | family | gen | official weighted | mean/table | cross-play mean | worst | meta |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows[:60]):
        f, g = fam.get(r["name"], ("?", "?"))
        x = xmap.get(r["name"], {})
        L.append("| %d | %s | %s | %s | %.1f | %.2f | %.2f | %.2f | %.3f |" % (
            k + 1, r["name"], f, g, r["final"], r["mean"], x.get("mean_vs_others", 0), x.get("worst", 0),
            x.get("meta_share", 0)))
    # aggregates by family and by generation band
    L.append("\n## By family (mean over its hall-of-fame members)\n")
    L.append("| family | members | official mean/table | cross-play mean | best member |")
    L.append("|---|---|---|---|---|")
    by = {}
    for r in rows:
        f, g = fam.get(r["name"], ("?", 0))
        by.setdefault(f, []).append(r)
    for f, lst in sorted(by.items(), key=lambda kv: -sum(x["mean"] for x in kv[1]) / len(kv[1])):
        L.append("| %s | %d | %.2f | %.2f | %s |" % (f, len(lst), sum(x["mean"] for x in lst) / len(lst),
                                                    sum(xmap[x["name"]]["mean_vs_others"] for x in lst) / len(lst),
                                                    max(lst, key=lambda x: x["mean"])["name"]))
    L.append("\n## By generation band (does later = stronger?)\n")
    L.append("| generations | members | official mean/table | cross-play mean |")
    L.append("|---|---|---|---|")
    maxg = max(g for _, g in fam.values())
    band = max(1, int(math.ceil((maxg + 1) / 6.0)))
    for lo in range(0, maxg + 1, band):
        lst = [r for r in rows if lo <= fam[r["name"]][1] < lo + band]
        if lst:
            L.append("| %d-%d | %d | %.2f | %.2f |" % (lo, lo + band - 1, len(lst), sum(x["mean"] for x in lst) / len(lst),
                                                      sum(xmap[x["name"]]["mean_vs_others"] for x in lst) / len(lst)))
    md = "\n".join(L) + "\n"
    open(os.path.join(run_dir, "verdict.md"), "w").write(md)
    log(md)
    return md
