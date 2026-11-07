"""Parameter search on expected tournament points.

  * random search: many candidates on a small matched screening budget,
    top-M finalists re-run on a larger budget with fresh seeds
  * CEM (evolution): Gaussian search distribution in normalised space,
    elites refit each generation; fresh matched seeds every generation;
    archived elites join the opponent pool (self-play / archive)
All candidates in one batch share identical tables (common random numbers).
"""
import json
import math
import os
import random
import time

from .sim import Runner, load_fields, matched_jobs, evaluate_candidates, mean_ci, summarize
from .strategies import resolve, MECHANISM
from .strategies.base import COMMON_PARAMS, resolve_params
from .strategies.learned import default_weights, N_FEAT, ACTS

INT_PARAMS = {"abstraction", "value_class", "call_class", "raise_class", "big_bet_call_class", "post",
              "post_class"}

SPACES = {
    "core": ["open_pct", "pos_widen", "steal_widen", "open_size", "threebet_pct", "resteal_rate",
             "call_margin", "value_eq", "raise_eq", "value_size", "bluff_size", "bluff_rate",
             "semibluff_rate", "cbet_rate", "shove_bb", "push_widen", "call_shove_margin",
             "commit_spr", "icm_risk", "bigstack_pressure", "multiway_tighten", "adapt",
             "implied", "endgame_push"],
    "preflop": ["open_pct", "pos_widen", "steal_widen", "open_size", "threebet_pct", "resteal_rate",
                "call_margin", "shove_bb", "push_widen", "call_shove_margin", "icm_risk",
                "bigstack_pressure"],
    "allin": ["push_pct", "call_pct", "ref_bb", "stack_exp", "post_class"],
    "postflop": ["value_eq", "raise_eq", "value_size", "bluff_size", "bluff_rate", "semibluff_rate",
                 "cbet_rate", "commit_spr", "multiway_tighten", "implied", "adapt"],
}
EXTRA_BOUNDS = {"push_pct": (1.0, 0.03, 1.0), "call_pct": (1.0, 0.02, 1.0), "ref_bb": (15.0, 3.0, 40.0),
                "stack_exp": (0.0, 0.0, 2.0), "post_class": (2, 0, 10), "prior_n": (12.0, 0.0, 60.0),
                "range_tighten": (0.6, 0.2, 1.0), "margin": (0.25, 0.0, 2.0), "mc_budget_us": (250.0, 100.0, 400.0),
                "value_class": (5, 3, 10), "call_class": (3, 1, 7), "raise_class": (8, 6, 12),
                "big_bet_call_class": (5, 3, 9)}


def bounds(name):
    if name in COMMON_PARAMS:
        return COMMON_PARAMS[name]
    return EXTRA_BOUNDS[name]


class Space(object):
    """Maps a vector u in [0,1]^d to a concrete agent spec."""

    def __init__(self, base_spec, space="core", linear_scale=3.0):
        self.base = resolve(base_spec)
        self.type = self.base["type"]
        self.linear = self.type == "linear" or space == "linear"
        if self.linear:
            self.names = ["w%d" % i for i in range(N_FEAT * len(ACTS))]
            self.w0 = list(self.base.get("params", {}).get("w") or default_weights())
            self.scale = linear_scale
        else:
            self.names = list(SPACES.get(space, space.split(",") if isinstance(space, str) else space))
            post = self.base.get("params", {}).get("post", 0)
            if space == "core" and (self.type in ("heuristic", "tree") or (self.type == "hybrid" and post)):
                self.names += ["value_class", "call_class", "raise_class", "big_bet_call_class"]
            if space == "core" and self.type == "tree":
                self.names += ["margin"]
            if space == "core" and self.type == "montecarlo":
                self.names += ["fold_prior", "fold_slope", "called_penalty", "realize", "prior_n",
                               "range_tighten"]
            if space == "core" and self.type in ("equity", "lookup", "linear") or \
                    (space == "core" and self.type == "hybrid" and not post):
                self.names += ["fold_prior", "fold_slope", "called_penalty", "realize"]

    @property
    def dim(self):
        return len(self.names)

    def center(self):
        if self.linear:
            return [0.5] * self.dim
        full = resolve_params(self.base.get("params"), self.base.get("styles"),
                              EXTRA_BOUNDS_DEFAULTS())
        u = []
        for n in self.names:
            d, lo, hi = bounds(n)
            u.append((full.get(n, d) - lo) / float(hi - lo))
        return [min(1.0, max(0.0, x)) for x in u]

    def spec(self, u, name):
        params = dict(self.base.get("params", {}))
        if self.linear:
            params["w"] = [round(w + (x - 0.5) * 2 * self.scale, 4) for w, x in zip(self.w0, u)]
        else:
            for n, x in zip(self.names, u):
                d, lo, hi = bounds(n)
                v = lo + min(1.0, max(0.0, x)) * (hi - lo)
                params[n] = int(round(v)) if n in INT_PARAMS else round(v, 4)
        s = dict(self.base)
        s["params"] = params
        s["name"] = name
        return s


def EXTRA_BOUNDS_DEFAULTS():
    d = {k: v[0] for k, v in EXTRA_BOUNDS.items()}
    # heuristic class defaults (match HeuristicBot.DEFAULTS)
    d.update({"value_class": 5, "call_class": 3, "raise_class": 8, "big_bet_call_class": 5})
    return d


def objective(res, kind="mean"):
    m = sum(res["points"]) / float(max(1, len(res["points"])))
    if kind == "mean":
        return m
    worst = min(sum(v) / float(len(v)) for v in res["by_field"].values())
    if kind == "worst":
        return worst
    return 0.5 * m + 0.5 * worst


def run(base, method="cem", space="core", fields_path=None, out_dir=None, generations=8,
        popsize=16, elite=4, screen_tables=240, final_tables=1200, finalists=4, seed=1,
        sigma=0.18, archive_frac=0.15, objective_kind="mean", workers=None, event=None,
        n_random=48, log=print):
    from .event import Event
    event = event or Event.load()
    fields = load_fields(fields_path or os.path.join("configs", "populations", "train.json"))
    sp = Space(base, space)
    rng = random.Random(seed)
    t0 = time.time()
    cpu = 0.0
    out_dir = out_dir or os.path.join("results", "opt_%s_%s_%d" % (sp.base["name"], method, seed))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    logf = open(os.path.join(out_dir, "log.jsonl"), "w")
    archive = []
    pool_specs = []  # (score, spec) screened
    base_spec = sp.spec(sp.center(), sp.base["name"] + "_base")
    with Runner(event, workers) as runner:
        if method == "random":
            cands = [base_spec] + [sp.spec([rng.random() for _ in range(sp.dim)], "%s_r%d" % (sp.base["name"], i))
                                   for i in range(n_random)]
            for b in range(0, len(cands), popsize):
                batch = cands[b:b + popsize]
                plan = matched_jobs(screen_tables, fields, seed * 7919 + b)
                res = evaluate_candidates(runner, batch, plan)
                for r in res:
                    sc = objective(r, objective_kind)
                    cpu += r["cpu"]
                    pool_specs.append((sc, r["spec"]))
                    logf.write(json.dumps({"stage": "screen", "batch": b, "score": sc,
                                           "spec": r["spec"]}) + "\n")
                log("screened %d/%d (best so far %.2f) %.0fs" % (min(b + popsize, len(cands)), len(cands),
                                                                max(s for s, _ in pool_specs), time.time() - t0))
        else:
            mu = sp.center()
            sig = [sigma] * sp.dim
            for g in range(generations):
                cands = [sp.spec(mu, "%s_g%d_mu" % (sp.base["name"], g))]
                for i in range(popsize - 1):
                    u = [min(1.0, max(0.0, m + s * rng.gauss(0, 1))) for m, s in zip(mu, sig)]
                    cands.append(sp.spec(u, "%s_g%d_%d" % (sp.base["name"], g, i)))
                plan = matched_jobs(screen_tables, fields, seed * 104729 + g, archive=archive,
                                    archive_frac=archive_frac if archive else 0.0)
                res = evaluate_candidates(runner, cands, plan)
                scored = []
                for r in res:
                    sc = objective(r, objective_kind)
                    cpu += r["cpu"]
                    scored.append((sc, r))
                    logf.write(json.dumps({"stage": "gen", "gen": g, "score": sc, "spec": r["spec"]}) + "\n")
                scored.sort(key=lambda x: -x[0])
                el = scored[:elite]
                vecs = [_encode(sp, r["spec"]) for _, r in el]
                mu = [sum(v[i] for v in vecs) / len(vecs) for i in range(sp.dim)]
                sig = [max(0.03, math.sqrt(sum((v[i] - mu[i]) ** 2 for v in vecs) / len(vecs)) * 1.1)
                       for i in range(sp.dim)]
                best = el[0][1]["spec"]
                pool_specs.append((el[0][0], best))
                pool_specs.append((None, sp.spec(mu, "%s_g%d_mean" % (sp.base["name"], g))))
                arch = dict(best)
                arch["name"] = "archive_" + best["name"]
                archive.append(arch)
                log("gen %d: best %.2f  mu-cand %.2f  median %.2f  %.0fs" % (
                    g, el[0][0], scored[[r["spec"]["name"] for _, r in scored].index(cands[0]["name"])][0],
                    scored[len(scored) // 2][0], time.time() - t0))
        # ---- finalists on a larger budget, fresh seeds, CRN across finalists
        ranked = [s for s in pool_specs if s[0] is not None]
        ranked.sort(key=lambda x: -x[0])
        fin = [base_spec] + [s for _, s in ranked[:finalists]]
        if method == "cem":
            fin.append(pool_specs[-1][1])  # final search mean
        seen, uniq = set(), []
        for s in fin:
            if s["name"] not in seen:
                seen.add(s["name"])
                uniq.append(s)
        plan = matched_jobs(final_tables, fields, seed * 15485863 + 99)
        res = evaluate_candidates(runner, uniq, plan)
        final = []
        for r in res:
            cpu += r["cpu"]
            m, ci = mean_ci(r["points"])
            final.append({"name": r["spec"]["name"], "score": objective(r, objective_kind), "mean": m,
                          "ci95": ci, "by_field": {k: round(sum(v) / len(v), 2) for k, v in r["by_field"].items()},
                          "spec": r["spec"]})
            logf.write(json.dumps({"stage": "final", "name": r["spec"]["name"], "mean": m, "ci95": ci}) + "\n")
        final.sort(key=lambda x: -x["score"])
    wall = time.time() - t0
    best = dict(final[0]["spec"])
    best["train_cpu_s"] = round(cpu, 1)
    best["train_wall_s"] = round(wall, 1)
    summary = {"method": method, "space": space, "base": sp.base["name"], "objective": objective_kind,
               "train_cpu_s": round(cpu, 1), "wall_s": round(wall, 1), "finalists": final,
               "settings": {"generations": generations, "popsize": popsize, "screen_tables": screen_tables,
                            "final_tables": final_tables, "seed": seed, "n_random": n_random}}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    json.dump(best, open(os.path.join(out_dir, "best.json"), "w"), indent=1)
    logf.close()
    for f in final:
        log("final %-28s %.2f +- %.2f  %s" % (f["name"], f["mean"], f["ci95"], f["by_field"]))
    log("best -> %s (cpu %.0fs, wall %.0fs)" % (os.path.join(out_dir, "best.json"), cpu, wall))
    return best, summary


def _encode(sp, spec):
    p = spec["params"]
    if sp.linear:
        return [min(1.0, max(0.0, (w - w0) / (2 * sp.scale) + 0.5)) for w, w0 in zip(p["w"], sp.w0)]
    u = []
    for n in sp.names:
        d, lo, hi = bounds(n)
        u.append((p[n] - lo) / float(hi - lo))
    return u


def train_mixture(components, fields_path=None, generations=6, popsize=10, tables=240, seed=3,
                  workers=None, out_dir=None, log=print):
    """CEM over mixture weights (softmax of logits)."""
    from .event import Event
    event = Event.load()
    fields = load_fields(fields_path or os.path.join("configs", "populations", "train.json"))
    rng = random.Random(seed)
    k = len(components)
    mu, sig = [0.0] * k, [1.0] * k
    t0, cpu = time.time(), 0.0
    best = None
    with Runner(event, workers) as runner:
        for g in range(generations):
            logits = [mu] + [[m + s * rng.gauss(0, 1) for m, s in zip(mu, sig)] for _ in range(popsize - 1)]
            specs = []
            for i, l in enumerate(logits):
                mx = max(l)
                w = [math.exp(x - mx) for x in l]
                tot = sum(w)
                specs.append({"type": "mixture", "name": "mix_g%d_%d" % (g, i),
                              "params": {"components": components, "weights": [round(x / tot, 3) for x in w]}})
            plan = matched_jobs(tables, fields, seed * 7 + g)
            res = evaluate_candidates(runner, specs, plan)
            sc = sorted([(objective(r), l, r["spec"]) for r, l in zip(res, logits)], key=lambda x: -x[0])
            cpu += sum(r["cpu"] for r in res)
            el = sc[:max(2, popsize // 4)]
            mu = [sum(e[1][i] for e in el) / len(el) for i in range(k)]
            sig = [max(0.2, math.sqrt(sum((e[1][i] - mu[i]) ** 2 for e in el) / len(el))) for i in range(k)]
            best = el[0][2]
            log("mix gen %d best %.2f weights %s" % (g, el[0][0], best["params"]["weights"]))
    best["train_cpu_s"] = round(cpu, 1)
    if out_dir:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        json.dump(best, open(os.path.join(out_dir, "best.json"), "w"), indent=1)
    return best
