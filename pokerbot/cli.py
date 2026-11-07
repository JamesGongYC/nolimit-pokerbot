"""Command line: python -m pokerbot.cli <command> [options]

  precompute     build equity lookup tables (offline MC)
  pushfold       solve push/fold equilibrium tables (fictitious play)
  simulate       play tables with a fixed lineup
  tournament     official-format sim (random tables, fillers, 6 weighted tournaments)
  optimize       random / CEM parameter search on tournament points
  train-table    learn a lookup table from exploring rollouts
  train-mixture  CEM over randomized-mixture weights
  evaluate       hold-out leaderboard (CI, placements, latency, errors, memory, cost)
  export         write the upload: submission/ + submission.zip (agent.py at root)
  bench          benchmark the exported bot through its interface
  test           run the unit tests
"""
import argparse
import json
import os
import random
import sys

from .event import Event


def _list(s):
    return [x for x in s.split(",") if x] if s else []


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pokerbot")
    ap.add_argument("--event", default=None, help="event config JSON (default: provisional)")
    ap.add_argument("--workers", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("precompute")
    s.add_argument("--samples", type=float, default=5.0)

    s = sub.add_parser("pushfold")
    s.add_argument("--iters", type=int, default=300)

    s = sub.add_parser("simulate")
    s.add_argument("--lineup", required=True, help="comma list of 8 presets (or fewer: cycled)")
    s.add_argument("--tables", type=int, default=200)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--rotate", action="store_true", help="rotate seats every table")

    s = sub.add_parser("tournament")
    s.add_argument("--entries", required=True, help="comma list; '@file.json' expands an entries file")
    s.add_argument("--iterations", type=int, default=1000)
    s.add_argument("--tournaments", type=int, default=6)
    s.add_argument("--seed", type=int, default=424242)
    s.add_argument("--out", default=None)

    s = sub.add_parser("optimize")
    s.add_argument("--base", required=True, help="preset name or spec JSON")
    s.add_argument("--method", default="cem", choices=["cem", "random"])
    s.add_argument("--space", default="core")
    s.add_argument("--generations", type=int, default=8)
    s.add_argument("--popsize", type=int, default=16)
    s.add_argument("--elite", type=int, default=4)
    s.add_argument("--n-random", type=int, default=48)
    s.add_argument("--screen-tables", type=int, default=240)
    s.add_argument("--final-tables", type=int, default=1200)
    s.add_argument("--finalists", type=int, default=4)
    s.add_argument("--sigma", type=float, default=0.18, help="initial CEM std in normalised space")
    s.add_argument("--archive-frac", type=float, default=0.15, help="share of opponents drawn from archived elites")
    s.add_argument("--objective", default="mean", choices=["mean", "worst", "mean_worst"])
    s.add_argument("--fields", default=None)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", default=None)

    s = sub.add_parser("train-table")
    s.add_argument("--name", default="qtable")
    s.add_argument("--generations", type=int, default=6)
    s.add_argument("--tables", type=int, default=600)
    s.add_argument("--res", type=int, default=2)
    s.add_argument("--postflop", action="store_true", help="also learn postflop states")
    s.add_argument("--base", default=None, help="spec JSON whose params seed the learner/fallback")
    s.add_argument("--seed", type=int, default=5)

    s = sub.add_parser("train-trees")
    s.add_argument("--base", required=True, help="base policy spec (preset or JSON)")
    s.add_argument("--name", default="tree")
    s.add_argument("--algo", default="gbdt", choices=["gbdt", "rf"])
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--tables", type=int, default=3000)
    s.add_argument("--eps", type=float, default=0.12)
    s.add_argument("--trees", type=int, default=120)
    s.add_argument("--depth", type=int, default=5)
    s.add_argument("--leaf", type=int, default=400)
    s.add_argument("--lr", type=float, default=0.08)
    s.add_argument("--margin", type=float, default=0.0)
    s.add_argument("--eval-tables", type=int, default=1500)
    s.add_argument("--seed", type=int, default=31)

    s = sub.add_parser("train-mixture")
    s.add_argument("--components", required=True)
    s.add_argument("--generations", type=int, default=6)
    s.add_argument("--tables", type=int, default=240)
    s.add_argument("--out", default="results/mixture")

    s = sub.add_parser("evaluate")
    s.add_argument("--candidates", required=True)
    s.add_argument("--tables", type=int, default=2000)
    s.add_argument("--fields", default=None)
    s.add_argument("--seed", type=int, default=777000111)
    s.add_argument("--out", default="results/leaderboard.json")
    s.add_argument("--no-memory", action="store_true")

    s = sub.add_parser("crossplay")
    s.add_argument("--strategies", required=True)
    s.add_argument("--tables", type=int, default=60)
    s.add_argument("--seed", type=int, default=5150)
    s.add_argument("--out", default="results/crossplay.json")

    s = sub.add_parser("evolve", help="evolutionary population training (top-cycle hall of fame)")
    s.add_argument("--name", default="main")
    s.add_argument("--generations", type=int, default=60)
    s.add_argument("--minutes", type=float, default=None, help="stop after this wall time")
    s.add_argument("--pop", type=int, default=8, help="candidates per family per generation")
    s.add_argument("--tables", type=int, default=150, help="tables per candidate per generation")
    s.add_argument("--duel-tables", type=int, default=24)
    s.add_argument("--progress-tables", type=int, default=80)
    s.add_argument("--retrain-every", type=int, default=10)
    s.add_argument("--retrain-tables", type=int, default=6000)
    s.add_argument("--group-cap", type=int, default=16, help="max top-cycle members kept (Copeland); 0 = no cap")
    s.add_argument("--seed", type=int, default=2026)

    s = sub.add_parser("evo-verdict", help="final verdict: whole hall of fame vs itself")
    s.add_argument("--run", required=True, help="results/evo_<name>")
    s.add_argument("--iterations", type=int, default=300)
    s.add_argument("--tournaments", type=int, default=6)
    s.add_argument("--xtables", type=int, default=1)

    s = sub.add_parser("export", help="write submission/ + submission.zip (agent.py at the zip root)")
    s.add_argument("--spec", required=True)
    s.add_argument("--out", default="submission")
    s.add_argument("--zip", default=None, help="upload zip path (default: <out>.zip)")

    s = sub.add_parser("bench")
    s.add_argument("--dir", default="submission")
    s.add_argument("--tables", type=int, default=40)

    sub.add_parser("test")

    a = ap.parse_args(argv)
    ev = Event.load(a.event) if a.cmd != "test" else None

    if a.cmd == "precompute":
        from .precompute import run
        run(ev, samples=a.samples, workers=a.workers)
    elif a.cmd == "pushfold":
        from .pfsolve import build
        build(ev.variant.name, iters=a.iters, workers=a.workers)
    elif a.cmd == "simulate":
        from .sim import Runner, mean_ci, summarize
        from .engine import SeatStats
        from .evaluate import load_spec
        from .strategies import resolve
        lineup = [resolve(load_spec(x)) for x in _list(a.lineup)]
        while len(lineup) < ev.table_size:
            lineup += lineup[:ev.table_size - len(lineup)]
        lineup = lineup[:ev.table_size]
        jobs = []
        for t in range(a.tables):
            k = t % ev.table_size if a.rotate else 0
            jobs.append((lineup[k:] + lineup[:k], a.seed * 100003 + t, None))
        with Runner(ev, a.workers) as r:
            res = r.run(jobs)
        n = ev.table_size
        agg = {}
        for t, rr in enumerate(res):
            k = t % n if a.rotate else 0
            for seat in range(n):
                idx = (seat + k) % n
                e = agg.setdefault(idx, {"pts": [], "pl": [], "st": SeatStats()})
                e["pts"].append(rr["points"][seat])
                e["pl"].append(rr["places"][seat])
                e["st"].merge(SeatStats.from_dict(rr["stats"][seat]))
        print("seat-slot  strategy                points   ±95%   1st%   p99us  errors")
        for idx in range(n):
            e = agg[idx]
            d = summarize(lineup[idx]["name"], e["pts"], e["pl"], e["st"])
            print("%-9d  %-22s %7.2f %6.2f %6.1f %7d %5d" % (idx, d["name"], d["points"], d["ci95"],
                                                         100 * d["place_dist"][0], d["lat_p99_us"],
                                                         d["errors"] + d["illegal"] + d["timeouts"]))
        print("avg hands/table %.1f, capped %.0f%%, chip failures %d" % (
            sum(x["hands"] for x in res) / len(res), 100 * sum(x["capped"] for x in res) / len(res),
            sum(x["chip_fail"] for x in res)))
    elif a.cmd == "tournament":
        from .evaluate import official
        entries = []
        for e in _list(a.entries):
            if e.startswith("@"):
                entries += json.load(open(e[1:]))["entries"]
            else:
                entries.append(e)
        official(entries, a.iterations, a.tournaments, a.seed, a.workers, ev, out=a.out)
    elif a.cmd == "optimize":
        from .optimize import run
        from .evaluate import load_spec
        run(load_spec(a.base), a.method, a.space, a.fields, a.out, a.generations, a.popsize, a.elite,
            a.screen_tables, a.final_tables, a.finalists, a.seed, sigma=a.sigma,
            archive_frac=a.archive_frac, objective_kind=a.objective,
            workers=a.workers, event=ev, n_random=a.n_random)
    elif a.cmd == "train-table":
        from .search import train_qtable
        base = json.load(open(a.base))["params"] if a.base else None
        train_qtable(a.name, base, a.generations, a.tables, res=a.res, seed=a.seed, workers=a.workers,
                     event=ev, preflop_only=0 if a.postflop else 1)
    elif a.cmd == "train-trees":
        from .treelearn import iterate
        iterate(a.base, a.name, a.rounds, a.tables, a.eps, a.algo, a.trees, a.depth, a.leaf, a.lr,
                eval_tables=a.eval_tables, margin=a.margin, seed=a.seed, workers=a.workers, event=ev)
    elif a.cmd == "train-mixture":
        from .optimize import train_mixture
        from .evaluate import load_spec
        train_mixture([load_spec(c) for c in _list(a.components)], generations=a.generations,
                      tables=a.tables, workers=a.workers, out_dir=a.out)
    elif a.cmd == "evaluate":
        from .evaluate import leaderboard
        leaderboard(_list(a.candidates), a.fields, a.tables, a.seed, a.workers, ev, a.out,
                    memory=not a.no_memory)
    elif a.cmd == "crossplay":
        from .evaluate import crossplay
        crossplay(_list(a.strategies), a.tables, a.seed, a.workers, ev, a.out)
    elif a.cmd == "evolve":
        from .evolve import Evolution
        Evolution(a.name, a.pop, a.tables, a.duel_tables, a.progress_tables, a.retrain_every, a.retrain_tables,
                  a.seed, a.workers, ev, group_cap=a.group_cap).run(a.generations, a.minutes)
    elif a.cmd == "evo-verdict":
        from .evolve import verdict
        verdict(a.run, a.iterations, a.tournaments, a.xtables, a.workers)
    elif a.cmd == "export":
        from .export import export
        from .evaluate import load_spec
        export(load_spec(a.spec), a.out, zip_path=a.zip,
               limit_mb=float(ev.cfg.get("submission_size_limit_mb", 50)))
    elif a.cmd == "bench":
        from .export import bench
        bench(a.dir, a.tables, ev)
    elif a.cmd == "test":
        import unittest
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        suite = unittest.defaultTestLoader.discover(os.path.join(root, "tests"), top_level_dir=root)
        r = unittest.TextTestRunner(verbosity=1).run(suite)
        sys.exit(0 if r.wasSuccessful() else 1)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
