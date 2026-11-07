"""CEM tuning of the jam-top family against replicas of the top real teams
(run with the kit venv; independent of the main evolution).

Each generation: `pop` candidates (one is the current mean) play the same
`tables` tables (common random numbers) against 7 opponents drawn from the
top-N team replicas (results/kitevo_main/replicas.json); elites = top third;
mean / spread refit in normalised space. Every generation also re-checks the
best-so-far on fresh tables, paired against jam_g24 and the clone.
Writes results/tunejam/state.json and bots/jj_jamtop_tuned/agent.py (best).
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
from kitsim import Pool, evaluate, matched_plan, mean_ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT = os.path.join(ROOT, "bots", "jj_jamtop", "agent.py")
SPACE = {"open_ep": (0.05, 0.5), "open_mp": (0.05, 0.6), "open_co": (0.1, 0.8), "open_btn": (0.2, 1.0),
         "open_sb": (0.2, 1.0), "ref_bb": (4.0, 40.0), "depth_exp": (0.0, 2.0), "deep_pct": (0.0, 0.12),
         "call_pct": (0.03, 0.5), "call_exp": (0.0, 2.0), "call_floor": (0.0, 0.08), "post_cat": (0.6, 3.0)}
SEED = {"open_ep": 0.24, "open_mp": 0.32, "open_co": 0.45, "open_btn": 0.72, "open_sb": 0.78, "ref_bb": 10.0,
        "depth_exp": 1.0, "deep_pct": 0.03, "call_pct": 0.25, "call_exp": 1.0, "call_floor": 0.02, "post_cat": 1.4}


JAM_BOT = os.path.join(ROOT, "bots", "jj_jam", "agent.py")
JAM_SPACE = {"push_th": (0.4, 0.95), "call_th": (0.45, 0.98), "ref_bb": (3.0, 40.0), "stack_exp": (0.0, 2.0),
             "post_cat": (1.0, 4.0)}


def to_p(u):
    return {k: round(lo + min(1, max(0, x)) * (hi - lo), 4) for (k, (lo, hi)), x in zip(SPACE.items(), u)}


def to_u(p):
    return [(p[k] - lo) / (hi - lo) for k, (lo, hi) in SPACE.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--tables", type=int, default=120)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--family", default="jamtop", choices=["jamtop", "jam"])
    ap.add_argument("--seed-member", default="jam_g24", help="jam family: HoF member whose params seed the search")
    a = ap.parse_args()
    global BOT, SPACE, SEED
    tag = "tunejam" if a.family == "jamtop" else "tunejam_%s" % a.seed_member
    if a.family == "jam":
        BOT, SPACE = JAM_BOT, JAM_SPACE
        st = json.load(open(os.path.join(ROOT, "results", "kitevo_main", "state.json")))
        SEED = {k: st["hof"][a.seed_member]["spec"]["params"][k] for k in JAM_SPACE}
    out_dir = os.path.join(ROOT, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    reps = json.load(open(os.path.join(ROOT, "results", "kitevo_main", "replicas.json")))["replicas"]
    top = sorted([r for r in reps if r["weight"] >= 40], key=lambda r: -r["points_real"])[:a.top_n]
    field = [r["spec"] for r in top]
    rng = random.Random(7)
    mu, sig = to_u(SEED), [a.sigma] * len(SPACE)
    best = (-1e9, SEED)
    refs = [{"path": os.path.join(ROOT, "bots", "auto", "jam_g24", "agent.py"), "name": "jam_g24"},
            {"path": os.path.join(ROOT, "bots", "jj_clone", "agent.py"), "name": "clone"}]
    hist = []
    with Pool(a.workers) as p:
        for g in range(a.gens):
            t0 = time.time()
            us = [list(mu)] + [[min(1, max(0, m + s * rng.gauss(0, 1))) for m, s in zip(mu, sig)] for _ in range(a.pop - 1)]
            cands = [{"path": BOT, "params": to_p(u), "name": "jt_g%d_%d" % (g, i)} for i, u in enumerate(us)]
            res = evaluate(p, cands, matched_plan(a.tables, field, 1000 + g))
            sc = sorted(((sum(r["points"]) / len(r["points"]), u) for r, u in zip(res, us)), key=lambda x: -x[0])
            el = sc[:max(2, a.pop // 3)]
            mu = [sum(e[1][i] for e in el) / len(el) for i in range(len(mu))]
            sig = [max(0.03, min(0.3, 1.1 * math.sqrt(sum((e[1][i] - mu[i]) ** 2 for e in el) / len(el)))) for i in range(len(mu))]
            # re-check the generation's best and the new mean on fresh tables, paired vs the references
            chk = [{"path": BOT, "params": to_p(el[0][1]), "name": "best"}, {"path": BOT, "params": to_p(mu), "name": "mean"}] + refs
            cr = evaluate(p, chk, matched_plan(a.tables, field, 50000 + g))
            ref = cr[2]["points"]
            row = {"gen": g, "best_screen": round(el[0][0], 2), "secs": round(time.time() - t0)}
            for c, r in zip(chk, cr):
                m, ci = mean_ci(r["points"])
                dm, dci = mean_ci([x - y for x, y in zip(r["points"], ref)])
                row[c["name"]] = [round(m, 2), round(dm, 2), round(dci, 2)]
            for nm, u in (("best", el[0][1]), ("mean", mu)):
                if row[nm][0] > best[0]:
                    best = (row[nm][0], to_p(u))
            row["best_so_far"] = round(best[0], 2)
            hist.append(row)
            print(json.dumps(row), flush=True)
            json.dump({"hist": hist, "best": best[1], "best_check": best[0], "mu": to_p(mu)},
                      open(os.path.join(out_dir, "state.json"), "w"), indent=1)
            dst = os.path.join(ROOT, "bots", "jj_jamtop_tuned" if a.family == "jamtop" else "jj_%s_tuned" % a.seed_member)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.makedirs(dst)
            src = open(BOT).read() + "\n\n# tuned vs top-team replicas (tools/tunejam.py, %s)\nAgent.P = dict(Agent.P, **%s)\n" % (
                tag, json.dumps(best[1]))
            open(os.path.join(dst, "agent.py"), "w").write(src)


if __name__ == "__main__":
    main()
