"""Field replicas: one bot per real team, matched on observed style
(run with the kit venv; does not touch a running evolution).

1. Read a round summary from tools/hands.py (--json): per team VPIP, PFR,
   preflop all-in rate, postflop aggression, fold-to-bet, tables seen.
2. Draw many random parameter vectors from our families (rule, jam, mc) plus the
   check/fold baseline, play them against each other (8 different draws per
   table) and measure the same statistics with the same definitions.
3. Match every team to the nearest draw (standardised distance on the five
   statistics); weight = how often the team appeared at our tables.
Output: results/kitevo_<name>/replicas.json, used by the auto-submit verdict
(replica-field evaluation) and by kitevolve's generator after its next restart.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitsim import Pool, run_table  # noqa: E402
from kitevolve import FAMILIES, ROOT, spec, space_names, to_params  # noqa: E402

KEYS = ["vpip", "pfr", "pf_allin", "post_aggr", "fold_to_bet"]


def stats_of(b):
    h = max(1, b["hands"])
    return {"vpip": b["vpip"] / h, "pfr": b["pfr"] / h, "pf_allin": b["pf_allin"] / h,
            "post_aggr": b["post_aggr"] / max(1, b["post_dec"]), "fold_to_bet": b["post_fold"] / max(1, b["post_faced"]),
            "n_hands": b["hands"], "n_post": b["post_dec"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--name", default="main")
    ap.add_argument("--draws", type=int, default=320)
    ap.add_argument("--rounds", type=int, default=6, help="tables per draw")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    # several rounds: merge per team, statistics and points weighted by tables seen
    merged = {}
    for path in a.summary.split(","):
        for t in json.load(open(path))["teams"]:
            if t["is_me"]:
                continue
            m = merged.setdefault(t["team"], {"team": t["team"], "is_me": False, "tables": 0, "pts": 0.0,
                                               **{k: 0.0 for k in KEYS}})
            w = t["tables"]
            m["tables"] += w
            m["pts"] += t["pts"] * w
            for k in KEYS:
                m[k] += t[k] * w
    teams = []
    for m in merged.values():
        if m["tables"] >= 20:
            w = float(m["tables"])
            m["pts"] /= w
            for k in KEYS:
                m[k] /= w
            teams.append(m)
    rng = random.Random(a.seed)
    mix = [("rule", 0.55), ("jam", 0.15), ("mc", 0.25), ("checkfold", 0.05)]
    draws = []
    for i in range(a.draws):
        x = rng.random()
        acc = 0.0
        for fam, w in mix:
            acc += w
            if x <= acc:
                break
        if fam == "checkfold":
            draws.append({"builtin": "checkfold", "name": "rep_checkfold_%d" % i})
        else:
            draws.append(spec(fam, to_params(fam, [rng.random() for _ in space_names(fam)]), "rep_%s_%d" % (fam, i)))
    # every table seats 8 different draws; each draw plays ~`rounds` tables
    jobs, seats = [], []
    order = list(range(len(draws)))
    for r in range(a.rounds):
        rng.shuffle(order)
        for t in range(0, len(order) - 7, 8):
            idx = order[t:t + 8]
            jobs.append(([draws[i] for i in idx], rng.randrange(1 << 30), {}))
            seats.append(idx)
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(run_table, jobs, chunksize=4)
    agg = [dict(hands=0, vpip=0, pfr=0, pf_allin=0, pf_dec=0, pf_fold=0, post_dec=0, post_aggr=0,
                post_faced=0, post_fold=0) for _ in draws]
    for idx, r in zip(seats, res):
        for seat, di in enumerate(idx):
            for k, v in r["behavior"][seat].items():
                agg[di][k] += v
    dstats = [stats_of(b) for b in agg]
    # standardise by the spread of the real teams
    sd = {}
    for k in KEYS:
        xs = [t[k] for t in teams]
        m = sum(xs) / len(xs)
        sd[k] = max(0.03, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))
    out = []
    for t in teams:
        best, bd = None, 1e9
        for i, s in enumerate(dstats):
            if s["n_hands"] < 50:
                continue
            d = math.sqrt(sum(((s[k] - t[k]) / sd[k]) ** 2 for k in KEYS))
            if d < bd:
                best, bd = i, d
        rep = dict(draws[best])
        rep["name"] = "replica:" + t["team"]
        out.append({"team": t["team"], "weight": t["tables"], "points_real": t["pts"], "dist": round(bd, 3),
                    "team_stats": {k: t[k] for k in KEYS},
                    "replica_stats": {k: round(dstats[best][k], 3) for k in KEYS}, "spec": rep})
    path = os.path.join(ROOT, "results", "kitevo_%s" % a.name, "replicas.json")
    json.dump({"source": a.summary, "teams": len(out), "built_s": round(time.time() - t0, 1), "replicas": out},
              open(path, "w"), indent=1)
    good = sum(1 for o in out if o["dist"] < 1.0)
    print("matched %d teams (%d within 1 sd) from %d draws over %d tables in %.0fs -> %s" % (
        len(out), good, len(draws), len(jobs), time.time() - t0, path))
    for o in sorted(out, key=lambda o: -o["weight"])[:12]:
        print("  %-24s d=%.2f team %s replica %s" % (o["team"][:24], o["dist"], o["team_stats"], o["replica_stats"]))


if __name__ == "__main__":
    main()
