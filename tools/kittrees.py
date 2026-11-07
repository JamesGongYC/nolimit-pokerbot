"""GBDT family pipeline on the kit engine (run with the kit venv):
collect -> fit -> gate (fitted policy iteration, same scheme as the hold'em sandbox).

  collect: CollectorBot = bots/jj_tree with eps > 0 and the current MC champion's
           parameters, 4 collector seats + 4 opponents from the evolution's pool
           (top-cycle group + generator) per table; rows = 18 features + action +
           the hand's ICM change in points
  fit:     tools/train_trees.py (scikit-learn HistGradientBoosting, one regressor
           per abstract action) -> JSON trees, compiled to Python by the bot
  gate:    tree bot (champion params + new model) vs the same params without a
           model on matched pool tables; adopt (copy to bots/jj_tree/model.json)
           only if it scores higher
usage:
  starter-kit/.venv/bin/python tools/kittrees.py --name main --tables 2000 --gate-tables 200 --workers 3
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitsim import Pool, evaluate, mean_ci, run_table  # noqa: E402
from kitevolve import ROOT, KitEvolution  # noqa: E402

TREE_DIR = os.path.join(ROOT, "bots", "jj_tree")
TREE = os.path.join(TREE_DIR, "agent.py")
DATA = os.path.join(ROOT, "results", "kittreedata")
NCOL = 18 + 2


def champion_params(st, fam="mc"):
    best = max(((h["fitness"], m) for m, h in st["hof"].items() if h["family"] == fam and h.get("fitness") is not None),
               default=None)
    if best is None:
        return {}, None
    return dict(st["hof"][best[1]]["spec"]["params"]), best[1]


def collect(p, ev, params, n_tables, seed, path, eps=0.25):
    rng = random.Random(seed)
    plan = ev.pool_plan(n_tables, seed)
    coll = {"path": TREE, "params": dict(params, eps=eps, model=""), "name": "collector"}
    jobs = []
    for opps, s, _ in plan:
        lineup = [coll] * 4 + list(opps[:4])
        rng.shuffle(lineup)
        jobs.append((lineup, s, {}))
    res = p.map(run_table, jobs, chunksize=max(1, len(jobs) // 64))
    buf = array("f")
    n = 0
    for r in res:
        for e in r["experience"]:
            for row in e or []:
                if len(row) == NCOL:
                    buf.extend(row)
                    n += 1
    with open(path, "wb") as f:
        buf.tofile(f)
    json.dump({"ncol": NCOL, "rows": n}, open(path + ".json", "w"))
    return n


def fit(paths, out, trees=40, depth=4, leaf=300, lr=0.12):
    py = os.path.join(ROOT, "starter-kit", ".venv", "bin", "python")
    cmd = [py, os.path.join(ROOT, "tools", "train_trees.py"), out, "gbdt", str(trees), str(depth), str(leaf), str(lr)] + paths
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()


def gate(p, ev, params, model_rel, n_tables, seed):
    with_model = {"path": TREE, "params": dict(params, eps=0.0, model=model_rel), "name": "tree_new"}
    base = {"path": TREE, "params": dict(params, eps=0.0, model=""), "name": "tree_base"}
    res = evaluate(p, [with_model, base], ev.pool_plan(n_tables, seed))
    a, b = res[0]["points"], res[1]["points"]
    diff = [x - y for x, y in zip(a, b)]
    return mean_ci(a)[0], mean_ci(b)[0], mean_ci(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--tables", type=int, default=2000)
    ap.add_argument("--gate-tables", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--window", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    ev = KitEvolution(a.name)
    ev.state = json.load(open(ev.path("state.json")))
    params, champ = champion_params(ev.state)
    tag = a.tag or "g%d" % (ev.state["gen"] - 1)
    t0 = time.time()
    with Pool(a.workers) as p:
        path = os.path.join(DATA, "%s_%s.bin" % (a.name, tag))
        n = collect(p, ev, params, a.tables, 9000 + ev.state["gen"], path)
        print("collected %d rows around %s in %.0fs" % (n, champ, time.time() - t0), flush=True)
        files = sorted((os.path.join(DATA, f) for f in os.listdir(DATA) if f.startswith(a.name + "_") and f.endswith(".bin")),
                       key=os.path.getmtime)[-a.window:]
        model_rel = "model_%s.json" % tag
        print(fit(files, os.path.join(TREE_DIR, model_rel)), flush=True)
        new_m, base_m, (dm, dci) = gate(p, ev, params, model_rel, a.gate_tables, 777 + ev.state["gen"])
    adopt = dm > 0
    print("gate: with model %.2f, base %.2f, diff %+.2f +- %.2f -> %s (%.0fs)" % (
        new_m, base_m, dm, dci, "ADOPT" if adopt else "reject", time.time() - t0), flush=True)
    rec = {"tag": tag, "rows": n, "champion": champ, "with_model": new_m, "base": base_m, "diff": dm, "ci": dci,
           "adopted": adopt, "t": time.time()}
    with open(os.path.join(ROOT, "results", "kitevo_%s" % a.name, "trees.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    if adopt:
        shutil.copy(os.path.join(TREE_DIR, model_rel), os.path.join(TREE_DIR, "model.json"))
        print("model.json <- %s" % model_rel)


if __name__ == "__main__":
    main()
