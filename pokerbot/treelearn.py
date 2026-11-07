"""Fitted policy iteration with tree ensembles (see strategies/treebot.py).

round r: collect exploring samples around policy pi_r -> fit Q trees on the
last `window` rounds of data -> pi_{r+1} = TreeBot(Q) -> matched evaluation
of pi_{r+1} vs pi_r and vs the original base on fresh training seeds.
Training runs in the offline venv (.venv-train) via tools/train_trees.py.
"""
import json
import os
import subprocess
import time
from array import array

from .sim import Runner, load_fields, matched_jobs, evaluate_candidates, mean_ci
from .strategies import resolve
from .strategies.treebot import FEATURE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv-train", "bin", "python")


def collect(runner, base_spec, tables, eps, seed, fields, path):
    spec = {"type": "collector", "name": "collector", "params": {"base": base_spec, "eps": eps}}
    plan = matched_jobs(tables, fields, seed)
    r = evaluate_candidates(runner, [spec], plan, collect=True)[0]
    buf = array("f")
    n = 0
    for rows in r["experience"]:
        for row in rows or []:
            buf.extend(row)
            n += 1
    with open(path, "wb") as f:
        buf.tofile(f)
    json.dump({"ncol": len(FEATURE_NAMES) + 2, "rows": n}, open(path + ".json", "w"))
    return n, r["cpu"]


def fit(data_paths, out, algo="gbdt", trees=120, depth=5, leaf=400, lr=0.08, log=print):
    cmd = [VENV_PY, os.path.join(ROOT, "tools", "train_trees.py"), out, algo, str(trees), str(depth), str(leaf),
           str(lr)] + list(data_paths)
    t0 = time.time()
    outp = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    log(outp.strip())
    return time.time() - t0


def tree_spec(base_spec, model, name, margin=0.0, streets="all"):
    b = resolve(base_spec)
    params = dict(b.get("params", {}))
    params.update({"model": model, "margin": margin, "tree_streets": streets})
    return {"type": "tree", "name": name, "params": params}


def iterate(base, name="tree", rounds=3, tables=3000, eps=0.12, algo="gbdt", trees=120, depth=5, leaf=400,
            lr=0.08, window=2, eval_tables=1500, margin=0.0, seed=31, workers=None, event=None, log=print):
    from .event import Event
    event = event or Event.load()
    fields = load_fields(os.path.join(ROOT, "configs", "populations", "train.json"))
    base_spec = resolve(json.load(open(base)) if isinstance(base, str) and base.endswith(".json") else base)
    data_dir = os.path.join(ROOT, "results", "treedata")
    model_dir = os.path.join(ROOT, "pokerbot", "data")
    for d in (data_dir, model_dir):
        if not os.path.isdir(d):
            os.makedirs(d)
    t0, cpu, fit_s = time.time(), 0.0, 0.0
    policy = base_spec
    paths, history = [], []
    best = (None, -1e9)
    with Runner(event, workers) as runner:
        for r in range(rounds):
            p = os.path.join(data_dir, "%s_r%d.bin" % (name, r))
            n, c = collect(runner, policy, tables, eps, seed * 100 + r, fields, p)
            cpu += c
            paths.append(p)
            log("round %d: collected %d samples around %s (%.0fs)" % (r, n, policy["name"], time.time() - t0))
            model_rel = os.path.join("pokerbot", "data", "%s_r%d.json" % (name, r))
            fit_s += fit(paths[-window:], os.path.join(ROOT, model_rel), algo, trees, depth, leaf, lr, log)
            cands = [tree_spec(base_spec, model_rel, "%s_r%d" % (name, r), margin),
                     tree_spec(base_spec, model_rel, "%s_r%d_pre" % (name, r), margin, "pre"),
                     policy] + ([base_spec] if policy is not base_spec else [])
            plan = matched_jobs(eval_tables, fields, seed * 7777 + r)
            res = evaluate_candidates(runner, cands, plan)
            row = {"round": r, "samples": n}
            for cs, rr in zip(cands, res):
                cpu += rr["cpu"]
                m, ci = mean_ci(rr["points"])
                row[cs["name"]] = (round(m, 2), round(ci, 2))
                if cs["type"] == "tree" and m > best[1]:
                    best = (cs, m)
            history.append(row)
            log("round %d eval: %s" % (r, json.dumps(row)))
            policy = cands[0] if row[cands[0]["name"]][0] >= row[cands[1]["name"]][0] else cands[1]
    best_spec = dict(best[0])
    best_spec["train_cpu_s"] = round(cpu + fit_s * 8, 1)
    out = {"history": history, "best": best_spec, "wall_s": round(time.time() - t0, 1), "fit_wall_s": round(fit_s, 1),
           "settings": {"rounds": rounds, "tables": tables, "eps": eps, "algo": algo, "trees": trees,
                        "depth": depth, "leaf": leaf, "lr": lr, "window": window}}
    rd = os.path.join(ROOT, "results", "opt_%s" % name)
    if not os.path.isdir(rd):
        os.makedirs(rd)
    json.dump(out, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    json.dump(best_spec, open(os.path.join(rd, "best.json"), "w"), indent=1)
    log("best %s %.2f -> %s" % (best_spec["name"], best[1], os.path.join(rd, "best.json")))
    return best_spec
