"""Verify a trained transformer bot: numpy == torch, forward latency, and a
paired simulation vs a reference bot on the top-team replicas (kit venv).

  tools/tfverify.py model.npz bot_dir reference_bot_dir
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tftrain  # noqa: E402
from kitsim import Pool, evaluate, matched_plan, mean_ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    model_path, bot, ref = sys.argv[1], sys.argv[2], sys.argv[3]
    sp = importlib.util.spec_from_file_location("tfbot_v", os.path.join(bot, "agent.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    W = np.load(model_path)
    C, H = W["ctx.weight"].shape[1], W["hpos"].shape[0]
    tm = tftrain.Model(C, H)
    tm.load_state_dict(torch.load(model_path + ".pt"))
    tm.eval()
    rng = np.random.RandomState(0)
    cards = np.array([rng.choice(52, 3, replace=False) for _ in range(100)])
    ranks, suits = cards // 4, tftrain.suit_pattern(cards)
    ctx = rng.rand(100, C).astype(np.float32)
    hist = rng.rand(100, H, 4).astype(np.float32)
    hist[:, :4] = 0
    hm = np.abs(hist).sum(-1) == 0
    with torch.no_grad():
        qt = tm(torch.from_numpy(ranks), torch.from_numpy(suits), torch.from_numpy(ctx), torch.from_numpy(hist),
                torch.from_numpy(hm)).numpy()
    qn = np.array([m.forward(ranks[i], suits[i], ctx[i].astype(np.float64), hist[i].astype(np.float64), hm[i])
                   for i in range(100)])
    t = []
    for i in range(1000):
        j = i % 100
        t0 = time.perf_counter()
        m.forward(ranks[j], suits[j], ctx[j].astype(np.float64), hist[j].astype(np.float64), hm[j])
        t.append((time.perf_counter() - t0) * 1e6)
    t.sort()
    print("numpy-torch max diff %.1e; forward p50 %.0fus p99 %.0fus" % (float(np.abs(qn - qt).max()), t[500], t[990]))
    reps = json.load(open(os.path.join(ROOT, "results", "kitevo_main", "replicas.json")))["replicas"]
    top = sorted([r for r in reps if r["weight"] >= 40], key=lambda r: -r["points_real"])[:10]
    cands = [{"path": os.path.join(bot, "agent.py")}, {"path": os.path.join(ref, "agent.py")}]
    with Pool(6) as p:
        res = evaluate(p, cands, matched_plan(240, [r["spec"] for r in top], int(time.time())))
    a, b = res[0]["points"], res[1]["points"]
    dm, dci = mean_ci([x - y for x, y in zip(a, b)])
    pl = res[0]["places"]
    lat = sorted(res[0]["lat"])
    print("sim vs %s: %.2f vs %.2f, diff %+.2f +- %.2f, 1st %.0f%%, fails %d, p99 %.0fus" % (
        os.path.basename(ref), mean_ci(a)[0], mean_ci(b)[0], dm, dci, 100.0 * pl.count(1) / len(pl), res[0]["failures"],
        lat[int(0.99 * len(lat))] * 1000 if lat else -1))


if __name__ == "__main__":
    main()
