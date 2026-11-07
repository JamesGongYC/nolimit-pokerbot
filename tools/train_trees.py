"""Fit per-action Q regressors (runs in the offline venv: numpy + scikit-learn).

usage: train_trees.py out.json algo trees depth min_leaf lr data1.bin [data2.bin ...]
algo: gbdt (HistGradientBoosting: LightGBM/XGBoost-style histogram boosting) | rf (random forest)
Each .bin is float32 rows of [features..., action, reward] with a .json sidecar {"ncol": n}.
"""
import json
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def load(paths):
    arrs = []
    for p in paths:
        ncol = json.load(open(p + ".json"))["ncol"]
        arrs.append(np.fromfile(p, dtype=np.float32).reshape(-1, ncol))
    return np.vstack(arrs)


def hgb_trees(est):
    base = float(np.ravel(est._baseline_prediction)[0])
    trees = []
    for it in est._predictors:
        nodes = it[0].nodes
        trees.append([[int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]), int(n["right"]),
                       bool(n["is_leaf"]), float(n["value"])] for n in nodes])
    return {"base": base, "trees": trees}


def rf_trees(est):
    trees = []
    for t in est.estimators_:
        tr = t.tree_
        nodes = []
        for i in range(tr.node_count):
            leaf = tr.children_left[i] == -1
            nodes.append([int(tr.feature[i]) if not leaf else 0, float(tr.threshold[i]), int(tr.children_left[i]),
                          int(tr.children_right[i]), bool(leaf), float(tr.value[i][0][0])])
        trees.append(nodes)
    return {"base": 0.0, "trees": trees, "mean": True}


def main():
    out, algo, n_trees, depth, leaf, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), \
        int(sys.argv[5]), float(sys.argv[6])
    data = load(sys.argv[7:])
    X, A, Y = data[:, :-2], data[:, -2].astype(int), data[:, -1]
    t0 = time.time()
    res = {"actions": {}, "meta": {"algo": algo, "rows": int(len(data)), "trees": n_trees, "depth": depth,
                                   "min_leaf": leaf, "lr": lr, "per_action": {}}}
    rng = np.random.RandomState(0)
    for a in sorted(set(A.tolist())):
        idx = np.where(A == a)[0]
        if len(idx) < 4 * leaf:
            continue
        rng.shuffle(idx)
        cut = int(len(idx) * 0.85)
        tr, va = idx[:cut], idx[cut:]
        if algo == "rf":
            est = RandomForestRegressor(n_estimators=n_trees, max_depth=depth, min_samples_leaf=leaf,
                                        max_features=0.5, n_jobs=-1, random_state=0)
            est.fit(X[tr], Y[tr])
            ens = rf_trees(est)
        else:
            est = HistGradientBoostingRegressor(max_iter=n_trees, max_depth=depth, min_samples_leaf=leaf,
                                                learning_rate=lr, l2_regularization=1.0, random_state=0)
            est.fit(X[tr], Y[tr])
            ens = hgb_trees(est)
        pred = est.predict(X[va])
        var = float(np.var(Y[va])) or 1.0
        r2 = 1.0 - float(np.mean((pred - Y[va]) ** 2)) / var
        res["actions"][str(a)] = ens
        res["meta"]["per_action"][str(a)] = {"n": int(len(idx)), "val_r2": round(r2, 4),
                                             "mean_reward": round(float(Y[idx].mean()), 4)}
        print("action %d: n=%d val_r2=%.4f mean=%.3f" % (a, len(idx), r2, Y[idx].mean()), flush=True)
    res["meta"]["fit_seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), separators=(",", ":"))
    print("wrote", out)


if __name__ == "__main__":
    main()
