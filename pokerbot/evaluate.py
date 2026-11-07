"""Hold-out evaluation and leaderboard.

Candidates play identical hold-out tables (same seeds, opponents, seat),
so differences are paired. Uncertainty: 95% CI over independent tables;
paired CI of each candidate minus the leader. Field breakdown gives
sensitivity to opponent composition.
"""
import json
import os
import subprocess
import sys

from .sim import (Runner, load_fields, matched_jobs, evaluate_candidates, mean_ci, summarize,
                  tournament, place_dist, SeatStats)
from .strategies import resolve, MECHANISM

HOLDOUT_SEED = 777000111


def load_spec(s):
    if isinstance(s, dict):
        return s
    if s.endswith(".json") and os.path.exists(s):
        spec = json.load(open(s))
        if os.path.basename(s) == "best.json":
            d = os.path.basename(os.path.dirname(os.path.abspath(s)))
            spec["name"] = "T-" + (d[4:] if d.startswith("opt_") else d)  # tuned result, named by run
        return spec
    return s


def memory_probe(spec):
    """Peak Python allocation (tracemalloc) and max RSS for one agent playing one table,
    measured in a fresh interpreter."""
    code = r"""
import json, sys, tracemalloc, resource
tracemalloc.start()
from pokerbot.event import Event
from pokerbot.engine import play_table
from pokerbot.strategies import make_agent
spec = json.loads(sys.argv[1])
ev = Event.load()
agents = [make_agent(spec, 1)] + [make_agent("station", i) for i in range(7)]
play_table(ev, agents, 1)
cur, peak = tracemalloc.get_traced_memory()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"py_peak_kb": peak // 1024, "rss_mb": round(rss / (1024 * 1024.0) if sys.platform == 'darwin' else rss / 1024.0, 1)}))
"""
    try:
        out = subprocess.check_output([sys.executable, "-c", code, json.dumps(resolve(spec))],
                                      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      stderr=subprocess.STDOUT, timeout=120)
        return json.loads(out.decode().strip().splitlines()[-1])
    except Exception as e:  # pragma: no cover
        return {"error": str(e)[:200]}


def leaderboard(candidates, fields_path=None, tables=2000, seed=HOLDOUT_SEED, workers=None, event=None,
                out=None, memory=True, log=print):
    from .event import Event
    event = event or Event.load()
    fields = load_fields(fields_path or os.path.join("configs", "populations", "holdout.json"))
    specs = [resolve(load_spec(c)) for c in candidates]
    plan = matched_jobs(tables, fields, seed)
    with Runner(event, workers) as runner:
        res = evaluate_candidates(runner, specs, plan)
    rows = []
    for spec, r in zip(specs, res):
        d = summarize(spec["name"], r["points"], r["places"], r["stats"], r["cpu"],
                      train_cost=spec.get("train_cpu_s"), mechanism=MECHANISM.get(spec["type"]))
        d["by_field"] = {k: round(sum(v) / len(v), 2) for k, v in sorted(r["by_field"].items())}
        d["worst_field"] = min(d["by_field"].values())
        d["_pts"] = r["points"]
        d["spec"] = spec
        rows.append(d)
    rows.sort(key=lambda d: -d["points"])
    lead = rows[0]["_pts"]
    for d in rows:
        diff = [a - b for a, b in zip(d["_pts"], lead)]
        m, ci = mean_ci(diff)
        d["vs_leader"] = round(m, 3)
        d["vs_leader_ci95"] = round(ci, 3)
        del d["_pts"]
        if memory:
            d["memory"] = memory_probe(d["spec"])
    report = {"tables": tables, "seed": seed, "fields": sorted(fields), "event": event.name,
              "payouts": event.payouts, "rows": rows}
    if out:
        d = os.path.dirname(out)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        json.dump(report, open(out, "w"), indent=1)
        open(out.replace(".json", ".md"), "w").write(render_md(report))
    log(render_md(report))
    return report


def render_md(rep):
    L = []
    L.append("# Leaderboard (%s, %d hold-out tables, payouts %s)\n" % (rep["event"], rep["tables"], rep["payouts"]))
    L.append("Points = mean official points per table (one candidate seat, 7 hold-out opponents); "
             "CI = 95%% over independent tables; vs leader = paired difference on identical tables.\n")
    L.append("| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, d in enumerate(rep["rows"]):
        pdist = d["place_dist"]
        mem = d.get("memory", {}).get("py_peak_kb", "")
        L.append("| %d | %s | %s | %.2f | %.2f | %+.2f (%.2f) | %.2f | %.0f/%.0f/%.0f | %d/%d/%.0f | %d/%d/%d | %s | %s |" % (
            i + 1, d["name"], d.get("mechanism", ""), d["points"], d["ci95"], d["vs_leader"],
            d["vs_leader_ci95"], d["worst_field"], 100 * pdist[0], 100 * pdist[1], 100 * pdist[2],
            d["lat_p50_us"], d["lat_p99_us"], d["lat_max_us"], d["errors"], d["timeouts"], d["illegal"],
            mem, d.get("train_cpu_s", "-")))
    L.append("\n## Field sensitivity (mean points by hold-out field)\n")
    fields = rep["fields"]
    L.append("| strategy | " + " | ".join(fields) + " |")
    L.append("|---|" + "---|" * len(fields))
    for d in rep["rows"]:
        L.append("| %s | " % d["name"] + " | ".join("%.2f" % d["by_field"].get(f, 0) for f in fields) + " |")
    L.append("\n## Behaviour (verifies styles differ)\n")
    keys = ["vpip", "pfr", "open", "steal", "3bet", "af", "fold_to_bet", "wtsd", "allin_per_hand"]
    L.append("| strategy | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for d in rep["rows"]:
        L.append("| %s | " % d["name"] + " | ".join(str(d["behavior"][k]) for k in keys) + " |")
    return "\n".join(L) + "\n"


def official(entries, iterations=2000, tournaments=6, seed=424242, workers=None, event=None, log=print, out=None):
    """Official-format simulation: N entries, random tables with fillers,
    six scored tournaments weighted 1..6."""
    from .event import Event
    event = event or Event.load()
    specs = [resolve(load_spec(e)) for e in entries]
    names, seen = [], {}
    for s in specs:
        k = seen[s["name"]] = seen.get(s["name"], 0) + 1
        names.append(s["name"] if k == 1 else "%s#%d" % (s["name"], k))
    weights = event.weights[:tournaments]
    per_t = []
    with Runner(event, workers) as runner:
        for t in range(len(weights)):
            r = tournament(runner, specs, iterations, seed + 1000 * t, event.table_size)
            per_t.append(r)
            log("T%d done" % (t + 1))
    rows = []
    for i, n in enumerate(names):
        scores = [sum(r["points"][i]) / float(iterations) for r in per_t]
        allpts = [x for r in per_t for x in r["points"][i]]
        m, ci = mean_ci(allpts)
        st = SeatStats()
        for r in per_t:
            st.merge(r["stats"][i])
        final = sum(w * s for w, s in zip(weights, scores))
        rows.append({"name": n, "final": round(final, 2), "per_tournament": [round(s, 2) for s in scores],
                     "mean": round(m, 3), "ci95": round(ci, 3),
                     "final_ci95": round(ci * sum(weights), 2),
                     "place_dist": place_dist([x for r in per_t for x in r["places"][i]])})
    rows.sort(key=lambda d: -d["final"])
    L = ["# Official-format simulation: %d entries, %d iterations x %d tournaments (weights %s)\n" % (
        len(specs), iterations, len(weights), weights),
         "| # | entry | final (weighted) | ± | mean/table | T1..T6 | 1st% |", "|---|---|---|---|---|---|---|"]
    for k, d in enumerate(rows):
        L.append("| %d | %s | %.1f | %.1f | %.2f | %s | %.1f |" % (k + 1, d["name"], d["final"], d["final_ci95"],
                                                            d["mean"], d["per_tournament"], 100 * d["place_dist"][0]))
    md = "\n".join(L) + "\n"
    log(md)
    if out:
        json.dump({"rows": rows, "iterations": iterations}, open(out, "w"), indent=1)
        open(out.replace(".json", ".md"), "w").write(md)
    return rows


def crossplay(strategies, tables=60, seed=5150, workers=None, event=None, out=None, log=print):
    """1-vs-7 cross-play: cell (A, B) = mean points of one A seated with seven
    copies of B (seat rotated; seeds shared down each column B, so rows are
    paired). Summaries: mean over opponents, worst opponent, and a replicator-
    dynamics population share (linear invasion approximation)."""
    from .event import Event
    event = event or Event.load()
    specs = [resolve(load_spec(s)) for s in strategies]
    names = [s["name"] for s in specs]
    n = len(specs)
    jobs, meta = [], []
    for bi, b in enumerate(specs):
        for t in range(tables):
            sd = (seed * 1000003 + bi * 7919 + t) & 0x3FFFFFFF
            for ai, a in enumerate(specs):
                seat = t % event.table_size
                lineup = [b] * (event.table_size - 1)
                lineup.insert(seat, a)
                jobs.append((lineup, sd, None))
                meta.append((ai, bi, seat))
    with Runner(event, workers) as runner:
        res = runner.run(jobs)
    acc = [[[] for _ in range(n)] for _ in range(n)]
    for (ai, bi, seat), r in zip(meta, res):
        acc[ai][bi].append(r["points"][seat])
    M = [[sum(acc[a][b]) / len(acc[a][b]) for b in range(n)] for a in range(n)]
    # replicator dynamics on the invasion payoff matrix
    x = [1.0 / n] * n
    for _ in range(5000):
        fit = [sum(M[a][b] * x[b] for b in range(n)) for a in range(n)]
        avg = sum(f * xi for f, xi in zip(fit, x))
        x = [xi * (f / avg) if avg > 0 else xi for xi, f in zip(x, fit)]
        s = sum(x)
        x = [xi / s for xi in x]
    rows = []
    for a in range(n):
        others = [M[a][b] for b in range(n) if b != a]
        rows.append({"name": names[a], "mean_vs_others": round(sum(others) / len(others), 2),
                     "worst": round(min(others), 2), "worst_vs": names[min((b for b in range(n) if b != a),
                                                                           key=lambda b: M[a][b])],
                     "meta_share": round(x[a], 3)})
    order = sorted(range(n), key=lambda a: -rows[a]["mean_vs_others"])
    L = ["# Cross-play matrix (1 row-strategy vs 7 copies of column-strategy, %d tables/cell)\n" % tables,
         "Cell = mean official points of the single row bot (12.5 = average seat). "
         "Meta share = replicator-dynamics equilibrium of this invasion game (approximation).\n",
         "| # | strategy | mean vs others | worst (vs) | meta share |", "|---|---|---|---|---|"]
    for k, a in enumerate(order):
        r = rows[a]
        L.append("| %d | %s | %.2f | %.2f (%s) | %.3f |" % (k + 1, r["name"], r["mean_vs_others"], r["worst"],
                                                          r["worst_vs"], r["meta_share"]))
    top = order[:min(n, 30)]
    L.append("\n## Matrix (top %d rows/cols in leaderboard order; S# = rank above)\n" % len(top))
    L.append("| row \\ col | " + " | ".join("S%d" % (k + 1) for k in range(len(top))) + " |")
    L.append("|---|" + "---|" * len(top))
    for k, a in enumerate(top):
        L.append("| S%d %s | " % (k + 1, names[a]) + " | ".join("%.1f" % M[a][b] for b in top) + " |")
    md = "\n".join(L) + "\n"
    log(md)
    if out:
        json.dump({"names": names, "matrix": M, "rows": rows, "tables": tables}, open(out, "w"), indent=1)
        open(out.replace(".json", ".md"), "w").write(md)
    return rows, M
