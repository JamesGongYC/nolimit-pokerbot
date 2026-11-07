"""Every `--every` minutes, alongside the evolution run: shortlist verdict ->
submit the newest winner (run with the kit venv).

Each cycle:
  1. snapshot results/kitevo_<name>/state.json;
  2. shortlist = top K hall-of-fame members per family by in-run fitness, plus
     the currently ACTIVE bot; official all-vs-all format (tables of 8, short
     table filled with non-seated entries, fillers never score), `--iterations`;
  3. if the winner is not the active bot and beats it by more than `--min-gain`
     points/table AND by half the combined 95% interval: export it to
     bots/auto/<member>, run `poker-agent check`, a latency/failure smoke test,
     pack, upload, and select it as the active bot (tools/submit.py);
  4. log everything to results/kitevo_<name>/autosubmit.log(.jsonl).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitsim import BASE_FIELD, Pool, evaluate, lat_summary, matched_plan, mean_ci  # noqa: E402
from kitevolve import ROOT, export, official  # noqa: E402

KIT = os.path.join(ROOT, "starter-kit")
PY = os.path.join(KIT, ".venv", "bin", "python")
POKER_AGENT = os.path.join(KIT, ".venv", "bin", "poker-agent")


def log(d, msg):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(d, "autosubmit.log"), "a") as f:
        f.write(line + "\n")


def shortlist(st, k):
    by = {}
    for m, h in st["hof"].items():
        if h.get("fitness") is None:
            continue
        by.setdefault(h["family"], []).append((h["fitness"], h["gen"], m))
    out = []
    for fam, lst in by.items():
        lst.sort(reverse=True)
        out += [m for _, _, m in lst[:k]]
    return out


def smoke(bot_dir, workers=1):
    """Kit checker + 24 tables vs the baseline field: failures and latency."""
    chk = subprocess.run([POKER_AGENT, "check", os.path.join(bot_dir, "agent.py")], capture_output=True, text=True,
                         cwd=KIT)
    if chk.returncode != 0:
        return False, {"check": chk.stdout[-500:] + chk.stderr[-500:]}
    with Pool(workers) as p:
        r = evaluate(p, [{"path": os.path.join(bot_dir, "agent.py"), "name": "cand"}],
                     matched_plan(24, BASE_FIELD, 4242))[0]
    lat = lat_summary(r["lat"])
    ok = r["failures"] == 0 and lat.get("p50_us", 1e9) < 700
    return ok, {"failures": r["failures"], "lat": lat, "points": round(sum(r["points"]) / len(r["points"]), 2)}


def cycle(a, d, state_file):
    st = json.load(open(os.path.join(d, "state.json")))
    act = json.load(open(state_file)) if os.path.exists(state_file) else {"member": "jj_v2", "dir": "bots/jj_v2"}
    names = shortlist(st, a.shortlist)
    if not names:
        log(d, "no promoted members yet")
        return
    entries = [st["hof"][m]["spec"] for m in names]
    names.append("ACTIVE:" + act["member"])
    entries.append({"path": os.path.join(ROOT, act["dir"], "agent.py"), "name": names[-1]})
    t0 = time.time()
    if a.field == "top":
        # every candidate plays identical tables vs replicas of the top real teams (CRN, paired)
        reps = json.load(open(os.path.join(d, "replicas.json")))["replicas"]
        top = sorted([r for r in reps if r["weight"] >= 40], key=lambda r: -r["points_real"])[:a.top_n]
        field = [r["spec"] for r in top]
        plan = matched_plan(a.tables, field, int(time.time()))
        with Pool(a.workers) as p:
            res_ = evaluate(p, entries, plan)
        act_pts = res_[-1]["points"]
        rows = []
        for n, r in zip(names, res_):
            m, ci = mean_ci(r["points"])
            dm, dci = mean_ci([x - y for x, y in zip(r["points"], act_pts)])
            rows.append((m, ci, n, dm, dci))
        rows.sort(key=lambda x: -x[0])
        log(d, "TOP-%d field verdict gen %d, %d entries x %d tables, %.0fs: %s" % (
            len(field), st["gen"] - 1, len(names), a.tables, time.time() - t0,
            json.dumps([(n, round(m, 2), round(dm, 2), round(dci, 2)) for m, ci, n, dm, dci in rows[:6]])))
        with open(os.path.join(d, "autosubmit.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "gen": st["gen"] - 1, "field": "top%d" % len(field),
                                "rows": rows, "active": act["member"]}) + "\n")
        best_m, best_ci, best, gain, gci = rows[0]
        if best.startswith("ACTIVE:"):
            log(d, "active bot %s still wins vs the top teams; no submission" % act["member"])
            return
        need = max(a.min_gain, 0.5 * gci)
        if gain <= need:
            log(d, "winner %s paired +%.2f +- %.2f over active (needs > %.2f); no submission" % (best, gain, gci, need))
            return
        return submit(a, d, st, act, best, gain, state_file)
    with Pool(a.workers) as p:
        pts = official(p, entries, a.iterations, int(time.time()))
    rows = sorted(((mean_ci(x)[0], mean_ci(x)[1], n) for x, n in zip(pts, names)), reverse=True)
    res = {n: (round(m, 2), round(ci, 2)) for m, ci, n in rows}
    log(d, "verdict gen %d, %d entries, %d iterations, %.0fs: %s" % (
        st["gen"] - 1, len(names), a.iterations, time.time() - t0, json.dumps(rows[:6])))
    best_m, best_ci, best = rows[0]
    am, aci = res[names[-1]]
    with open(os.path.join(d, "autosubmit.jsonl"), "a") as f:
        f.write(json.dumps({"t": time.time(), "gen": st["gen"] - 1, "rows": rows, "active": act["member"]}) + "\n")
    if best.startswith("ACTIVE:"):
        log(d, "active bot %s still wins; no submission" % act["member"])
        return
    gain = best_m - am
    need = max(a.min_gain, 0.5 * math.sqrt(best_ci ** 2 + aci ** 2))
    if gain <= need:
        log(d, "winner %s +%.2f over active (needs > %.2f); no submission" % (best, gain, need))
        return
    return submit(a, d, st, act, best, gain, state_file)


def submit(a, d, st, act, best, gain, state_file):
    out = os.path.join(ROOT, "bots", "auto", best)
    export(a.name, best, out, state=st)
    ok, info = smoke(out)
    log(d, "smoke %s: %s %s" % (best, "OK" if ok else "FAILED", json.dumps(info)))
    if not ok:
        return
    zp = os.path.join(ROOT, "bots", "auto", best + ".zip")
    if os.path.exists(zp):
        os.remove(zp)
    pk = subprocess.run([POKER_AGENT, "pack", out, zp], capture_output=True, text=True, cwd=KIT)
    if pk.returncode != 0:
        log(d, "pack failed: %s" % (pk.stdout[-400:] + pk.stderr[-400:]))
        return
    if a.dry_run:
        log(d, "DRY RUN: would upload %s and activate" % zp)
        return
    up = subprocess.run(["python3", os.path.join(ROOT, "tools", "submit.py"), "upload", zp, "--activate",
                         "--wait", "300"], capture_output=True, text=True, cwd=ROOT)
    log(d, "upload %s: rc %d %s" % (best, up.returncode, up.stdout[-600:].replace("\n", " | ") + up.stderr[-300:]))
    if up.returncode == 0 and "ACTIVE bot is now" in up.stdout:
        json.dump({"member": best, "dir": os.path.relpath(out, ROOT), "since": time.time(), "gain": gain},
                  open(state_file, "w"))
        log(d, "ACTIVE -> %s (+%.2f points/table over %s)" % (best, gain, act["member"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--every", type=float, default=20.0, help="minutes between cycles")
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--shortlist", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.25)
    ap.add_argument("--field", default="top", choices=["top", "self"],
                    help="top: paired tables vs replicas of the top real teams; self: all-vs-all shortlist")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--tables", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--first-now", action="store_true", help="run the first cycle immediately")
    ap.add_argument("--start-at", default=None, help="local HH:MM of the first cycle (today)")
    a = ap.parse_args()
    d = os.path.join(ROOT, "results", "kitevo_%s" % a.name)
    state_file = os.path.join(d, "active.json")
    nxt = time.time() if a.first_now else time.time() + a.every * 60
    if a.start_at:
        hh, mm = (int(x) for x in a.start_at.split(":"))
        lt = time.localtime()
        nxt = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        if nxt < time.time():
            nxt += 86400
    log(d, "autosubmit loop: first cycle at %s, every %.0f min, shortlist %d, %d iterations, dry_run=%s" % (
        time.strftime("%H:%M", time.localtime(nxt)), a.every, a.shortlist, a.iterations, a.dry_run))
    for _ in range(a.cycles):
        while time.time() < nxt:
            time.sleep(5)
        nxt += a.every * 60
        try:
            cycle(a, d, state_file)
        except Exception as e:  # keep the loop alive
            log(d, "cycle error: %r" % (e,))


if __name__ == "__main__":
    main()
