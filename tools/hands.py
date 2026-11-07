"""Analyse a downloaded round hand history (participant format, see
starter-kit/hand-history/README.md).

  python3 tools/hands.py results/hands/practice1.jsonl.gz [--json out.json]

Reports: our automatic (failed) decisions, payouts seen, our results, and per
opponent team (by entry -> team name): hands, VPIP, PFR, preflop all-in rate,
postflop aggression, fold-to-bet, showdown rate, average points as observed at
our tables. Streams the file (concatenated gzip members).
"""
import argparse
import collections
import gzip
import json
import sys


def stream(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        header = json.loads(f.readline())
        while True:
            a = f.readline()
            if not a:
                break
            b = f.readline()
            yield header, json.loads(a), json.loads(b)["you"]


def analyse(path):
    teams = None
    me = None
    T = collections.defaultdict(lambda: collections.Counter())
    pay = collections.Counter()
    ours = collections.Counter()
    auto = []
    n_tables = 0
    for header, table, own in stream(path):
        if teams is None:
            teams = {e["entry_id"] if isinstance(e, dict) else e: (e.get("team_name") if isinstance(e, dict) else e)
                     for e in (header["entries"] if isinstance(header["entries"], list) else
                               [{"entry_id": k, "team_name": v} for k, v in header["entries"].items()])}
            me = header["you"]
        n_tables += 1
        seats = table["seats"]
        fillers = table.get("fillers", [False] * len(seats))
        pay[tuple(table["payouts"])] += 1
        my = own["seat"]
        if own.get("automatic"):
            auto += [(table["iteration"], table["table"], h, i) for h, i in own["automatic"]]
        hands = table["hands"]
        if hands:
            last = hands[-1]
            if not fillers[my]:
                ours["tables"] += 1
                ours["points"] += last["equity_after"][my]
                ours["hands"] += len(hands)
        for hand in hands:
            ev = hand["events"]
            pre = True
            vp, pr, seen = set(), set(), set()
            cur = 0
            for e in ev:
                k = e.get("kind")
                s = e.get("seat")
                if k == "board":
                    pre = False
                    cur = 0
                    continue
                if s is None or k not in ("fold", "check", "call", "raise"):
                    if k == "blind":
                        cur = max(cur, e.get("amount", 0))
                    continue
                name = seats[s]
                c = T[name]
                facing = cur > 0 and k in ("fold", "call", "raise")
                if pre:
                    seen.add(name)
                    if k in ("call", "raise"):
                        vp.add(name)
                    if k == "raise":
                        pr.add(name)
                        if e.get("amount", 0) >= hand["stacks_before"][s]:
                            c["pf_allin"] += 1
                    c["pf_dec"] += 1
                    if k == "fold":
                        c["pf_fold"] += 1
                else:
                    c["post_dec"] += 1
                    if k == "raise":
                        c["post_aggr"] += 1
                    elif k == "call":
                        c["post_call"] += 1
                    if facing:
                        c["post_faced"] += 1
                        if k == "fold":
                            c["post_fold"] += 1
                if k == "raise":
                    cur = max(cur, e.get("amount", 0))
            for e in ev:
                if e.get("kind") == "show" and e.get("seat") is not None:
                    T[seats[e["seat"]]]["showdowns"] += 1
            for name in seen:
                T[name]["hands"] += 1
            for name in vp:
                T[name]["vpip"] += 1
            for name in pr:
                T[name]["pfr"] += 1
        if hands:
            last = hands[-1]
            for s, name in enumerate(seats):
                if not fillers[s]:
                    T[name]["tables"] += 1
                    T[name]["points"] += last["equity_after"][s]
    rows = []
    for name, c in T.items():
        h = max(1, c["hands"])
        rows.append({
            "entry": name, "team": teams.get(name, name) if teams else name, "is_me": name == me,
            "tables": c["tables"], "pts": round(c["points"] / max(1, c["tables"]), 2),
            "hands": c["hands"], "vpip": round(c["vpip"] / h, 3), "pfr": round(c["pfr"] / h, 3),
            "pf_allin": round(c["pf_allin"] / h, 3), "pf_fold": round(c["pf_fold"] / max(1, c["pf_dec"]), 3),
            "post_aggr": round(c["post_aggr"] / max(1, c["post_dec"]), 3),
            "fold_to_bet": round(c["post_fold"] / max(1, c["post_faced"]), 3),
            "wtsd": round(c["showdowns"] / h, 3)})
    rows.sort(key=lambda r: -r["pts"])
    return {"tables": n_tables, "me": me, "ours": dict(ours), "automatic": auto, "payouts": pay.most_common(10),
            "teams": rows}


def archetype(r):
    if r["pf_allin"] > 0.25:
        return "jammer"
    if r["vpip"] < 0.08:
        return "nit/folder"
    if r["vpip"] > 0.6 and r["pfr"] < 0.15:
        return "station"
    if r["pfr"] > 0.45:
        return "maniac"
    if r["vpip"] > 0.4:
        return "loose"
    return "tight-ish"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = analyse(a.path)
    if a.json:
        json.dump(d, open(a.json, "w"), indent=1)
    o = d["ours"]
    print("tables in file %d; our scored tables %d, mean points %.2f, mean hands/table %.1f" % (
        d["tables"], o.get("tables", 0), o.get("points", 0) / max(1, o.get("tables", 1)),
        o.get("hands", 0) / max(1, o.get("tables", 1))))
    print("our automatic (failed) decisions: %d %s" % (len(d["automatic"]), d["automatic"][:5]))
    print("payout structures seen:", d["payouts"][:5])
    print("\n%-26s %6s %6s %6s %6s %7s %7s %7s %6s %6s  %s" % (
        "team", "tables", "pts", "vpip", "pfr", "pfAllin", "pfFold", "postAgg", "fold2b", "wtsd", "archetype"))
    for r in d["teams"]:
        print("%-26s %6d %6.2f %6.3f %6.3f %7.3f %7.3f %7.3f %6.3f %6.3f  %s%s" % (
            r["team"][:26], r["tables"], r["pts"], r["vpip"], r["pfr"], r["pf_allin"], r["pf_fold"], r["post_aggr"],
            r["fold_to_bet"], r["wtsd"], archetype(r), "  <== US" if r["is_me"] else ""))
    arch = collections.Counter(archetype(r) for r in d["teams"] if not r["is_me"])
    print("\nfield archetypes (teams we met):", dict(arch))


if __name__ == "__main__":
    main()
