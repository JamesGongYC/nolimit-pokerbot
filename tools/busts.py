"""Where and how a bot busts (participant hand history), for us or any team.

  python3 tools/busts.py <round>.jsonl.gz [US | "Team Name" ...]

Per finishing place: tables, average bust hand / stack (BB) / players alive,
how the chips went in (open shove, re-shove over a raise, called a shove,
open raise, postflop all-in, blind), and at the bust showdown our retained
two cards vs the winner's (pair / unpaired, higher / lower) -- public events
only, so the same analysis applies to any team at our tables.
"""
import collections
import gzip
import json
import sys

RANKS = "23456789TJQKA"


def stream(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        while True:
            a = f.readline()
            if not a:
                break
            b = f.readline()
            yield header, json.loads(a), json.loads(b)["you"]


def bust_info(t, s):
    hands = t["hands"]
    for i, h in enumerate(hands):
        if h["stacks_before"][s] > 0 and h["stacks_after"][s] == 0:
            return i
    return None


def classify(h, s):
    ev = h["events"]
    street, prior_raise, how = 0, False, "blind/forced"
    for e in ev:
        k = e.get("kind")
        if k == "board":
            street += 1
            continue
        if k not in ("raise", "call"):
            continue
        if e.get("seat") == s:
            if street == 0:
                if k == "raise" and not prior_raise:
                    how = "open shove" if e.get("amount", 0) >= h["stacks_before"][s] * 0.9 else "open raise"
                elif k == "raise":
                    how = "re-shove/3bet over a raise"
                else:
                    how = "called a raise/shove"
            else:
                how = "postflop all-in"
        elif k == "raise" and street == 0:
            prior_raise = True
    shows = {e["seat"]: e.get("cards", []) for e in ev if e.get("kind") == "show"}
    mine = shows.get(s, [])
    win = [e["seat"] for e in ev if e.get("kind") == "win"]
    opp = next((shows[w] for w in win if w != s and w in shows), [])
    rk = lambda cs: sorted((RANKS.index(c[0]) for c in cs), reverse=True)
    tag = "no showdown"
    if len(mine) == 2 and len(opp) == 2:
        mp, op = mine[0][0] == mine[1][0], opp[0][0] == opp[1][0]
        if mp and not op:
            tag = "our pair vs their unpaired (outdrawn)"
        elif op and not mp:
            tag = "their pair vs our unpaired"
        elif op and mp:
            tag = "pair vs pair, ours %s" % ("higher" if rk(mine) > rk(opp) else "lower")
        else:
            tag = "both unpaired, %s" % ("ours higher" if rk(mine) > rk(opp) else "theirs higher")
    return how, tag, mine, opp


def analyse(path, who):
    places = collections.Counter()
    hows = collections.defaultdict(collections.Counter)
    tags = collections.defaultdict(collections.Counter)
    depth = collections.defaultdict(list)
    handno = collections.defaultdict(list)
    for header, t, own in stream(path):
        names = {e["entry_id"]: e["team_name"] for e in header["entries"]}
        seats = [own["seat"]] if who == "US" else [k for k, e in enumerate(t["seats"]) if names.get(e) == who]
        for s in seats:
            if t["fillers"][s] or not t["hands"]:
                continue
            b = bust_info(t, s)
            if b is None:
                last = t["hands"][-1]["stacks_after"]
                places[1 + sum(1 for x in last if x > last[s])] += 1
                continue
            h = t["hands"][b]
            alive = sum(1 for x in h["stacks_before"] if x > 0)
            same = [k for k, x in enumerate(h["stacks_before"]) if x > 0 and h["stacks_after"][k] == 0]
            place = alive - len(same) + 1 + sum(1 for k in same if h["stacks_before"][k] > h["stacks_before"][s])
            places[place] += 1
            how, tag, _, _ = classify(h, s)
            hows[place][how] += 1
            tags[place][tag] += 1
            depth[place].append(h["stacks_before"][s] / float(h["blinds"][1] or 1))
            handno[place].append(h["hand"])
    return places, hows, tags, depth, handno


def report(path, who):
    places, hows, tags, depth, handno = analyse(path, who)
    n = float(sum(places.values()))
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else "-"
    print("\n######## %s: %d tables; places %%: %s" % (who, n, {k: round(100 * v / n, 1) for k, v in sorted(places.items())}))
    early = sum(places[p] for p in (6, 7, 8))
    allh = collections.Counter()
    for p in (6, 7, 8):
        allh.update(hows[p])
    print("  early busts (6th-8th): %.1f%% of tables; how: %s" % (
        100 * early / n, {k: "%d%%" % round(100.0 * v / max(1, early)) for k, v in allh.most_common()}))
    for p in sorted(hows):
        tot = sum(hows[p].values())
        print("  place %d: n=%d, bust hand #%s, stack %s BB | %s | %s" % (
            p, tot, avg(handno[p]), avg(depth[p]),
            {k: "%d%%" % round(100.0 * v / tot) for k, v in hows[p].most_common(3)},
            {k: v for k, v in tags[p].most_common(3)}))


if __name__ == "__main__":
    for who in (sys.argv[2:] or ["US"]):
        report(sys.argv[1], who)
