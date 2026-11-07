"""Endgame anatomy of a round: why a bot finishes 2nd/3rd instead of 1st.

  python3 tools/endgame.py results/hands/practice5.jsonl.gz [--teams "Team 72,TeamBallHog"]

Per bot (us + chosen teams), over all their non-filler tables:
  places; P(reach final 3), P(reach heads-up), P(win | heads-up),
  P(win | final 3); chip share when 3-handed / heads-up starts;
  heads-up behaviour: hands per heads-up match, share of hands won without
  showdown, fold rate from the small blind (button) and the big blind, all-in
  rate, showdown win rate; hands-to-bust and chip growth in the 8..4-handed
  phase (who accumulates chips before the endgame).
"""
import argparse
import collections
import gzip
import json


def stream(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        while True:
            a = f.readline()
            if not a:
                break
            f.readline()
            yield header, json.loads(a)


def analyse(path, teams):
    S = collections.defaultdict(lambda: collections.Counter())
    lists = collections.defaultdict(lambda: collections.defaultdict(list))
    me_entry = None
    for header, t in stream(path):
        names = {e["entry_id"]: e["team_name"] for e in header["entries"]}
        me_entry = header["you"]
        seats = t["seats"]
        hands = t["hands"]
        if not hands:
            continue
        for s, ent in enumerate(seats):
            if t["fillers"][s]:
                continue
            name = "US" if ent == me_entry else names.get(ent, ent)
            if name != "US" and name not in teams:
                continue
            c = S[name]
            L = lists[name]
            c["tables"] += 1
            reached3 = reached2 = False
            share3 = share2 = None
            hu_hands = hu_won_nosd = hu_sb_fold = hu_sb = hu_bb_fold = hu_bb = hu_allin = hu_sd = hu_sd_won = 0
            peak_mid = 0
            for h in hands:
                st = h["stacks_before"]
                alive = [i for i, x in enumerate(st) if x > 0]
                if st[s] <= 0:
                    break
                tot = float(sum(st))
                if len(alive) <= 3 and not reached3:
                    reached3, share3 = True, st[s] / tot
                if len(alive) == 2 and not reached2:
                    reached2, share2 = True, st[s] / tot
                if len(alive) >= 4:
                    peak_mid = max(peak_mid, st[s] / tot * len(alive))
                if len(alive) == 2:
                    hu_hands += 1
                    sb = h["button"] == s  # heads-up: button posts the small blind
                    acts = [e for e in h["events"] if e.get("seat") == s and e.get("kind") in ("fold", "check", "call", "raise")]
                    first = acts[0]["kind"] if acts else None
                    if sb:
                        hu_sb += 1
                        hu_sb_fold += first == "fold"
                    else:
                        hu_bb += 1
                        hu_bb_fold += first == "fold"
                    hu_allin += any(e.get("kind") == "raise" and e.get("amount", 0) >= st[s] * 0.95 for e in acts)
                    shows = [e for e in h["events"] if e.get("kind") == "show"]
                    gained = h["stacks_after"][s] - st[s]
                    if shows:
                        hu_sd += 1
                        hu_sd_won += gained > 0
                    elif gained > 0:
                        hu_won_nosd += 1
            last = hands[-1]["stacks_after"]
            won = last[s] == sum(last) or (sum(1 for x in last if x > 0) > 1 and last[s] == max(last))
            pts = hands[-1]["equity_after"][s]
            c["reach3"] += reached3
            c["reach2"] += reached2
            c["win"] += won
            c["win_from2"] += reached2 and won
            c["win_from3"] += reached3 and won
            for k, v in (("hu_hands", hu_hands), ("hu_won_nosd", hu_won_nosd), ("hu_sb", hu_sb), ("hu_sb_fold", hu_sb_fold),
                         ("hu_bb", hu_bb), ("hu_bb_fold", hu_bb_fold), ("hu_allin", hu_allin), ("hu_sd", hu_sd),
                         ("hu_sd_won", hu_sd_won)):
                c[k] += v
            c["pts"] += pts
            if share3 is not None:
                L["share3"].append(share3)
            if share2 is not None:
                L["share2"].append(share2)
                L["share2_won" if won else "share2_lost"].append(share2)
            L["peak_mid"].append(peak_mid)
    return S, lists


def pr(name, c, L):
    n = float(max(1, c["tables"]))
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("\n== %s: %d tables, %.2f pts/table, wins %.1f%%" % (name, c["tables"], c["pts"] / n, 100 * c["win"] / n))
    print("  reach final 3 %.1f%%  -> win from final 3 %.1f%%" % (100 * c["reach3"] / n, 100 * c["win_from3"] / max(1, c["reach3"])))
    print("  reach heads-up %.1f%% -> WIN HEADS-UP %.1f%%" % (100 * c["reach2"] / n, 100 * c["win_from2"] / max(1, c["reach2"])))
    print("  chip share entering 3-handed %.2f (fair .33), entering heads-up %.2f (fair .50); HU start share when it won %.2f, lost %.2f" % (
        avg(L["share3"]), avg(L["share2"]), avg(L["share2_won"]), avg(L["share2_lost"])))
    print("  peak chips before the endgame (x fair share, 4+ players): %.2f" % avg(L["peak_mid"]))
    hh = max(1, c["hu_hands"])
    print("  heads-up: %.1f hands/match, fold SB/button %.0f%%, fold BB %.0f%%, all-in %.0f%% of hands, "
          "won without showdown %.0f%%, showdowns %.0f%% (won %.0f%%)" % (
              c["hu_hands"] / max(1.0, c["reach2"]), 100 * c["hu_sb_fold"] / max(1, c["hu_sb"]),
              100 * c["hu_bb_fold"] / max(1, c["hu_bb"]), 100 * c["hu_allin"] / hh, 100 * c["hu_won_nosd"] / hh,
              100 * c["hu_sd"] / hh, 100 * c["hu_sd_won"] / max(1, c["hu_sd"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--teams", default="Team 72,TeamBallHog,AZ,2 Curry,WAYS-ER TEAM")
    a = ap.parse_args()
    teams = [x for x in a.teams.split(",") if x]
    S, L = analyse(a.path, set(teams))
    pr("US", S["US"], L["US"])
    for t in teams:
        if t in S:
            pr(t, S[t], L[t])


if __name__ == "__main__":
    main()
