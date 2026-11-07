"""Reverse-engineer opponents' strategies from downloaded round hand histories.

  python3 tools/profile.py --top 8 results/hands/practice1.jsonl.gz results/hands/practice2.jsonl.gz ...

For each of the top teams (by points over all given rounds), replays every
hand they played at our tables, tracking the pot, street, position and stack
depth, and reports:
  preflop:  open-raise rate by position, open size (BB), limp rate, 3-bet rate
            and size, preflop all-in rate by stack depth, call-vs-open rate
  ranges:   retained two cards at showdown by preflop action (pair rate,
            average high card, suited/connected), i.e. what they open/3-bet/jam
  flop:     exposure policy (expose the higher or lower retained card, a card
            pairing the board, from a pocket pair)
  postflop: bet/raise frequency and sizing (pot fraction) by street, c-bet
            rate, fold-to-bet, check-raise rate; at showdown, the made-hand
            category (with eval7 when available) at each bet/call/check
Output: printed summary + JSON (--json).
"""
import argparse
import collections
import gzip
import json

RANKS = "23456789TJQKA"
try:
    import eval7
    _C = {r + s: eval7.Card(r + s) for r in RANKS for s in "cdhs"}

    def category(cards, board):
        if len(cards) + len(board) < 5:
            return None
        return eval7.evaluate([_C[c] for c in list(cards) + list(board)]) >> 24
except Exception:  # pragma: no cover
    def category(cards, board):
        return None

CAT = ["high", "pair", "2pair", "trips", "straight", "flush", "full", "quads", "sf"]


def rk(c):
    return RANKS.index(c[0])


def stream(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        while True:
            a = f.readline()
            if not a:
                break
            f.readline()
            yield header, json.loads(a)


def team_points(paths):
    pts = collections.defaultdict(lambda: [0.0, 0])
    names = {}
    for p in paths:
        for header, t in stream(p):
            if not names:
                for e in header["entries"]:
                    names[e["entry_id"]] = e["team_name"]
            for e in header["entries"]:
                names[e["entry_id"]] = e["team_name"]
            hands = t["hands"]
            if not hands:
                continue
            for s, ent in enumerate(t["seats"]):
                if not t["fillers"][s]:
                    pts[names.get(ent, ent)][0] += hands[-1]["equity_after"][s]
                    pts[names.get(ent, ent)][1] += 1
    return {k: v[0] / v[1] for k, v in pts.items() if v[1] >= 50}, names


def new_profile():
    return {"hands": 0, "open_opp": collections.Counter(), "open": collections.Counter(), "open_size": [],
            "limp": 0, "unopened": 0, "faced_open": 0, "call_open": 0, "threebet": 0, "threebet_mult": [],
            "fold_to_open": 0, "pf_allin": collections.Counter(), "pf_hands_by_depth": collections.Counter(),
            "range": collections.defaultdict(list), "expose": collections.Counter(),
            "post_dec": collections.Counter(), "post_bet": collections.Counter(), "bet_frac": collections.defaultdict(list),
            "cbet_opp": 0, "cbet": 0, "faced_bet": collections.Counter(), "fold_bet": collections.Counter(),
            "call_bet": collections.Counter(), "raise_bet": collections.Counter(), "xr": 0, "xr_opp": 0,
            "sd_actions": collections.defaultdict(collections.Counter), "showdowns": 0}


def pos_label(seat, button, live):
    """BTN, SB, BB, EP, MP, CO by distance from the button among live seats."""
    n = len(live)
    if seat not in live or button not in live:
        return "?"
    d = (live.index(seat) - live.index(button)) % n
    if d == 0:
        return "BTN"
    if d == 1:
        return "SB"
    if d == 2:
        return "BB"
    left = n - 3  # seats between BB and BTN
    k = d - 3
    if left <= 0:
        return "?"
    if k == left - 1:
        return "CO"
    return "EP" if k < (left - 1) / 2.0 else "MP"


def profile(paths, top_n):
    pts, names = team_points(paths)
    top = [t for t, _ in sorted(pts.items(), key=lambda kv: -kv[1])[:top_n]]
    P = {t: new_profile() for t in top}
    for p in paths:
        for header, t in stream(p):
            nm = {e["entry_id"]: e["team_name"] for e in header["entries"]}
            seats = [nm.get(e, e) for e in t["seats"]]
            watch = {s: P[n] for s, n in enumerate(seats) if n in P and not t["fillers"][s]}
            if not watch:
                continue
            for hand in t["hands"]:
                analyse_hand(hand, watch)
    return top, pts, P


def analyse_hand(hand, watch):
    bb = hand["blinds"][1] or 1
    stacks0 = hand["stacks_before"]
    live = [s for s, x in enumerate(stacks0) if x > 0]
    shown = {}
    for e in hand["events"]:
        if e.get("kind") == "show" and e.get("seat") is not None:
            shown[e["seat"]] = e.get("cards", [])
    for s in watch:
        if s in live:
            watch[s]["hands"] += 1
    street = 0
    board = []
    pot = 0
    contrib = collections.Counter()   # this street
    cur = 0
    raises_pf = 0
    first_raiser = None
    pf_action = {}
    aggressor_pf = None
    checked = set()
    bet_this_street = False
    exposed = {}
    sd_log = collections.defaultdict(list)
    for e in hand["events"]:
        k = e.get("kind")
        s = e.get("seat")
        if k in ("blind", "ante"):
            amt = e.get("amount", 0)
            contrib[s] += amt
            pot += amt
            cur = max(cur, contrib[s]) if k == "blind" else cur
            continue
        if k == "board":
            street += 1
            board += e.get("cards", [])
            contrib = collections.Counter()
            cur = 0
            checked = set()
            bet_this_street = False
            continue
        if k == "reveal" and s is not None:
            exposed[s] = (e.get("cards") or [None])[0]
            continue
        if k not in ("fold", "check", "call", "raise") or s is None:
            continue
        pr = watch.get(s)
        to_call = max(0, cur - contrib[s])
        if street == 0:
            if pr is not None:
                pos = pos_label(s, hand["button"], live)
                depth = stacks0[s] / float(bb)
                dk = "<10" if depth < 10 else ("10-25" if depth < 25 else ("25-60" if depth < 60 else "60+"))
                if raises_pf == 0:
                    pr["unopened"] += 1
                    pr["open_opp"][pos] += 1
                    if k == "raise":
                        pr["open"][pos] += 1
                        pr["open_size"].append(e["amount"] / float(bb))
                        pf_action.setdefault(s, "open")
                    elif k == "call":
                        pr["limp"] += 1
                        pf_action.setdefault(s, "limp")
                elif raises_pf == 1 and first_raiser != s:
                    pr["faced_open"] += 1
                    if k == "raise":
                        pr["threebet"] += 1
                        pr["threebet_mult"].append(e["amount"] / float(max(1, cur)))
                        pf_action[s] = "3bet"
                    elif k == "call":
                        pr["call_open"] += 1
                        pf_action.setdefault(s, "call")
                    elif k == "fold":
                        pr["fold_to_open"] += 1
                pr["pf_hands_by_depth"][dk] += 1
                if k == "raise" and e["amount"] >= stacks0[s] * 0.95:
                    pr["pf_allin"][dk] += 1
                    pf_action[s] = "jam"
        else:
            if pr is not None:
                st = ("flop", "turn", "river")[min(2, street - 1)]
                pr["post_dec"][st] += 1
                if to_call > 0:
                    pr["faced_bet"][st] += 1
                    if k == "fold":
                        pr["fold_bet"][st] += 1
                    elif k == "call":
                        pr["call_bet"][st] += 1
                    elif k == "raise":
                        pr["raise_bet"][st] += 1
                        if s in checked:
                            pr["xr"] += 1
                    if s in checked:
                        pr["xr_opp"] += 1
                if k == "raise":
                    pr["post_bet"][st] += 1
                    pr["bet_frac"][st].append((e["amount"] - cur) / float(max(1, pot)))
                if street == 1 and not bet_this_street and aggressor_pf == s and to_call == 0:
                    pr["cbet_opp"] += 1
                    if k == "raise":
                        pr["cbet"] += 1
                sd_log[s].append((st, k, list(board), (e["amount"] - cur) / float(max(1, pot)) if k == "raise" else 0.0))
        if k == "check":
            checked.add(s)
        if k == "call":
            amt = e.get("amount", 0)
            contrib[s] += amt
            pot += amt
        elif k == "raise":
            add = e["amount"] - contrib[s]
            contrib[s] = e["amount"]
            pot += max(0, add)
            cur = max(cur, e["amount"])
            if street == 0:
                raises_pf += 1
                if first_raiser is None:
                    first_raiser = s
                aggressor_pf = s
            else:
                bet_this_street = True
    # showdown: ranges, exposure, made-hand at each postflop decision
    for s, cards in shown.items():
        pr = watch.get(s)
        if pr is None or len(cards) != 2:
            continue
        pr["showdowns"] += 1
        a, b = cards
        hi, lo = max(rk(a), rk(b)), min(rk(a), rk(b))
        pr["range"][pf_action.get(s, "check/blind")].append((hi, lo, rk(a) == rk(b), a[1] == b[1]))
        ex = exposed.get(s)
        if ex in cards:
            other = b if ex == a else a
            brk = {c[0] for c in board[:3]}
            if rk(a) == rk(b):
                pr["expose"]["pocket pair"] += 1
            elif ex[0] in brk and other[0] not in brk:
                pr["expose"]["exposed the board-pairing card"] += 1
            elif other[0] in brk and ex[0] not in brk:
                pr["expose"]["hid the board-pairing card"] += 1
            elif rk(ex) > rk(other):
                pr["expose"]["exposed higher"] += 1
            else:
                pr["expose"]["exposed lower"] += 1
        for st, k, bd, frac in sd_log.get(s, []):
            c = category(cards, bd)
            if c is not None:
                pr["sd_actions"]["%s %s" % (st, CAT[c])][k] += 1


def summarize(top, pts, P):
    out = {}
    for t in top:
        p = P[t]
        h = max(1, p["hands"])
        o = {"points": round(pts[t], 2), "hands": p["hands"]}
        o["open_by_pos"] = {pos: round(p["open"][pos] / float(max(1, p["open_opp"][pos])), 2)
                            for pos in ("EP", "MP", "CO", "BTN", "SB") if p["open_opp"][pos] >= 20}
        os_ = sorted(p["open_size"])
        o["open_size_bb_median"] = round(os_[len(os_) // 2], 2) if os_ else None
        o["limp_rate"] = round(p["limp"] / float(max(1, p["unopened"])), 3)
        o["vs_open"] = {"call": round(p["call_open"] / float(max(1, p["faced_open"])), 3),
                        "3bet": round(p["threebet"] / float(max(1, p["faced_open"])), 3),
                        "fold": round(p["fold_to_open"] / float(max(1, p["faced_open"])), 3)}
        tm = sorted(p["threebet_mult"])
        o["3bet_mult_median"] = round(tm[len(tm) // 2], 2) if tm else None
        o["pf_allin_by_depth"] = {d: round(p["pf_allin"][d] / float(max(1, p["pf_hands_by_depth"][d])), 3)
                                  for d in ("<10", "10-25", "25-60", "60+") if p["pf_hands_by_depth"][d] >= 20}
        rng = {}
        for act, lst in p["range"].items():
            if len(lst) >= 8:
                rng[act] = {"n": len(lst), "pair%": round(100 * sum(1 for x in lst if x[2]) / float(len(lst))),
                            "avg_hi": RANKS[int(round(sum(x[0] for x in lst) / float(len(lst))))],
                            "avg_lo": RANKS[int(round(sum(x[1] for x in lst) / float(len(lst))))],
                            "suited%": round(100 * sum(1 for x in lst if x[3]) / float(len(lst)))}
        o["showdown_ranges"] = rng
        tot = sum(p["expose"].values())
        o["exposure"] = {k: round(v / float(tot), 2) for k, v in p["expose"].most_common()} if tot >= 8 else {}
        o["postflop"] = {}
        for st in ("flop", "turn", "river"):
            d = p["post_dec"][st]
            if d < 20:
                continue
            fr = sorted(p["bet_frac"][st])
            o["postflop"][st] = {"bet_or_raise": round(p["post_bet"][st] / float(d), 2),
                                 "size_pot_median": round(fr[len(fr) // 2], 2) if fr else None,
                                 "fold_to_bet": round(p["fold_bet"][st] / float(max(1, p["faced_bet"][st])), 2),
                                 "raise_vs_bet": round(p["raise_bet"][st] / float(max(1, p["faced_bet"][st])), 2)}
        o["cbet"] = round(p["cbet"] / float(max(1, p["cbet_opp"])), 2) if p["cbet_opp"] >= 10 else None
        o["check_raise"] = round(p["xr"] / float(max(1, p["xr_opp"])), 2) if p["xr_opp"] >= 10 else None
        sd = {}
        for key, c in sorted(p["sd_actions"].items()):
            n = sum(c.values())
            if n >= 10:
                sd[key] = {"n": n, "bet%": round(100 * c["raise"] / float(n)), "call%": round(100 * c["call"] / float(n)),
                           "check%": round(100 * c["check"] / float(n))}
        o["showdown_strength_actions"] = sd
        out[t] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    top, pts, P = profile(a.paths, a.top)
    out = summarize(top, pts, P)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    for t, o in out.items():
        print("\n=== %s  (%.2f pts/table at our tables, %d hands seen)" % (t, o["points"], o["hands"]))
        print("  open by position:", o["open_by_pos"], "| open size %sBB | limp %s" % (o["open_size_bb_median"], o["limp_rate"]))
        print("  vs an open:", o["vs_open"], "| 3bet size x%s" % o["3bet_mult_median"])
        print("  preflop all-in rate by depth (BB):", o["pf_allin_by_depth"])
        print("  showdown ranges by preflop action:", o["showdown_ranges"])
        print("  exposure:", o["exposure"])
        print("  postflop:", o["postflop"], "| cbet", o["cbet"], "| check-raise", o["check_raise"])
        sd = o["showdown_strength_actions"]
        if sd:
            print("  at showdown, action by made hand:", {k: v for k, v in list(sd.items())[:14]})


if __name__ == "__main__":
    main()
