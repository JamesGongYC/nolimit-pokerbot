"""Real-hand extraction for (1) backtesting rule changes and (2) training the
transformer Q-model (run with the kit venv: numpy).

  starter-kit/.venv/bin/python tools/tfdata.py out.npz results/hands/scored1.jsonl.gz ...

One row per preflop decision of OUR seat (we know our three dealt cards):
  cards  (3,)    rank*4+suit of the original three cards
  ctx    (C,)    context: eff/my stack (log BB), position one-hot, players
                 alive, raises before us, to-call / stack, pot (BB), facing an
                 all-in, hand fraction, blind level, payout top share
  hist   (H, 4)  prior preflop actions this hand: relative seat, kind
                 (fold/call/raise/all-in), size (log BB), is-blind
  action         0 fold, 1 call/check, 2 raise (not all-in), 3 all-in
  reward         the hand's change in our ICM points (equity_after - equity_before)
Also prints the backtest of jamtop3's two fixes against jamtop_tuned's real
decisions when --backtest is given.
"""
import argparse
import gzip
import json
import math
import os
import sys

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
H = 10
C = 18
POS = ["sb", "bb", "ep", "mp", "co", "btn"]


def cid(c):
    return RANKS.index(c[0]) * 4 + SUITS.index(c[1])


def stream(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        while True:
            a = f.readline()
            if not a:
                break
            b = f.readline()
            yield header, json.loads(a), json.loads(b)["you"]


def pos_of(s, button, live):
    n = len(live)
    if s not in live or button not in live:
        return "btn"
    d = (live.index(s) - live.index(button)) % n
    if d == 0:
        return "btn"
    if d == 1:
        return "sb"
    if d == 2:
        return "bb"
    left, k = n - 3, d - 3
    if k == left - 1:
        return "co"
    return "ep" if k < (left - 1) / 2.0 else "mp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    pct = None
    if a.backtest:
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sp = importlib.util.spec_from_file_location("jt", os.path.join(root, "bots", "jj_jamtop3", "agent.py"))
        jt = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(jt)
        pct = jt.pct
        bt = {"deep_reshove": [], "deep_reshove_busts": 0, "deep_reshove_n": 0,
              "hu_sb_fold": 0, "hu_sb_fold_would_push": 0, "hu_bb_fold_vs_shove": 0, "hu_bb_would_call": 0,
              "hu_bb_fold_pts": []}
    cards_l, ctx_l, hist_l, act_l, rew_l = [], [], [], [], []
    for path in a.paths:
        for header, t, own in stream(path):
            s = own["seat"]
            if t["fillers"][s]:
                continue
            pay = t["payouts"]
            top_share = pay[0] / float(max(1, sum(pay)))
            for hi, hand in enumerate(t["hands"]):
                st = hand["stacks_before"]
                if st[s] <= 0 or hi >= len(own["cards"]):
                    continue
                mine = own["cards"][hi]
                if len(mine) != 3:
                    continue
                bb = float(hand["blinds"][1] or 1)
                live = [k for k, x in enumerate(st) if x > 0]
                n_live = len(live)
                pos = pos_of(s, hand["button"], live)
                others = [st[k] for k in live if k != s]
                eff = min(st[s], max(others)) if others else st[s]
                reward = hand["equity_after"][s] - hand["equity_before"][s]
                bets = {}
                cur = 0.0
                raises = 0
                hist = []
                for e in hand["events"]:
                    k = e.get("kind")
                    if k == "board":
                        break
                    seat = e.get("seat")
                    if k == "blind":
                        bets[seat] = e.get("amount", 0)
                        cur = max(cur, bets[seat])
                        hist.append([((seat - s) % 8) / 7.0, 1.0, math.log1p(e.get("amount", 0) / bb), 1.0])
                        continue
                    if k not in ("fold", "check", "call", "raise"):
                        continue
                    if seat == s:
                        to_call = max(0.0, cur - bets.get(s, 0))
                        allin = k == "raise" and e.get("amount", 0) >= st[s] * 0.95
                        act = 0 if k == "fold" else (1 if k in ("call", "check") else (3 if allin else 2))
                        pot = sum(bets.values())
                        facing_allin = 1.0 if any(h[1] == 3 for h in hist) else 0.0
                        ctx = [math.log1p(eff / bb), math.log1p(st[s] / bb)] + [1.0 if pos == p_ else 0.0 for p_ in POS] + \
                              [n_live / 8.0, raises / 3.0, to_call / max(1.0, st[s]), math.log1p(pot / bb), facing_allin,
                               hand["hand"] / 250.0, math.log(bb / 20.0) / 5.0, top_share, 1.0 if n_live == 2 else 0.0,
                               1.0 if n_live <= 3 else 0.0]
                        hh = hist[-H:]
                        hh = [[0.0, 0.0, 0.0, 0.0]] * (H - len(hh)) + hh
                        cards_l.append([cid(c) for c in mine])
                        ctx_l.append(ctx)
                        hist_l.append(hh)
                        act_l.append(act)
                        rew_l.append(reward)
                        if a.backtest:
                            p = pct(mine)
                            ebb = eff / bb
                            if n_live > 2 and raises >= 1 and act == 3 and ebb > 20 and p > 0.04:
                                bt["deep_reshove"].append(reward)
                                bt["deep_reshove_n"] += 1
                                bt["deep_reshove_busts"] += hand["stacks_after"][s] == 0
                            if n_live == 2 and raises == 0 and pos in ("btn", "sb") and act == 0:
                                bt["hu_sb_fold"] += 1
                                bt["hu_sb_fold_would_push"] += p <= min(1.0, 12.0 / ebb)
                            if n_live == 2 and facing_allin and act == 0:
                                bt["hu_bb_fold_vs_shove"] += 1
                                bt["hu_bb_would_call"] += p <= min(1.0, 6.0 / ebb)
                        kind_code = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}[act]
                        hist.append([0.0, kind_code, math.log1p(e.get("amount", 0) / bb), 0.0])
                    else:
                        kc = {"fold": 0.0, "check": 1.0, "call": 1.0, "raise": 2.0}[k]
                        if k == "raise" and e.get("amount", 0) >= st[seat] * 0.95:
                            kc = 3.0
                        hist.append([((seat - s) % 8) / 7.0, kc, math.log1p(e.get("amount", 0) / bb), 0.0])
                    if k == "raise":
                        raises += 1
                        bets[seat] = e.get("amount", 0)
                        cur = max(cur, bets[seat])
                    elif k == "call":
                        bets[seat] = bets.get(seat, 0) + e.get("amount", 0)
    np.savez_compressed(a.out, cards=np.array(cards_l, np.int16), ctx=np.array(ctx_l, np.float32),
                        hist=np.array(hist_l, np.float32), action=np.array(act_l, np.int8),
                        reward=np.array(rew_l, np.float32))
    acts = np.bincount(np.array(act_l, np.int64), minlength=4)
    print("rows %d, actions fold/call/raise/allin = %s, mean reward by action: %s" % (
        len(act_l), acts.tolist(), [round(float(np.mean([r for r, x in zip(rew_l, act_l) if x == k])), 3) if acts[k] else None
                                    for k in range(4)]))
    if a.backtest:
        d = bt["deep_reshove"]
        print("\nBACKTEST fix 1 (deep re-shove cap, >3 players, >20 BB, hand outside top 4%):")
        print("  jamtop_tuned made %d such re-shoves; they busted us %d times; mean realised %+.3f ICM points per "
              "decision (folding instead loses at most the blind already posted)" % (
                  bt["deep_reshove_n"], bt["deep_reshove_busts"], (sum(d) / len(d)) if d else float("nan")))
        print("  total realised from these decisions: %+.1f points over the round" % sum(d))
        print("BACKTEST fix 2 (heads-up): SB/button folds %d, of which jamtop3 would push %d; "
              "BB folds vs an all-in %d, of which jamtop3 would call %d (counterfactual results not observable)" % (
                  bt["hu_sb_fold"], bt["hu_sb_fold_would_push"], bt["hu_bb_fold_vs_shove"], bt["hu_bb_would_call"]))


if __name__ == "__main__":
    main()
