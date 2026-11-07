"""Top-clone family, Bay Area variant: the playbook the top teams share
(reverse-engineered from practice-round hand histories, tools/profile.py).

Preflop: raise-or-fold, position-dependent opening ranges (top X% of three-card
hands by a precomputed percentile), 2.5 BB opens, value-heavy 3-bets, mostly
fold to opens, push/fold when short with depth-scaled ranges.
Flop selection: keep the best made pair; never expose a card that pairs the
board; otherwise expose the LOWER retained card.
Postflop: c-bet most flops at ~0.6 pot, value-bet top pair or better, bluff
turn/river rarely, fold to bets without a real hand (fold-to-bet ~55-60%),
raise only with strong made hands.
"""

import bisect
import random
from itertools import combinations

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action, GameState

RANKS = "23456789TJQKA"
DECK = [r + s for r in RANKS for s in "cdhs"]
CARD = {c: eval7.Card(c) for c in DECK}
EVAL = eval7.evaluate
PAIRS3 = ((0, 1), (0, 2), (1, 2))

DEFAULTS = {
    "open_ep": 0.24, "open_mp": 0.32, "open_co": 0.42, "open_btn": 0.65, "open_sb": 0.70,
    "open_size": 2.5, "threebet_pct": 0.04, "threebet_mult": 3.2, "call_open_pct": 0.13,
    "call_max_frac": 0.15, "fourbet_pct": 0.02, "call_3bet_pct": 0.045,
    "push_bb": 12.0, "push_k": 3.0, "call_push_k": 1.8,
    "cbet": 0.82, "cbet_size": 0.6, "value_cat": 1.4, "value_size": 0.66,
    "bluff_turn": 0.2, "bluff_river": 0.1, "call_cat": 1.0, "big_call_cat": 1.4, "big_bet": 0.75,
    "raise_cat": 3.0, "multiway_pen": 0.3,
}
_rng = random.Random()


def _r(c):
    return RANKS.index(c[0])


def pair_score(a, b):
    ra, rb = _r(a), _r(b)
    hi, lo = max(ra, rb), min(ra, rb)
    s = 2 * hi + lo + (14 if ra == rb else 0) + (3 if a[1] == b[1] else 0)
    gap = hi - lo
    if 0 < gap <= 2:
        s += 3 - gap
    return s


def score3(cs):
    sc = sorted((pair_score(cs[a], cs[b]) for a, b in PAIRS3), reverse=True)
    return sc[0] + 0.25 * sc[1]


# percentile table over all 22,100 three-card hands (0 = best)
_ALL = sorted(score3(h) for h in combinations(DECK, 3))
_N = float(len(_ALL))


def pct(cards):
    cs = list(cards)
    s = score3(cs) if len(cs) == 3 else pair_score(*cs) * 1.25
    return 1.0 - bisect.bisect_right(_ALL, s) / _N


def made(cards, board):
    full = EVAL([CARD[c] for c in cards] + [CARD[c] for c in board])
    cat = full >> 24
    bcat = (EVAL([CARD[c] for c in board]) >> 24) if len(board) >= 5 else 0
    v = float(cat)
    if cat <= bcat:
        v -= 0.8
    if cat == 1:
        top = max(_r(c) for c in board)
        if any(_r(c) == top for c in cards) or (_r(cards[0]) == _r(cards[1]) > top):
            v += 0.4
    return v


def position_key(state):
    live = [s.seat for s in state.seats if s.status != "busted"]
    n = len(live)
    if state.seat not in live or state.button not in live or n < 2:
        return "btn"
    d = (live.index(state.seat) - live.index(state.button)) % n
    if d == 0:
        return "btn"
    if d == 1:
        return "sb"
    if d == 2:
        return "bb"
    left = n - 3
    k = d - 3
    if k == left - 1:
        return "co"
    return "ep" if k < (left - 1) / 2.0 else "mp"


def raise_to(state, amount, my_total):
    r = state.can("raise")
    if not r:
        return Action.call() if state.can("call") else Action.check()
    amt = int(max(r.min_amount, min(r.max_amount, amount)))
    if amt >= 0.55 * my_total:
        amt = r.max_amount
    return Action.raise_to(amt)


def passive(state):
    return Action.check() if state.can("check") else Action.fold()


def call_or_check(state):
    return Action.call() if state.can("call") else Action.check()


def select(state):
    cs = list(state.cards)
    if len(cs) != 3:
        return Action("select", cards=(2, 0))
    board = list(state.board)
    best = max(range(3), key=lambda i: (made([cs[PAIRS3[i][0]], cs[PAIRS3[i][1]]], board),
                                        pair_score(cs[PAIRS3[i][0]], cs[PAIRS3[i][1]])))
    a, b = PAIRS3[best]
    brk = {c[0] for c in board}
    pa, pb = cs[a][0] in brk, cs[b][0] in brk
    if pa != pb:                       # never expose the board-pairing card
        reveal = b if pa else a
    else:                              # expose the lower card
        reveal = a if _r(cs[a]) <= _r(cs[b]) else b
    return Action("select", cards=(3 - a - b, reveal))


def bet(state, P):
    me = next((s for s in state.seats if s.seat == state.seat), None)
    if me is None:
        return passive(state)
    live = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
    if not live:
        return call_or_check(state)
    bb = max(1, state.blinds.big_blind)
    my_total = me.stack + me.bet
    eff_bb = min(my_total, max(s.stack + s.bet for s in live)) / float(bb)
    pot, to_call = state.pot, state.to_call
    cur = max(s.bet for s in state.seats)
    street, raises, pf_aggr = 0, 0, None
    for e in state.history:
        if e.kind == "board":
            street += 1
            raises = 0
        elif e.kind == "raise":
            raises += 1
            if street == 0:
                pf_aggr = e.seat
    if not state.board:
        p = pct(state.cards)
        if eff_bb <= P["push_bb"]:
            if raises == 0:
                return raise_to(state, 10 ** 12, my_total) if p <= min(1.0, P["push_k"] / eff_bb) else passive(state)
            return call_or_check(state) if p <= min(1.0, P["call_push_k"] / eff_bb) else passive(state)
        if raises == 0:
            pos = position_key(state)
            if pos == "bb":
                return Action.check() if state.can("check") else passive(state)
            if p <= P["open_" + pos]:
                return raise_to(state, P["open_size"] * bb, my_total)
            return passive(state)
        if raises == 1:
            if p <= P["threebet_pct"]:
                return raise_to(state, P["threebet_mult"] * cur, my_total)
            if p <= P["call_open_pct"] and to_call <= P["call_max_frac"] * my_total:
                return call_or_check(state)
            return passive(state)
        if p <= P["fourbet_pct"]:
            return raise_to(state, 10 ** 12, my_total)
        if p <= P["call_3bet_pct"]:
            return call_or_check(state)
        return passive(state)
    m = made(list(state.cards), list(state.board)) - P["multiway_pen"] * (len(live) - 1)
    if to_call == 0:
        if m >= P["value_cat"]:
            return raise_to(state, cur + max(bb, P["value_size"] * pot), my_total)
        if street == 1 and pf_aggr == state.seat and _rng.random() < P["cbet"]:
            return raise_to(state, cur + max(bb, P["cbet_size"] * pot), my_total)
        bl = P["bluff_turn"] if street == 2 else (P["bluff_river"] if street >= 3 else 0.0)
        if len(live) == 1 and _rng.random() < bl:
            return raise_to(state, cur + max(bb, P["cbet_size"] * pot), my_total)
        return Action.check()
    frac = to_call / float(max(1, pot - to_call))
    if m >= P["raise_cat"]:
        return raise_to(state, 3 * cur, my_total)
    need = P["big_call_cat"] if frac > P["big_bet"] else P["call_cat"]
    if m >= need:
        return call_or_check(state)
    return passive(state)


class Agent(poker_agent_sdk.Agent):
    P = DEFAULTS

    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        try:
            if state.phase == "select":
                return select(state), memory
            if state.phase != "betting":
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), memory
            return bet(state, self.P), memory
        except Exception:
            if state.phase == "select":
                return Action("select", cards=(2, 0)), memory
            return passive(state), memory
