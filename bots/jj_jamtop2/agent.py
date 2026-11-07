"""Jam-top family, Bay Area variant: push/fold with the top teams' shape.

Preflop is all-in or fold. The push range is positional like the top teams'
opening ranges (tight early, wide on the button / small blind) and shrinks
with stack depth: range = open_<pos> * min(1, (ref_bb / eff_bb) ** depth_exp),
with a floor `deep_pct` of premium hands that are always pushed. Facing a raise
or an all-in, call/re-shove with the top `call_pct` scaled the same way.
Flop selection uses the top teams' rule (keep the best made pair, never expose
a board-pairing card, otherwise expose the lower card). Postflop (only after a
check from the big blind or a called shove that is not all-in): shove made
hands >= post_cat, otherwise check/fold.
"""

import bisect
from itertools import combinations

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action, GameState

RANKS = "23456789TJQKA"
DECK = [r + s for r in RANKS for s in "cdhs"]
CARD = {c: eval7.Card(c) for c in DECK}
EVAL = eval7.evaluate
PAIRS3 = ((0, 1), (0, 2), (1, 2))

DEFAULTS = {"open_ep": 0.24, "open_mp": 0.32, "open_co": 0.45, "open_btn": 0.72, "open_sb": 0.78,
            "ref_bb": 10.0, "depth_exp": 1.0, "deep_pct": 0.03, "call_pct": 0.25, "call_exp": 1.0,
            "call_floor": 0.02, "post_cat": 1.4}


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


def passive(state):
    return Action.check() if state.can("check") else Action.fold()


def shove(state):
    r = state.can("raise")
    if r:
        return Action.raise_to(r.max_amount)
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
    reveal = (b if pa else a) if pa != pb else (a if _r(cs[a]) <= _r(cs[b]) else b)
    return Action("select", cards=(3 - a - b, reveal))


FIX = {"hu_k": 14.0, "hu_open": 0.85, "hu_call_k": 7.0, "bb_reshove": 0.30, "bb_call": 0.62,
       "steal_bb": 15.0, "steal_size": 2.3, "cbet": 0.75, "cbet_size": 0.55, "three_mult": 1.35}


def raise_amt(state, amt):
    r = state.can("raise")
    if not r:
        return Action.call() if state.can("call") else Action.check()
    return Action.raise_to(int(max(r.min_amount, min(r.max_amount, amt))))


def bet(state, P):
    """Fixes (endgame analysis of round 5): heads-up / 3-handed endgame mode and
    small steals + c-bets when deep, instead of all-in-or-fold everywhere."""
    me = next((s for s in state.seats if s.seat == state.seat), None)
    live = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
    if me is None or not live:
        return passive(state)
    F = FIX
    bb = max(1, state.blinds.big_blind)
    eff_bb = max(1.0, min(me.stack + me.bet, max(s.stack + s.bet for s in live)) / float(bb))
    alive = sum(1 for s in state.seats if s.status != "busted")
    raises = [e for e in state.history if e.kind == "raise"]
    pf_raises = []
    for e in state.history:
        if e.kind == "board":
            break
        if e.kind == "raise":
            pf_raises.append(e)
    if state.board:
        m = made(list(state.cards), list(state.board))
        if m >= P["post_cat"]:
            return shove(state)
        street_raises = 0
        for e in reversed(state.history):
            if e.kind == "board":
                break
            if e.kind == "raise":
                street_raises += 1
        if state.to_call == 0 and street_raises == 0 and len(state.board) == 3 and len(live) == 1:
            i_raised = bool(pf_raises) and pf_raises[-1].seat == state.seat
            if (i_raised or alive == 2) and __import__("random").random() < F["cbet"]:
                return raise_amt(state, max(bb, F["cbet_size"] * state.pot))
        return passive(state)
    p = pct(state.cards)
    pos = position_key(state)
    if alive == 2:  # heads-up endgame: the button is the small blind
        if not pf_raises:
            if state.can("check"):  # big blind, limped or option
                return shove(state) if p <= F["bb_reshove"] else Action.check()
            if p <= min(1.0, F["hu_k"] / eff_bb):
                return shove(state)
            if p <= F["hu_open"]:
                return raise_amt(state, F["steal_size"] * bb)
            return passive(state)
        last = pf_raises[-1]
        shoved = any(s.seat == last.seat and s.status == "all_in" for s in state.seats) or \
            state.to_call >= 0.6 * (me.stack + me.bet)
        if shoved:
            return shove(state) if p <= min(1.0, F["hu_call_k"] / eff_bb) else passive(state)
        if p <= F["bb_reshove"]:
            return shove(state)
        if p <= F["bb_call"]:
            return Action.call() if state.can("call") else Action.check()
        return passive(state)
    if not pf_raises and eff_bb > F["steal_bb"] and pos in ("co", "btn", "sb"):
        w = P["open_" + pos] * (1.3 if alive == 3 else 1.0)
        if p <= min(1.0, w):
            return raise_amt(state, F["steal_size"] * bb)
        return passive(state)
    if len(pf_raises) == 1 and pf_raises[0].seat == state.seat:  # our steal got re-raised
        rng = max(P["call_floor"], P["call_pct"] * min(1.0, (P["ref_bb"] / eff_bb) ** P["call_exp"]))
        return shove(state) if p <= rng else passive(state)
    if alive == 3 and not pf_raises and pos in ("btn", "sb"):
        P = dict(P, **{"open_" + pos: min(1.0, P["open_" + pos] * 1.3)})
    return bet_jam(state, P)


def bet_jam(state, P):
    me = next((s for s in state.seats if s.seat == state.seat), None)
    live = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
    if me is None or not live:
        return passive(state)
    if state.board:
        if made(list(state.cards), list(state.board)) >= P["post_cat"]:
            return shove(state)
        return passive(state)
    bb = max(1, state.blinds.big_blind)
    eff_bb = max(1.0, min(me.stack + me.bet, max(s.stack + s.bet for s in live)) / float(bb))
    p = pct(state.cards)
    raised = any(e.kind == "raise" for e in state.history)
    if not raised:
        pos = position_key(state)
        if pos == "bb":
            return Action.check() if state.can("check") else passive(state)
        rng = max(P["deep_pct"], P["open_" + pos] * min(1.0, (P["ref_bb"] / eff_bb) ** P["depth_exp"]))
        return shove(state) if p <= rng else passive(state)
    rng = max(P["call_floor"], P["call_pct"] * min(1.0, (P["ref_bb"] / eff_bb) ** P["call_exp"]))
    return shove(state) if p <= rng else passive(state)


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


# tuned vs top-team replicas (tools/tunejam.py)
Agent.P = dict(Agent.P, **{"open_ep": 0.0731, "open_mp": 0.2884, "open_co": 0.4295, "open_btn": 0.703, "open_sb": 0.7944, "ref_bb": 17.4335, "depth_exp": 1.4059, "deep_pct": 0.0224, "call_pct": 0.2985, "call_exp": 1.2165, "call_floor": 0.0208, "post_cat": 1.4863})
