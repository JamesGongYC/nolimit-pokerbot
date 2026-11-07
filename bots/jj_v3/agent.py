"""Rule-based family (no Monte Carlo), Bay Area variant.

Preflop: a closed-form strength of the best two of three cards (pairs, high
cards, suitedness, connectedness, plus a small bonus for the third card's
options), compared with position-dependent thresholds; push/fold when short.
Flop selection: keep the pair whose made hand with the flop is best, expose the
lower card. Postflop: eval7 hand category with the board, discounted when the
board alone already makes it, against bet / call / raise thresholds.
"""

import random

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action, GameState

RANKS = "23456789TJQKA"
CARD = {r + s: eval7.Card(r + s) for r in RANKS for s in "cdhs"}
EVAL = eval7.evaluate
PAIRS3 = ((0, 1), (0, 2), (1, 2))

DEFAULTS = {
    "open_th": 0.62, "pos_bonus": 0.12, "open_size": 2.5, "call_th": 0.72, "threebet_th": 0.86,
    "push_bb": 12.0, "push_th": 0.66, "call_push_th": 0.74,
    "bet_cat": 1.6, "call_cat": 1.2, "raise_cat": 2.6, "big_bet_extra": 0.6,
    "cbet_rate": 0.5, "bluff_rate": 0.1, "value_size": 0.66, "bluff_size": 0.5,
}
_rng = random.Random()


def _r(c):
    return RANKS.index(c[0])


def pair_score(a, b):
    ra, rb = _r(a), _r(b)
    hi, lo = max(ra, rb), min(ra, rb)
    s = 2 * hi + lo
    if ra == rb:
        s += 14
    if a[1] == b[1]:
        s += 3
    gap = hi - lo
    if 0 < gap <= 2:
        s += 3 - gap
    return s


def strength(cards):
    """0..1 preflop strength of 2-3 cards (best pair + third-card options)."""
    cs = list(cards)
    if len(cs) == 2:
        return min(1.0, pair_score(*cs) / 52.0)
    scores = sorted((pair_score(cs[a], cs[b]) for a, b in PAIRS3), reverse=True)
    return min(1.0, (scores[0] + 0.25 * scores[1]) / 62.0)


def made(cards, board):
    """Hand category (0 high .. 8 straight flush) with the board, minus how much
    the board alone already gives everyone; fractional bonus for top pair."""
    full = EVAL([CARD[c] for c in cards] + [CARD[c] for c in board])
    cat = full >> 24
    bcat = (EVAL([CARD[c] for c in board]) >> 24) if len(board) >= 5 else 0
    v = float(cat)
    if cat <= bcat:
        v -= 0.8
    if cat == 1:
        top = max(_r(c) for c in board)
        if any(_r(c) == top for c in cards) or (len(cards) == 2 and _r(cards[0]) == _r(cards[1]) > top):
            v += 0.4
    return v


def position(state):
    live = [s.seat for s in state.seats if s.status != "busted"]
    if state.button not in live or state.seat not in live:
        return 0.5
    n = len(live)
    bi = live.index(state.button)
    order = [live[(bi + 1 + k) % n] for k in range(n)]
    return order.index(state.seat) / float(max(1, n - 1))


def raise_to(state, amount, my_total):
    r = state.can("raise")
    if not r:
        return Action.call() if state.can("call") else Action.check()
    amt = int(max(r.min_amount, min(r.max_amount, amount)))
    if amt >= 0.6 * my_total:
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
    best = max(range(3), key=lambda i: (made([cs[PAIRS3[i][0]], cs[PAIRS3[i][1]]], list(state.board)),
                                        pair_score(cs[PAIRS3[i][0]], cs[PAIRS3[i][1]])))
    a, b = PAIRS3[best]
    reveal = a if _r(cs[a]) <= _r(cs[b]) else b
    return Action("select", cards=(3 - a - b, reveal))


def bet(state, P):
    me = next((s for s in state.seats if s.seat == state.seat), None)
    if me is None:
        return passive(state)
    bb = max(1, state.blinds.big_blind)
    live = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
    if not live:
        return call_or_check(state)
    my_total = me.stack + me.bet
    eff_bb = min(my_total, max(s.stack + s.bet for s in live)) / float(bb)
    pot, to_call = state.pot, state.to_call
    cur = max(s.bet for s in state.seats)
    raises = 0
    pf_aggr = None
    street = 0
    for e in state.history:
        if e.kind == "board":
            street += 1
            raises = 0
        elif e.kind == "raise":
            raises += 1
            if street == 0:
                pf_aggr = e.seat
    if not state.board:
        st = strength(state.cards)
        if eff_bb <= P["push_bb"]:
            if raises == 0:
                return raise_to(state, 10 ** 12, my_total) if st >= P["push_th"] - 0.08 * position(state) else passive(state)
            return call_or_check(state) if st >= P["call_push_th"] else passive(state)
        if raises == 0:
            if st >= P["open_th"] - P["pos_bonus"] * position(state):
                return raise_to(state, P["open_size"] * bb, my_total)
            return passive(state)
        if st >= P["threebet_th"]:
            return raise_to(state, 3 * cur, my_total)
        if st >= P["call_th"] and to_call <= 0.25 * me.stack:
            return call_or_check(state)
        return passive(state)
    m = made(list(state.cards), list(state.board))
    n = len(live)
    m -= 0.3 * (n - 1)
    if to_call == 0:
        if m >= P["bet_cat"]:
            return raise_to(state, cur + max(bb, P["value_size"] * pot), my_total)
        if pf_aggr == state.seat and len(state.board) == 3 and n == 1 and _rng.random() < P["cbet_rate"]:
            return raise_to(state, cur + max(bb, P["bluff_size"] * pot), my_total)
        if n == 1 and _rng.random() < P["bluff_rate"]:
            return raise_to(state, cur + max(bb, P["bluff_size"] * pot), my_total)
        return Action.check()
    big = to_call > 0.6 * (pot - to_call)
    if m >= P["raise_cat"]:
        return raise_to(state, 3 * cur, my_total)
    if m >= P["call_cat"] + (P["big_bet_extra"] if big else 0.0):
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


# evolved parameters: rule_g9 (main)
Agent.P = dict(Agent.P, **{"open_th": 0.5783, "pos_bonus": 0.2095, "open_size": 2.58, "call_th": 0.8486, "threebet_th": 0.8739, "push_bb": 11.4134, "push_th": 0.5832, "call_push_th": 0.7496, "bet_cat": 1.5074, "call_cat": 1.2125, "raise_cat": 2.1499, "big_bet_extra": 0.5859, "cbet_rate": 0.364, "bluff_rate": 0.1345, "value_size": 0.6703, "bluff_size": 0.6926})
