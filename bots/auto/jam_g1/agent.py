"""All-in (jam) family, Bay Area variant.

Preflop push/fold only: push the top hands by a closed-form three-card
strength, with the threshold scaled by stack depth ((ref_bb / eff_bb) ** exp);
call all-ins above `call_th`. Flop selection keeps the best made pair.
Postflop (only after a call or a limp): shove made hands at or above
`post_cat`, otherwise check/fold.
"""

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action, GameState

RANKS = "23456789TJQKA"
CARD = {r + s: eval7.Card(r + s) for r in RANKS for s in "cdhs"}
EVAL = eval7.evaluate
PAIRS3 = ((0, 1), (0, 2), (1, 2))
DEFAULTS = {"push_th": 0.7, "call_th": 0.78, "ref_bb": 15.0, "stack_exp": 0.5, "post_cat": 2.0}


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


def strength(cards):
    cs = list(cards)
    if len(cs) == 2:
        return min(1.0, pair_score(*cs) / 52.0)
    sc = sorted((pair_score(cs[a], cs[b]) for a, b in PAIRS3), reverse=True)
    return min(1.0, (sc[0] + 0.25 * sc[1]) / 62.0)


def passive(state):
    return Action.check() if state.can("check") else Action.fold()


def bet(state, P):
    me = next((s for s in state.seats if s.seat == state.seat), None)
    live = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
    if me is None or not live:
        return passive(state)
    r = state.can("raise")
    if state.board:
        cat = EVAL([CARD[c] for c in state.cards] + [CARD[c] for c in state.board]) >> 24
        if cat >= P["post_cat"]:
            if r:
                return Action.raise_to(r.max_amount)
            return Action.call() if state.can("call") else Action.check()
        return passive(state)
    bb = max(1, state.blinds.big_blind)
    eff_bb = min(me.stack + me.bet, max(s.stack + s.bet for s in live)) / float(bb)
    scale = (P["ref_bb"] / max(1.0, eff_bb)) ** P["stack_exp"]
    st = strength(state.cards)
    raised = any(e.kind == "raise" for e in state.history)
    th = (P["call_th"] if raised else P["push_th"])
    if st >= 1.0 - (1.0 - th) * scale:
        if r:
            return Action.raise_to(r.max_amount)
        return Action.call() if state.can("call") else Action.check()
    return passive(state)


def select(state):
    cs = list(state.cards)
    if len(cs) != 3:
        return Action("select", cards=(2, 0))
    bcs = [CARD[c] for c in state.board]
    best = max(range(3), key=lambda i: (EVAL([CARD[cs[PAIRS3[i][0]]], CARD[cs[PAIRS3[i][1]]]] + bcs),
                                        pair_score(cs[PAIRS3[i][0]], cs[PAIRS3[i][1]])))
    a, b = PAIRS3[best]
    return Action("select", cards=(3 - a - b, a if _r(cs[a]) <= _r(cs[b]) else b))


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


# evolved parameters: jam_g1 (main)
Agent.P = dict(Agent.P, **{"push_th": 0.7594, "call_th": 0.8376, "ref_bb": 22.2282, "stack_exp": 0.6085, "post_cat": 1.0})
