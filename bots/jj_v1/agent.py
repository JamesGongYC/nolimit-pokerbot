"""Team JJ bot, Bay Area variant: three-card hold'em with a sealed flop discard
and one exposed retained card (see starter-kit/RULES.md).

Mechanism: time-bounded Monte Carlo equity with eval7 (C evaluator) against the
live opponents, using every public card (board, exposed cards) and our own
discard as dead cards; a pot-odds / ICM-margin betting policy; the flop
selection keeps the pair with the best Monte Carlo equity (common random
numbers across the three options) and exposes the card that gives away least.
Stateless: everything is recomputed from `state` each turn.
"""

import random
import time

import eval7
import poker_agent_sdk
from poker_agent_sdk import Action, GameState

RANKS = "23456789TJQKA"
CARD = {r + s: eval7.Card(r + s) for r in RANKS for s in "cdhs"}
ALL = tuple(CARD)
EVAL = eval7.evaluate
PAIRS3 = ((0, 1), (0, 2), (1, 2))

# ---- tunable parameters (one place) -------------------------------------
P = {
    "budget_us": 420.0,        # Monte Carlo time budget per betting decision
    "select_budget_us": 560.0,  # flop selection (three options, shared samples)
    "max_samples": 600,
    "open_ratio": 1.18,        # open if equity / fair share >= this (first position)
    "pos_bonus": 0.22,         # ... minus this times position (0 first .. 1 button)
    "threebet_ratio": 1.75,
    "call_margin": 0.04,       # extra equity over pot odds to call
    "push_bb": 12.0,           # push/fold below this effective stack (big blinds)
    "push_ratio": 1.25,
    "value_eq": 0.62,          # postflop: bet for value (heads-up equity scale)
    "raise_eq": 0.80,
    "cbet_rate": 0.55,
    "bluff_rate": 0.12,
    "value_size": 0.66,
    "bluff_size": 0.5,
    "icm_margin": 0.10,        # extra equity per fraction of stack risked
}

_rng = random.Random()


def _r(c):
    return RANKS.index(c[0])


# ---------------------------------------------------------------- equity
def mc_equity(hero_options, board, opps, dead, budget_us, max_n):
    """Share of the pot won by each hero option vs all opponents.

    hero_options: list of card-string lists (2 or 3 cards; 3 = choose best 2).
    opps: list of (known_cards, n_hidden, choose_best2_of_3).
    Common random numbers: every option is scored on the same samples.
    """
    deadline = time.perf_counter() + budget_us * 1e-6
    used = set(board) | set(dead)
    for h in hero_options:
        used |= set(h)
    for known, _, _ in opps:
        used |= set(known)
    deck = [CARD[c] for c in ALL if c not in used]
    bcards = [CARD[c] for c in board]
    need_b = 5 - len(bcards)
    heroes = [[CARD[c] for c in h] for h in hero_options]
    ops = [([CARD[c] for c in k], n, ch) for k, n, ch in opps]
    need = need_b + sum(n for _, n, _ in ops)
    if need > len(deck):
        return [0.5] * len(heroes)
    tot = [0.0] * len(heroes)
    n = 0
    sample = _rng.sample
    while n < max_n:
        if (n & 7) == 0 and n and time.perf_counter() > deadline:
            break
        draw = sample(deck, need)
        b5 = bcards + draw[:need_b]
        i = need_b
        best_opp = -1
        for known, k, choose in ops:
            cards = known + draw[i:i + k]
            i += k
            if choose and len(cards) == 3:
                v = max(EVAL([cards[a], cards[b]] + b5) for a, b in PAIRS3)
            else:
                v = EVAL(cards + b5)
            if v > best_opp:
                best_opp = v
        for j, h in enumerate(heroes):
            if len(h) == 3:
                hv = max(EVAL([h[a], h[b]] + b5) for a, b in PAIRS3)
            else:
                hv = EVAL(h + b5)
            if hv > best_opp:
                tot[j] += 1.0
            elif hv == best_opp:
                tot[j] += 0.5
        n += 1
    if n == 0:
        return [0.5] * len(heroes)
    return [t / n for t in tot]


def opponents(state):
    """(known, n_hidden, choose) per live opponent; also the list of seats."""
    selected = len(state.board) >= 3 and state.phase == "betting" and len(state.cards) == 2
    out, seats = [], []
    for s in state.seats:
        if s.seat == state.seat or s.status not in ("active", "all_in"):
            continue
        known = list(s.shown_cards)
        if selected:
            out.append((known[:2], max(0, 2 - len(known)), False))
        else:
            out.append((known[:3], max(0, 3 - len(known)), True))
        seats.append(s)
    return out, seats


# ---------------------------------------------------------------- helpers
def street_raises(state):
    """Raises by others on the current street and whether we raised preflop."""
    street = 0
    raises = 0
    pf_aggr = None
    for e in state.history:
        if e.kind == "board":
            street += 1
            raises = 0
        elif e.kind == "raise":
            raises += 1
            if street == 0:
                pf_aggr = e.seat
    return raises, pf_aggr


def position(state):
    """0 = first to act .. 1 = button (among seats still holding chips)."""
    live = [s.seat for s in state.seats if s.status != "busted"]
    if state.button not in live or state.seat not in live:
        return 0.5
    n = len(live)
    bi = live.index(state.button)
    order = [live[(bi + 1 + k) % n] for k in range(n)]  # SB first .. button last
    return order.index(state.seat) / float(max(1, n - 1))


def raise_to(state, amount):
    r = state.can("raise")
    if not r:
        return Action.call() if state.can("call") else Action.check()
    amt = int(max(r.min_amount, min(r.max_amount, amount)))
    me = state.seats[state.seat] if state.seat < len(state.seats) else None
    if me is not None and amt >= 0.6 * (me.stack + me.bet):
        amt = r.max_amount
    return Action.raise_to(amt)


def passive(state):
    if state.can("check"):
        return Action.check()
    return Action.fold()


def call_or_check(state):
    if state.can("call"):
        return Action.call()
    return Action.check()


def my_seat(state):
    for s in state.seats:
        if s.seat == state.seat:
            return s
    return None


# ---------------------------------------------------------------- policy
def select(state):
    cards = list(state.cards)
    if len(cards) != 3:
        return Action("select", cards=(2, 0))
    opps, _ = opponents(state)
    opps = [(k, n, True) for k, n, _ in opps]
    options = [[cards[a], cards[b]] for a, b in PAIRS3]  # discard 2, 1, 0 respectively
    eqs = mc_equity(options, list(state.board), opps, [], P["select_budget_us"], P["max_samples"])
    best = max(range(3), key=lambda i: eqs[i])
    a, b = PAIRS3[best]
    discard = 3 - a - b
    board_ranks = {c[0] for c in state.board}
    ka, kb = cards[a], cards[b]
    # expose the card that tells least: not pairing the board, then the lower rank
    def leak(c):
        return (1 if c[0] in board_ranks else 0, 1 if ka[0] == kb[0] else 0, _r(c))
    reveal = a if leak(ka) <= leak(kb) else b
    return Action("select", cards=(discard, reveal))


def bet(state):
    me = my_seat(state)
    if me is None:
        return passive(state)
    bb = max(1, state.blinds.big_blind)
    opps, oseats = opponents(state)
    n = len(opps)
    if n == 0:
        return call_or_check(state)
    to_call = state.to_call
    pot = state.pot
    my_total = me.stack + me.bet
    eff = min(my_total, max(s.stack + s.bet for s in oseats))
    eff_bb = eff / float(bb)
    pot_odds = to_call / float(pot + to_call) if to_call else 0.0
    risk_frac = min(1.0, to_call / float(max(1, me.stack)))
    icm = P["icm_margin"] * risk_frac
    raises, pf_aggr = street_raises(state)
    pre = len(state.board) == 0
    dead = list(state.variant.get("discarded", []) or [])
    if pre:
        k = min(n, 4)
        eq = mc_equity([list(state.cards)], [], opps[:k], dead, P["budget_us"], P["max_samples"])[0]
        if n > k:
            eq = eq ** (n / float(k))
    else:
        eq = mc_equity([list(state.cards)], list(state.board), opps, dead, P["budget_us"], P["max_samples"])[0]
    fair = 1.0 / (n + 1)
    ratio = eq / fair
    cur_bet = max(s.bet for s in state.seats)

    if pre:
        if eff_bb <= P["push_bb"]:
            if raises == 0:
                if ratio >= P["push_ratio"] - 0.15 * position(state):
                    return raise_to(state, 10 ** 12)
                return passive(state)
            if eq >= pot_odds + P["call_margin"] + icm:
                return call_or_check(state) if not state.can("raise") or ratio < 1.6 else raise_to(state, 10 ** 12)
            return passive(state)
        if raises == 0:
            if ratio >= P["open_ratio"] - P["pos_bonus"] * position(state):
                limpers = sum(1 for e in state.history if e.kind == "call")
                return raise_to(state, (2.5 + limpers) * bb)
            return passive(state)
        if ratio >= P["threebet_ratio"]:
            return raise_to(state, 3 * cur_bet)
        if eq >= pot_odds + P["call_margin"] + icm:
            return call_or_check(state)
        return passive(state)

    # postflop: compare to a heads-up scale so multiway pots need more
    hu_eq = eq ** (1.0 / max(1, n)) if n > 1 else eq
    if to_call == 0:
        if hu_eq >= P["value_eq"]:
            return raise_to(state, cur_bet + max(bb, P["value_size"] * pot))
        if pf_aggr == state.seat and len(state.board) == 3 and n == 1 and _rng.random() < P["cbet_rate"]:
            return raise_to(state, cur_bet + max(bb, P["bluff_size"] * pot))
        if n == 1 and _rng.random() < P["bluff_rate"]:
            return raise_to(state, cur_bet + max(bb, P["bluff_size"] * pot))
        return Action.check()
    if hu_eq >= P["raise_eq"] and state.can("raise"):
        return raise_to(state, cur_bet * 3)
    if eq >= pot_odds + P["call_margin"] + icm:
        return call_or_check(state)
    return passive(state)


class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        try:
            if state.phase == "select":
                return select(state), memory
            if state.phase != "betting":
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), memory
            return bet(state), memory
        except Exception:
            if state.phase == "select":
                return Action("select", cards=(2, 0)), memory
            return passive(state), memory
