"""Team JJ bot v2, Bay Area variant: three-card hold'em with a sealed flop
discard and one exposed retained card (starter-kit/RULES.md).

Mechanism
- Monte Carlo equity with eval7 under a hard per-decision time budget, using
  every public card (board, exposed cards) and our own discard as dead cards.
  Before selection every player holds three cards and keeps the best two.
- Flop selection: keep the pair with the best equity (the three options share
  the same samples), expose the card that gives away least.
- Betting: equity vs pot odds with an ICM risk premium (state.config.payouts),
  push/fold when short, opponent tendencies from table memory (vpip / pfr /
  aggression, JSON in `memory`).
Stateless by design: everything comes from `state` and `memory`.
"""

import json
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
_perf = time.perf_counter

DEFAULTS = {
    "total_us": 330.0,          # hard budget for Monte Carlo per decision
    "select_us": 380.0,
    "max_samples": 500,
    "open_ratio": 1.18,         # open if equity / fair share >= this (first position)
    "pos_bonus": 0.22,          # ... minus this times position (0 first .. 1 button)
    "open_size": 2.5,           # big blinds (+1 per limper)
    "threebet_ratio": 1.75,
    "threebet_mult": 3.0,
    "call_margin": 0.04,        # extra equity over pot odds to call
    "push_bb": 12.0,            # push/fold below this effective stack (big blinds)
    "push_ratio": 1.25,
    "call_push_margin": 0.03,
    "value_eq": 0.62,           # postflop, heads-up-scaled equity to bet for value
    "raise_eq": 0.80,
    "cbet_rate": 0.55,
    "bluff_rate": 0.12,
    "value_size": 0.66,
    "bluff_size": 0.5,
    "icm_weight": 1.0,          # weight on the ICM risk premium
    "adapt": 0.5,               # weight on observed opponent tendencies
    "multiway_pow": 1.0,        # postflop: eq ** (1 / n**pow) heads-up scaling
}

_rng = random.Random()

# warm up the evaluator and move import-time objects out of GC traversals
EVAL([CARD["As"], CARD["Kd"], CARD["Qh"], CARD["Js"], CARD["9c"], CARD["2d"], CARD["3h"]])
import gc  # noqa: E402
gc.collect()
gc.freeze()


def _r(c):
    return RANKS.index(c[0])


# ---------------------------------------------------------------- equity
def mc_equity(hero_options, board, opps, dead, deadline, max_n):
    """Pot share of each hero option vs all opponents, sampled until `deadline`.

    hero_options: list of card-string lists (2 cards, or 3 = keeps the best 2).
    opps: list of (known_cards, n_hidden, keeps_best_2_of_3).
    Every option is scored on the same samples (common random numbers).
    """
    used = set(board) | set(dead)
    for h in hero_options:
        used.update(h)
    for known, _, _ in opps:
        used.update(known)
    deck = [CARD[c] for c in ALL if c not in used]
    bcards = [CARD[c] for c in board]
    need_b = 5 - len(bcards)
    heroes = [[CARD[c] for c in h] for h in hero_options]
    ops = [([CARD[c] for c in k], n, ch) for k, n, ch in opps]
    need = need_b + sum(n for _, n, _ in ops)
    if need > len(deck) or not ops:
        return [0.5] * len(heroes), 0
    tot = [0.0] * len(heroes)
    n = 0
    sample = _rng.sample
    while n < max_n:
        if (n & 3) == 0 and _perf() > deadline:
            break
        draw = sample(deck, need)
        b5 = bcards + draw[:need_b]
        i = need_b
        best_opp = -1
        for known, k, choose in ops:
            cards = known + draw[i:i + k]
            i += k
            if choose and len(cards) == 3:
                c0, c1, c2 = cards
                v = EVAL([c0, c1] + b5)
                w = EVAL([c0, c2] + b5)
                if w > v:
                    v = w
                w = EVAL([c1, c2] + b5)
                if w > v:
                    v = w
            else:
                v = EVAL(cards + b5)
            if v > best_opp:
                best_opp = v
        for j, h in enumerate(heroes):
            if len(h) == 3:
                c0, c1, c2 = h
                hv = EVAL([c0, c1] + b5)
                w = EVAL([c0, c2] + b5)
                if w > hv:
                    hv = w
                w = EVAL([c1, c2] + b5)
                if w > hv:
                    hv = w
            else:
                hv = EVAL(h + b5)
            if hv > best_opp:
                tot[j] += 1.0
            elif hv == best_opp:
                tot[j] += 0.5
        n += 1
    if n == 0:
        return [0.5] * len(heroes), 0
    return [t / n for t in tot], n


def opponents(state):
    """(known, n_hidden, keeps_best_2) per live opponent, and their seats."""
    selected = len(state.board) >= 3 and len(state.cards) == 2
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


# ---------------------------------------------------------------- ICM
def hero_icm(stacks, me, payouts):
    """Malmuth-Harville equity of seat `me` (exact for up to 3 paid places)."""
    st = [float(s) for s in stacks if s > 0]
    if stacks[me] <= 0 or not st:
        return 0.0
    pay = [p for p in payouts[:len(st)]]
    while pay and pay[-1] == 0:
        pay.pop()
    if not pay:
        return 0.0
    s = float(stacks[me])
    total = sum(st)
    others = [float(x) for i, x in enumerate(stacks) if x > 0 and i != me]
    eq = pay[0] * s / total
    if len(pay) >= 2:
        p2 = 0.0
        for a in others:
            p2 += a / total * s / (total - a)
        eq += pay[1] * p2
        if len(pay) >= 3:
            p3 = 0.0
            for i, a in enumerate(others):
                ra = total - a
                for j, b in enumerate(others):
                    if i != j:
                        p3 += a / total * b / ra * s / (ra - b)
            eq += pay[2] * p3
            rest = max(0.0, 1.0 - s / total - p2 - p3)
            if len(pay) > 3:
                eq += rest * sum(pay[3:]) / float(max(1, len(st) - 3))
    return eq


def icm_premium(state, me, vil, risk):
    """Extra equity needed (vs chip EV) to put `risk` more chips in against `vil`."""
    payouts = list(state.config.payouts)
    stacks = [s.stack + 0 for s in state.seats]
    idx = {s.seat: i for i, s in enumerate(state.seats)}
    mi, vi = idx[me.seat], idx[vil.seat]
    pot = state.pot
    r = min(risk, me.stack)
    m = max(0, min(r - state.to_call, vil.stack))
    r_eff = min(r, state.to_call + m)
    big = pot + r_eff + m
    if big <= 0:
        return 0.0
    fold = list(stacks)
    fold[vi] += pot
    win = list(stacks)
    win[mi] += big - r_eff
    win[vi] -= m
    lose = list(stacks)
    lose[mi] -= r_eff
    lose[vi] += big - m
    ef, ew, el = hero_icm(fold, mi, payouts), hero_icm(win, mi, payouts), hero_icm(lose, mi, payouts)
    if ew - el <= 1e-9:
        return 0.0
    return max(0.0, (ef - el) / (ew - el) - r_eff / float(big))


# ---------------------------------------------------------------- memory
def update_model(model, state):
    """Per-seat counts from public events since our last turn:
    [hands, vpip, pfr, post_bets, post_calls, faced_bet, folded_to_bet]."""
    street = model.get("_st", 0)
    cur = model.get("_cur", 0)
    seen = set(model.get("_seen", []))
    for e in state.events:
        k = e.kind
        if k == "hand":
            for s in seen:
                model.setdefault(str(s), [0] * 7)[0] += 1
            seen = set()
            street, cur = 0, 0
            continue
        if k == "board":
            street += 1
            cur = 0
            continue
        if e.seat is None or e.seat == state.seat:
            if k == "raise":
                cur = max(cur, e.amount)
            continue
        row = model.setdefault(str(e.seat), [0] * 7)
        if k in ("blind", "ante"):
            cur = max(cur, e.amount)
            continue
        if k not in ("fold", "check", "call", "raise"):
            continue
        if street == 0 and e.seat not in seen:
            seen.add(e.seat)
            if k in ("call", "raise"):
                row[1] += 1
            if k == "raise":
                row[2] += 1
        elif street == 0 and k == "raise":
            row[2] += 0  # counted once per hand above
        if street > 0:
            if k == "raise":
                row[3] += 1
            elif k == "call":
                row[4] += 1
            if cur > 0 and k in ("fold", "call", "raise"):
                row[5] += 1
                if k == "fold":
                    row[6] += 1
        if k == "raise":
            cur = max(cur, e.amount)
    model["_st"], model["_cur"], model["_seen"] = street, cur, sorted(seen)
    return model


def rate(model, seat, idx, den_idx, prior, k=6.0):
    row = model.get(str(seat))
    if not row:
        return prior
    return (row[idx] + prior * k) / (row[den_idx] + k)


# ---------------------------------------------------------------- policy
def street_info(state):
    street, raises, pf_aggr, last_raiser = 0, 0, None, None
    for e in state.history:
        if e.kind == "board":
            street += 1
            raises = 0
            last_raiser = None
        elif e.kind == "raise":
            raises += 1
            last_raiser = e.seat
            if street == 0:
                pf_aggr = e.seat
    return raises, pf_aggr, last_raiser


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


def select(state, P, t0):
    cards = list(state.cards)
    if len(cards) != 3:
        return Action("select", cards=(2, 0))
    opps, _ = opponents(state)
    opps = [(k, n, True) for k, n, _ in opps]
    options = [[cards[a], cards[b]] for a, b in PAIRS3]
    eqs, _ = mc_equity(options, list(state.board), opps, [], t0 + P["select_us"] * 1e-6, int(P["max_samples"]))
    best = max(range(3), key=lambda i: eqs[i])
    a, b = PAIRS3[best]
    discard = 3 - a - b
    board_ranks = {c[0] for c in state.board}
    ka, kb = cards[a], cards[b]

    def leak(c):
        return (1 if c[0] in board_ranks else 0, _r(c))
    reveal = a if leak(ka) <= leak(kb) else b
    return Action("select", cards=(discard, reveal))


def preflop_eq(state, opps, dead, deadline, P):
    n = len(opps)
    k = min(n, 3)
    eq = mc_equity([list(state.cards)], [], opps[:k], dead, deadline, int(P["max_samples"]))[0][0]
    if n > k:
        eq = eq ** (n / float(k))
    return eq


def bet(state, P, model, t0, eq_override=None):
    me = next((s for s in state.seats if s.seat == state.seat), None)
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
    raises, pf_aggr, last_raiser = street_info(state)
    pre = len(state.board) == 0
    dead = list(state.variant.get("discarded", []) or []) if isinstance(state.variant, dict) else []
    deadline = t0 + P["total_us"] * 1e-6
    if eq_override is not None:
        eq = eq_override
    elif pre:
        eq = preflop_eq(state, opps, dead, deadline, P)
    else:
        eq = mc_equity([list(state.cards)], list(state.board), opps, dead, deadline, int(P["max_samples"]))[0][0]
    fair = 1.0 / (n + 1)
    ratio = eq / fair
    cur_bet = max(s.bet for s in state.seats)
    a = P["adapt"]
    # ICM premium for meaningful risks against the biggest live stack that covers us
    icm = 0.0
    if to_call > 0 and to_call >= 0.15 * me.stack and P["icm_weight"] > 0:
        vil = max(oseats, key=lambda s: s.stack + s.bet)
        icm = P["icm_weight"] * icm_premium(state, me, vil, to_call)
    # a raiser that rarely raises is stronger than average: demand more equity
    tight_pen = 0.0
    if last_raiser is not None and to_call > 0:
        pfr = rate(model, last_raiser, 2, 0, 0.18)
        tight_pen = a * max(-0.04, min(0.08, (0.18 - pfr) * 0.4))

    if pre:
        if eff_bb <= P["push_bb"]:
            if raises == 0:
                if ratio >= P["push_ratio"] - 0.15 * position(state):
                    return raise_to(state, 10 ** 12, my_total)
                return passive(state)
            need = pot_odds + P["call_push_margin"] + icm + tight_pen
            if eq >= need:
                if state.can("raise") and ratio >= 1.6:
                    return raise_to(state, 10 ** 12, my_total)
                return call_or_check(state)
            return passive(state)
        if raises == 0:
            if ratio >= P["open_ratio"] - P["pos_bonus"] * position(state):
                limpers = sum(1 for e in state.history if e.kind == "call")
                return raise_to(state, (P["open_size"] + limpers) * bb, my_total)
            return passive(state)
        if ratio >= P["threebet_ratio"]:
            return raise_to(state, P["threebet_mult"] * cur_bet, my_total)
        if eq >= pot_odds + P["call_margin"] + icm + tight_pen:
            return call_or_check(state)
        return passive(state)

    hu_eq = eq ** (1.0 / (n ** P["multiway_pow"])) if n > 1 else eq
    if to_call == 0:
        if hu_eq >= P["value_eq"]:
            return raise_to(state, cur_bet + max(bb, P["value_size"] * pot), my_total)
        fold_rate = sum(rate(model, s.seat, 6, 5, 0.4) for s in oseats) / n
        bluff = P["bluff_rate"] * (1 + a * (fold_rate - 0.4) * 2)
        if pf_aggr == state.seat and len(state.board) == 3 and n == 1 and _rng.random() < P["cbet_rate"]:
            return raise_to(state, cur_bet + max(bb, P["bluff_size"] * pot), my_total)
        if n == 1 and _rng.random() < bluff:
            return raise_to(state, cur_bet + max(bb, P["bluff_size"] * pot), my_total)
        return Action.check()
    if hu_eq >= P["raise_eq"] and state.can("raise"):
        return raise_to(state, cur_bet * 3, my_total)
    if eq >= pot_odds + P["call_margin"] + icm + tight_pen:
        return call_or_check(state)
    return passive(state)


def _load(memory):
    if not memory:
        return {}
    try:
        m = json.loads(memory)
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


class Agent(poker_agent_sdk.Agent):
    P = DEFAULTS

    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        t0 = _perf()
        P = self.P
        model = _load(memory)
        try:
            model = update_model(model, state)
        except Exception:
            model = {}
        try:
            mem = json.dumps(model, separators=(",", ":")).encode()
        except Exception:
            mem = b""
        if len(mem) > 12000:
            mem = b""
        try:
            if state.phase == "select":
                return select(state, P, t0), mem
            if state.phase != "betting":
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), mem
            return bet(state, P, model, t0), mem
        except Exception:
            if state.phase == "select":
                return Action("select", cards=(2, 0)), mem
            return passive(state), mem
