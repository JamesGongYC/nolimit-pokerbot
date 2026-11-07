"""Lookup-table policies over a configurable state abstraction.

The table maps abstract state -> per-action values (Q, in ICM-points of the
hand outcome) learned offline from exploring rollouts (see search.py:
`train_qtable`). At runtime the bot takes the best legal action for its key
when the key has enough visits, and otherwise falls back to an embedded
EquityBot. `res` (1 or 2) is the abstraction resolution.

Actions: 0 fold, 1 check/call, 2 raise small, 3 raise big, 4 all-in.
"""
import json
import os

from ..agent_api import FOLD, CHECK, CALL, RAISE
from .base import hero_icm
from .parametric import EquityBot

N_ACT = 5
ACT_NAMES = ["fold", "call", "raise_s", "raise_b", "allin"]


def _bucket(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def state_key(bot, c, res=2):
    if c.si == 0:
        pct = bot.f.preflop_pct(c.hole)
        pb = int(pct * (10 if res == 1 else 20))
        pos = 0 if c.pos < 0.34 else (1 if c.pos < 0.8 else 2)
        if c.facing_allin:
            facing = 3
        else:
            facing = min(c.raises, 2)
        stk = _bucket(c.eff_bb, (8, 15, 30) if res == 1 else (5, 8, 12, 18, 30))
        nb = 0 if c.n_alive <= 3 else 1
        return "p%d.%d.%d.%d.%d" % (pb, pos, facing, stk, nb)
    eq = bot.equity(c)
    eb = int(eq * (10 if res == 1 else 20))
    if c.to_call == 0:
        facing = 0
    elif c.facing_allin:
        facing = 3
    else:
        facing = 1 if c.to_call <= 0.5 * (c.pot - c.to_call) else 2
    spr = _bucket(c.spr, (1, 3) if res == 1 else (0.7, 1.5, 3, 6))
    no = 0 if c.n_opp == 1 else 1
    return "f%d.%d.%d.%d.%d.%d" % (c.si, eb, facing, spr, no, int(c.in_pos))


def legal_actions(c):
    o = c.o
    acts = [1]
    if c.to_call > 0:
        acts.append(0)
    if o["can_raise"]:
        acts += [2, 3, 4] if o["max_raise_to"] > o["min_raise_to"] else [4]
    return acts


def concrete(bot, c, a):
    o = c.o
    if a == 0:
        return (FOLD, 0)
    if a == 1:
        return (CALL, 0) if c.to_call else (CHECK, 0)
    if a == 4:
        return (RAISE, o["max_raise_to"])
    if c.si == 0:
        if c.unopened:
            amt = (2.5 if a == 2 else 4.0) * c.bb + c.limpers * c.bb
        else:
            amt = (2.5 if a == 2 else 3.5) * o["current_bet"]
        return (RAISE, int(amt))
    return (RAISE, bot.bet_amount(c, 0.5 if a == 2 else 1.0))


class LookupBot(EquityBot):
    name = "lookup"
    DEFAULTS = {"res": 2, "min_visits": 30, "explore": 0.0, "table": "", "temp": 0.0,
                "preflop_only": 0}

    def __init__(self, params=None, seed=0, styles=None, variant=None, table=None):
        EquityBot.__init__(self, params, seed, styles, variant)
        self.table = table if table is not None else load_qtable(self.p["table"])
        self.record = self.p["explore"] > 0
        self.experience = {} if self.record else None
        self._pending = []
        self._start_eq = None

    def decide(self, c):
        p = self.p
        if p["preflop_only"] and c.si > 0:
            return EquityBot.decide(self, c)
        key = state_key(self, c, int(p["res"]))
        legal = legal_actions(c)
        a = None
        if self.record and self.rng.random() < p["explore"]:
            a = self.rng.choice(legal)
        else:
            row = self.table.get(key) if self.table else None
            if row is not None:
                q, nvis = row
                cands = [x for x in legal if nvis[x] >= p["min_visits"]]
                if cands:
                    a = max(cands, key=lambda x: q[x])
        if a is None:
            act = EquityBot.decide(self, c)
            if not self.record:
                return act
            a = _abstract(act, c)
        if self.record:
            if self._start_eq is None:
                self._start_eq = (list(c.o["stacks"]), list(c.o["contrib"]))
            self._pending.append((key, a))
        return concrete(self, c, a)

    def on_hand_end(self, info):
        EquityBot.on_hand_end(self, info)
        if not self.record or not self._pending:
            self._pending = []
            self._start_eq = None
            return
        stacks0, contrib0 = self._start_eq
        start = [s + c for s, c in zip(stacks0, contrib0)]
        me = self._me
        pay = self._payouts
        alive0 = sum(1 for s in start if s > 0)
        e0 = hero_icm(start, me, pay[:alive0])
        e1 = hero_icm(info["stacks"], me, pay[:alive0])
        r = e1 - e0
        for key, a in self._pending:
            row = self.experience.get(key)
            if row is None:
                row = self.experience[key] = [[0.0] * N_ACT, [0] * N_ACT]
            row[0][a] += r
            row[1][a] += 1
        self._pending = []
        self._start_eq = None

    def act(self, o):
        self._me = o["seat"]
        self._payouts = o["payouts"]
        return EquityBot.act(self, o)


def _abstract(act, c):
    kind, amt = act
    if kind == FOLD:
        return 0 if c.to_call else 1
    if kind in (CHECK, CALL):
        return 1
    if amt >= c.o["max_raise_to"]:
        return 4
    pot_frac = (amt - c.o["current_bet"]) / float(max(1, c.pot + c.to_call))
    return 2 if pot_frac < 0.75 else 3


_QT = {}


def load_qtable(path):
    if not path:
        return None
    if path not in _QT:
        from ..event import ROOT
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        _QT[path] = json.load(open(full))["table"] if os.path.exists(full) else None
    return _QT[path]
