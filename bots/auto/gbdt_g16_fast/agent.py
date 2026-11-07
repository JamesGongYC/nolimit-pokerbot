"""GBDT family, Bay Area variant: per-action Q trees on top of the Monte Carlo bot.

Same scheme as the hold'em sandbox (fitted policy iteration):
- collection (local sims only): play the base policy; with probability `eps`
  once per hand, deviate preflop to a uniformly random legal abstract action;
  every preflop decision from the deviation on is a sample of
  Q_base(state, action); reward = the hand's change in our ICM equity in
  points (state.config.payouts), read from the hand-end `stack` events.
- runtime: trees (exported from scikit-learn / LightGBM-style histogram
  boosting) are compiled once at import into plain Python if/else; preflop
  only, the tree action replaces the base action when its predicted value beats
  the base action's by `margin`. Hard time guard on the tree evaluation.
Abstract actions: 0 fold, 1 check/call, 2 raise (open 2.5 BB / 3x), 3 all-in.
"""

import importlib.util
import json
import os
import random
import sys
import time

import poker_agent_sdk
from poker_agent_sdk import Action, GameState

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("jjtree_base", os.path.join(HERE, "jjbase.py"))
B = importlib.util.module_from_spec(_spec)
sys.modules["jjtree_base"] = B
_spec.loader.exec_module(B)

FEATURES = ["eq", "ratio", "strength", "eff_bb", "pot_odds", "commit", "n_opp", "pos", "raises",
            "facing_allin", "alive", "chip_rank", "hands_frac", "level", "icm", "pay_top", "opp_pfr", "my_bb"]
N_ACT = 4


def _pair_score(a, b):
    ra, rb = B._r(a), B._r(b)
    hi, lo = max(ra, rb), min(ra, rb)
    s = 2 * hi + lo + (14 if ra == rb else 0) + (3 if a[1] == b[1] else 0)
    gap = hi - lo
    if 0 < gap <= 2:
        s += 3 - gap
    return s


def strength(cards):
    cs = list(cards)
    if len(cs) == 2:
        return min(1.0, _pair_score(*cs) / 52.0)
    sc = sorted((_pair_score(cs[a], cs[b]) for a, b in B.PAIRS3), reverse=True)
    return min(1.0, (sc[0] + 0.25 * sc[1]) / 62.0)


# ---------------------------------------------------------------- trees
def _tree_src(nodes, out, depth):
    def rec(i, d):
        f, thr, l, r, leaf, val = nodes[i]
        pad = "    " * d
        if leaf:
            out.append("%ss += %r" % (pad, val))
            return
        out.append("%sif x[%d] <= %r:" % (pad, f, thr))
        rec(l, d + 1)
        out.append("%selse:" % pad)
        rec(r, d + 1)
    rec(0, depth)


def compile_ensemble(ens, name):
    lines = ["def %s(x):" % name, "    s = %r" % ens["base"]]
    for t in ens["trees"]:
        _tree_src(t, lines, 1)
    if ens.get("mean"):
        lines.append("    s = %r + (s - %r) / %d" % (ens["base"], ens["base"], max(1, len(ens["trees"]))))
    lines.append("    return s")
    return "\n".join(lines)


_MODELS = {}


def load_model(path):
    if not path:
        return None
    full = path if os.path.isabs(path) else os.path.join(HERE, path)
    if full in _MODELS:
        return _MODELS[full]
    if not os.path.exists(full):
        _MODELS[full] = None
        return None
    d = json.load(open(full))
    ns, q = {}, {}
    for a, ens in d["actions"].items():
        exec(compile(compile_ensemble(ens, "q%s" % a), "<tree_%s>" % a, "exec"), ns)
        q[int(a)] = ns["q%s" % a]
    _MODELS[full] = q
    return q


DEFAULTS = dict(B.DEFAULTS, margin=0.35, tree_us=120.0, model="model.json", eps=0.0)
# the default model ships next to this file; compile it at import
load_model(DEFAULTS["model"])


# ---------------------------------------------------------------- features
def features(state, eq, model_mem):
    me = next(s for s in state.seats if s.seat == state.seat)
    opps, oseats = B.opponents(state)
    n = max(1, len(opps))
    bb = float(max(1, state.blinds.big_blind))
    my_total = me.stack + me.bet
    eff = min(my_total, max((s.stack + s.bet for s in oseats), default=my_total))
    raises, pf_aggr, last_raiser = B.street_info(state)
    to_call = state.to_call
    facing_allin = 1.0 if (last_raiser is not None and any(s.seat == last_raiser and s.status == "all_in"
                                                             for s in state.seats)) else 0.0
    tot = [s.stack + s.bet for s in state.seats if s.status != "busted"]
    icm = 0.0
    if to_call > 0 and oseats:
        vil = max(oseats, key=lambda s: s.stack + s.bet)
        icm = B.icm_premium(state, me, vil, to_call)
    pay = list(state.config.payouts)
    pfr = B.rate(model_mem, last_raiser, 2, 0, 0.18) if last_raiser is not None else 0.18
    return [eq, eq * (n + 1), strength(state.cards), min(eff / bb, 100.0),
            to_call / float(state.pot + to_call) if to_call else 0.0, to_call / float(max(1, my_total)),
            float(n), B.position(state), float(raises), facing_allin, float(state.players_remaining),
            float(sum(1 for t in tot if t > my_total)), state.hand_number / float(max(1, state.config.hand_cap)),
            float(state.blind_level), icm, pay[0] / float(max(1, sum(pay))), pfr, my_total / bb]


def legal_abstract(state):
    acts = [1]
    if state.can("fold") and state.to_call > 0:
        acts.append(0)
    r = state.can("raise")
    if r:
        acts += [2, 3] if r.max_amount > r.min_amount else [3]
    return acts


def concrete(state, a):
    if a == 0:
        return Action.fold() if state.to_call > 0 else Action.check()
    if a == 1:
        return B.call_or_check(state)
    r = state.can("raise")
    if not r:
        return B.call_or_check(state)
    if a == 3:
        return Action.raise_to(r.max_amount)
    bb = max(1, state.blinds.big_blind)
    raised = any(e.kind == "raise" for e in state.history)
    cur = max(s.bet for s in state.seats)
    amt = 3 * cur if raised else 2.5 * bb
    return Action.raise_to(int(max(r.min_amount, min(r.max_amount, amt))))


def abstract(state, action):
    k = action.kind
    if k == "fold":
        return 0 if state.to_call > 0 else 1
    if k in ("check", "call"):
        return 1
    if k == "raise":
        r = state.can("raise")
        if r and action.amount >= r.max_amount:
            return 3
        return 2
    return 1


# ---------------------------------------------------------------- agent
class Agent(poker_agent_sdk.Agent):
    P = DEFAULTS

    def __init__(self):
        self.experience = []   # local collection only
        self._rows = []
        self._dev = False
        self._hand = None
        self._start = None
        self._rng = random.Random()

    def seed(self, s):
        self._rng.seed(s)

    # -- collection bookkeeping (local sims; the platform never sets eps) --
    def _close_hand(self, final_stacks):
        if self._start is not None and self._rows:
            stacks0, me, pay = self._start
            r = B.hero_icm(final_stacks, me, pay) - B.hero_icm(stacks0, me, pay)
            for x, a in self._rows:
                self.experience.append(x + [a, round(r, 4)])
        self._rows, self._dev, self._start = [], False, None

    def on_table_end(self, final_stacks):
        self._close_hand(list(final_stacks))

    def _track(self, state):
        if self.P.get("eps", 0.0) <= 0:
            return
        ends = {}
        for e in state.events:
            if e.kind == "stack" and e.seat is not None:
                ends[e.seat] = e.amount
        if self._hand is not None and self._hand != state.hand_number and self._start is not None:
            if ends:
                final = list(self._start[0])
                idx = {s.seat: i for i, s in enumerate(state.seats)}
                for seat, amt in ends.items():
                    if seat in idx:
                        final[idx[seat]] = amt
                self._close_hand(final)
            else:
                self._rows, self._dev, self._start = [], False, None
        if self._hand != state.hand_number:
            self._hand = state.hand_number
            self._start = ([s.stack + s.bet for s in state.seats],
                           next(i for i, s in enumerate(state.seats) if s.seat == state.seat),
                           list(state.config.payouts))

    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        t0 = B._perf()
        P = self.P
        model_mem = B._load(memory)
        try:
            model_mem = B.update_model(model_mem, state)
        except Exception:
            model_mem = {}
        try:
            mem = json.dumps(model_mem, separators=(",", ":")).encode()
        except Exception:
            mem = b""
        if len(mem) > 12000:
            mem = b""
        try:
            self._track(state)
            if state.phase == "select":
                return B.select(state, P, t0), mem
            if state.phase != "betting":
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), mem
            if state.board:
                return B.bet(state, P, model_mem, t0), mem
            # preflop: one equity estimate shared by the base policy and the trees
            opps, _ = B.opponents(state)
            dead = []
            eq = B.preflop_eq(state, opps, dead, t0 + P["total_us"] * 1e-6, P) if opps else 0.5
            base = B.bet(state, P, model_mem, t0, eq_override=eq)
            x = None
            if P.get("eps", 0.0) > 0 or P.get("model"):
                x = features(state, eq, model_mem)
            if P.get("eps", 0.0) > 0:
                legal = legal_abstract(state)
                if not self._dev and self._rng.random() < P["eps"]:
                    self._dev = True
                    self._rows = []
                    a = self._rng.choice(legal)
                    act = concrete(state, a)
                else:
                    act, a = base, abstract(state, base)
                if self._dev:
                    self._rows.append((x, a))
                return act, mem
            q = load_model(P.get("model"))
            if not q:
                return base, mem
            ab = abstract(state, base)
            legal = [a for a in legal_abstract(state) if a in q]
            deadline = t0 + (P["total_us"] + P["tree_us"]) * 1e-6
            vals = {}
            for a in ([ab] if ab in legal else []) + [a for a in legal if a != ab]:
                if B._perf() > deadline:
                    break
                vals[a] = q[a](x)
            if not vals or ab not in vals:
                return base, mem
            best = max(vals, key=lambda a: vals[a])
            if vals[best] - vals[ab] <= P["margin"]:
                return base, mem
            return concrete(state, best), mem
        except Exception:
            if state.phase == "select":
                return Action("select", cards=(2, 0)), mem
            return B.passive(state), mem


# evolved parameters: gbdt_g16 (main)
Agent.P = dict(Agent.P, **{"open_ratio": 1.3606, "pos_bonus": 0.2397, "open_size": 2.625, "threebet_ratio": 1.8086, "threebet_mult": 2.6373, "call_margin": -0.0333, "push_bb": 11.9984, "push_ratio": 1.2029, "call_push_margin": -0.0282, "value_eq": 0.5738, "raise_eq": 0.8105, "cbet_rate": 0.4815, "bluff_rate": 0.0615, "value_size": 0.8698, "bluff_size": 0.8354, "icm_weight": 0.3299, "adapt": 0.3571, "multiway_pow": 0.8139, "margin": 0.5528, "model": "model_main_g16.json", "eps": 0.0})

# latency guard for submission
Agent.P = dict(Agent.P, total_us=220.0, select_us=260.0, tree_us=100.0)
