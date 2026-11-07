"""Transformer Q-model on top of jamtop_tuned (Bay Area variant).

The model (tools/tftrain.py) was trained on millions of OUR real preflop
decisions from scored rounds, each labelled with the hand's real change in our
ICM points. At runtime a numpy forward pass (no torch, ~100-200 us) scores
fold / call / all-in for the current preflop spot; the bot deviates from the
jamtop_tuned base action only when the predicted gain exceeds `margin` points.
Hard time guard: skip the model if the decision is already slow.
Everything else (flop selection, postflop) is jamtop_tuned.
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
import poker_agent_sdk
from poker_agent_sdk import Action, GameState

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("jj_tf_base", os.path.join(HERE, "jtbase.py"))
B = importlib.util.module_from_spec(_spec)
sys.modules["jj_tf_base"] = B
_spec.loader.exec_module(B)

RANKS = "23456789TJQKA"
SUITS = "cdhs"
POS = ["sb", "bb", "ep", "mp", "co", "btn"]
H = 10
D, NH = 32, 4
MARGIN = 0.3
BUDGET_US = 600.0

W = {k: v.astype(np.float64) for k, v in np.load(os.path.join(HERE, "model.npz")).items()}
L = sum(1 for k in W if k.endswith("self_attn.in_proj_weight"))


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def forward(ranks, suits, ctx, hist, hmask):
    """Exact numpy replica of tools/tftrain.Model for one decision."""
    c = W["rank.weight"][ranks] + W["suit.weight"][suits] + W["slot.weight"][[0, 1, 2]]
    x = np.concatenate([(W["ctx.weight"] @ ctx + W["ctx.bias"])[None], c,
                        hist @ W["hist.weight"].T + W["hist.bias"] + W["hpos"]], 0)
    keymask = np.concatenate([np.zeros(4, bool), hmask])
    dh = D // NH
    for i in range(L):
        p = "enc.layers.%d." % i
        h = _ln(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
        qkv = h @ W[p + "self_attn.in_proj_weight"].T + W[p + "self_attn.in_proj_bias"]
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        T_ = x.shape[0]
        q = q.reshape(T_, NH, dh).transpose(1, 0, 2)
        k = k.reshape(T_, NH, dh).transpose(1, 0, 2)
        v = v.reshape(T_, NH, dh).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
        att[:, :, keymask] = -1e30
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ v).transpose(1, 0, 2).reshape(T_, D)
        x = x + o @ W[p + "self_attn.out_proj.weight"].T + W[p + "self_attn.out_proj.bias"]
        h = _ln(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
        f = np.maximum(0.0, h @ W[p + "linear1.weight"].T + W[p + "linear1.bias"])
        x = x + f @ W[p + "linear2.weight"].T + W[p + "linear2.bias"]
    y = _ln(x[0], W["norm.weight"], W["norm.bias"])
    return W["head.weight"] @ y + W["head.bias"]


def encode(state):
    """Same features as tools/tfdata.py, from the live GameState (preflop)."""
    me = state.seat
    seats = {s.seat: s for s in state.seats}
    start = {s.seat: s.stack + s.bet for s in state.seats}
    live = sorted(s.seat for s in state.seats if s.status != "busted")
    st = start[me]
    others = [start[k] for k in live if k != me]
    eff = min(st, max(others)) if others else st
    bb = float(max(1, state.blinds.big_blind))
    n = len(live)
    if me in live and state.button in live:
        d = (live.index(me) - live.index(state.button)) % n
        if d == 0:
            pos = "btn"
        elif d == 1:
            pos = "sb"
        elif d == 2:
            pos = "bb"
        else:
            left, k = n - 3, d - 3
            pos = "co" if k == left - 1 else ("ep" if k < (left - 1) / 2.0 else "mp")
    else:
        pos = "btn"
    hist = []
    raises = 0
    for e in state.history:
        if e.kind == "board":
            break
        if e.kind == "blind":
            hist.append([((e.seat - me) % 8) / 7.0, 1.0, math.log1p(e.amount / bb), 1.0])
            continue
        if e.kind not in ("fold", "check", "call", "raise"):
            continue
        kc = {"fold": 0.0, "check": 1.0, "call": 1.0, "raise": 2.0}[e.kind]
        if e.kind == "raise" and e.amount >= start.get(e.seat, 1e18) * 0.95:
            kc = 3.0
        rel = 0.0 if e.seat == me else ((e.seat - me) % 8) / 7.0
        hist.append([rel, kc, math.log1p(e.amount / bb), 0.0])
        if e.kind == "raise":
            raises += 1
    pot = sum(s.bet for s in state.seats)
    facing_allin = 1.0 if any(h_[1] == 3.0 for h_ in hist) else 0.0
    pay = list(state.config.payouts)
    ctx = [math.log1p(eff / bb), math.log1p(st / bb)] + [1.0 if pos == p_ else 0.0 for p_ in POS] + \
          [n / 8.0, raises / 3.0, state.to_call / max(1.0, st), math.log1p(pot / bb), facing_allin,
           state.hand_number / 250.0, math.log(bb / 20.0) / 5.0, pay[0] / float(max(1, sum(pay))),
           1.0 if n == 2 else 0.0, 1.0 if n <= 3 else 0.0]
    hh = hist[-H:]
    hh = [[0.0, 0.0, 0.0, 0.0]] * (H - len(hh)) + hh
    cards = [RANKS.index(c[0]) * 4 + SUITS.index(c[1]) for c in state.cards]
    ranks = np.array([c // 4 for c in cards])
    m, suits = {}, []
    for c in cards:
        suits.append(m.setdefault(c % 4, len(m)))
    hist = np.array(hh, np.float64)
    return ranks, np.array(suits), np.array(ctx, np.float64), hist, np.abs(hist).sum(-1) == 0


def abstract(action, state):
    if action.kind == "fold":
        return 0
    if action.kind in ("check", "call"):
        return 1
    if action.kind == "raise":
        r = state.can("raise")
        return 3 if (r and action.amount >= r.max_amount) else 2
    return 1


def concrete(a, state):
    if a == 0:
        return Action.fold() if state.to_call > 0 else Action.check()
    if a == 1:
        return Action.call() if state.can("call") else Action.check()
    return B.shove(state)


# warm up once at import (first numpy calls are slower)
try:
    forward(np.array([12, 11, 10]), np.array([0, 1, 2]), np.zeros(18), np.zeros((H, 4)), np.ones(H, bool))
except Exception:
    pass


class Agent(poker_agent_sdk.Agent):
    P = B.Agent.P

    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        t0 = time.perf_counter()
        try:
            if state.phase == "select":
                return B.select(state), memory
            if state.phase != "betting":
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), memory
            base = B.bet(state, self.P)
            if state.board or len(state.cards) != 3:
                return base, memory
            if (time.perf_counter() - t0) * 1e6 > BUDGET_US:
                return base, memory
            q = forward(*encode(state))
            ab = abstract(base, state)
            legal = [1]
            if state.to_call > 0 and state.can("fold"):
                legal.append(0)
            if state.can("raise") or (state.can("call") and state.to_call > 0):
                legal.append(3)
            if ab not in legal:
                return base, memory
            best = max(legal, key=lambda a: q[a])
            if best != ab and q[best] - q[ab] > MARGIN:
                return concrete(best, state), memory
            return base, memory
        except Exception:
            if state.phase == "select":
                return Action("select", cards=(2, 0)), memory
            return B.passive(state), memory
