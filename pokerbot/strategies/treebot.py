"""Tree-ensemble policies (gradient-boosted trees / random forests).

Learning (offline, pokerbot/treelearn.py): a CollectorBot plays a base
policy and, with probability `eps` per decision (at most once per hand),
deviates to a uniformly random legal abstract action. Every decision from
the deviation point on is a valid sample of Q_base(state, action): the
continuation is pure base policy. Reward = hand's change in the bot's ICM
equity in official points. One regressor per abstract action is fit
(sklearn HistGradientBoosting = LightGBM/XGBoost-style histogram GBDT, or a
random forest); the improved policy picks argmax_a Q(s, a), optionally only
when it beats the base action by `margin` (conservative improvement).
Repeating collect -> fit with the new policy is fitted policy iteration.

Runtime: trees are stored as JSON and compiled once at load into nested
Python if/else source (no numpy/sklearn needed; ~2-5 us per tree).
"""
import json
import os
import time

from ..agent_api import FOLD, CHECK, CALL, RAISE
from .base import hero_icm
from .lookup import legal_actions, concrete, _abstract, N_ACT
from .parametric import ParametricBot, HeuristicBot

FEATURE_NAMES = ["street", "pct", "eq", "made", "draw", "texture", "pot_odds", "commit", "spr",
                 "eff_bb", "n_opp", "n_behind", "pos", "raises", "facing_allin", "pf_aggr_me",
                 "icm", "n_alive", "chip_rank", "hands_left", "level", "opp_vpip", "opp_pfr",
                 "opp_fold_post", "opp_aggr", "cover", "pot_bb", "limpers", "in_pos"]


def feature_vector(bot, c):
    f = bot.f
    pre = c.si == 0
    pct = f.preflop_pct(c.hole)
    if pre:
        made = draw = tex = -1
        k = min(3, max(1, c.n_behind if c.unopened else c.n_opp))
        eq = f.preflop_equity(c.hole, k, 0 if c.unopened else min(3, c.raises + 1))
    else:
        made, draw, tex = f.bucket(c.hole, c.board)
        eq = f.bucket_equity(len(c.board), made, draw, tex, max(1, c.n_opp), 0, 2)
    m = bot.model
    main = c.last_raiser if c.last_raiser >= 0 and c.last_raiser != c.me else (c.opps[0] if c.opps else -1)
    if m is not None and main >= 0:
        ov, op, of, oa = m.rate(main, "vpip"), m.rate(main, "pfr"), m.rate(main, "fold_post"), m.rate(main, "aggr")
    else:
        ov, op, of, oa = 0.3, 0.15, 0.4, 0.4
    icm = bot.icm_premium(c, c.to_call if c.to_call else int(0.7 * c.pot))
    cover = (sum(1 for i in c.opps if c.stacks[i] + c.bets[i] < c.my_total) / float(len(c.opps))) if c.opps else 0.0
    return [c.si, pct, eq, made, draw, tex, c.pot_odds, c.to_call / float(max(1, c.my_total)), min(c.spr, 20.0),
            min(c.eff_bb, 100.0), c.n_opp, c.n_behind, c.pos, c.raises, 1 if c.facing_allin else 0,
            1 if c.pf_aggressor == c.me else 0, icm, c.n_alive, c.chip_rank, c.hands_left, c.o["level"],
            ov, op, of, oa, cover, c.pot / float(c.bb), c.limpers, 1 if c.in_pos else 0]


# ---------------------------------------------------------------- compiler
def _tree_src(nodes, out, depth):
    """nodes: list of [feat, thr, left, right, is_leaf, value]; emits `s += v` code."""
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
    if path in _MODELS:
        return _MODELS[path]
    from ..event import ROOT
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        _MODELS[path] = None
        return None
    d = json.load(open(full))
    ns = {}
    fns = {}
    for a, ens in d["actions"].items():
        src = compile_ensemble(ens, "q%s" % a)
        exec(compile(src, "<tree_%s>" % a, "exec"), ns)
        fns[int(a)] = ns["q%s" % a]
    model = {"q": fns, "meta": d.get("meta", {}), "streets": d.get("streets", "all")}
    _MODELS[path] = model
    return model


# ------------------------------------------------------------------ bots
class TreeBot(HeuristicBot):
    """Q-tree policy on top of a base heuristic policy (same params)."""

    name = "tree"
    DEFAULTS = dict(HeuristicBot.DEFAULTS, model="", margin=0.0, tree_streets="all", tree_budget_us=450.0)

    def act(self, o):
        self._t0 = time.perf_counter()
        return HeuristicBot.act(self, o)

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        HeuristicBot.__init__(self, params, seed, styles, variant)
        self.m = load_model(self.p["model"]) if self.p["model"] else None

    def decide(self, c):
        base = HeuristicBot.decide(self, c)
        m = self.m
        if m is None:
            return base
        ts = self.p["tree_streets"]
        if (ts == "pre" and c.si > 0) or (ts == "post" and c.si == 0):
            return base
        x = feature_vector(self, c)
        q = m["q"]
        legal = [a for a in legal_actions(c) if a in q]
        if not legal:
            return base
        ab = _abstract(base, c)
        # hard time guard: evaluate the base action first, stop when over budget
        deadline = self._t0 + self.p["tree_budget_us"] * 1e-6
        order = ([ab] if ab in q and ab in legal else []) + [a for a in legal if a != ab]
        vals = {}
        for a in order:
            if time.perf_counter() > deadline:
                break
            vals[a] = q[a](x)
        if not vals:
            return base
        best = max(vals, key=lambda a: vals[a])
        if ab in vals and vals[best] - vals[ab] <= self.p["margin"]:
            return base
        return concrete(self, c, best)


class CollectorBot(ParametricBot):
    """Explores around an inner base policy and records Q-learning samples."""

    name = "collector"
    DEFAULTS = {"base": "heur_tag", "eps": 0.12}

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        ParametricBot.__init__(self, params, seed, styles, variant)
        from . import make_agent
        self.inner = make_agent(self.p["base"], seed=seed + 7, variant=variant)
        self.experience = []
        self._rows = []
        self._dev = False
        self._start = None

    def act(self, o):
        self._ensure(o)
        if self._start is None:
            self._start = ([s + b for s, b in zip(o["stacks"], o["contrib"])], o["seat"], o["payouts"])
        from .base import Ctx
        c = Ctx(o)
        x = feature_vector(self, c)
        legal = legal_actions(c)
        if not self._dev and self.rng.random() < self.p["eps"]:
            self._dev = True
            self._rows = []  # earlier decisions were followed by a deviation -> drop
            a = self.rng.choice(legal)
            act = concrete(self, c, a)
            self.inner_act_skip = True
        else:
            act = self.inner.act(o)
            a = _abstract(act, c)
        self._rows.append((x, a))
        return self.finish(c, act)

    def on_hand_end(self, info):
        ParametricBot.on_hand_end(self, info)
        try:
            self.inner.on_hand_end(info)
        except Exception:
            pass
        if self._start is not None and self._rows:
            start, me, pay = self._start
            alive0 = sum(1 for s in start if s > 0)
            r = hero_icm(info["stacks"], me, pay[:alive0]) - hero_icm(start, me, pay[:alive0])
            for x, a in self._rows:
                self.experience.append(x + [a, round(r, 4)])
        self._rows = []
        self._dev = False
        self._start = None
