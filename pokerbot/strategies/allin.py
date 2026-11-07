"""All-in (jam) family.

AllInBot: preflop-only jam/fold. Push range = push_pct scaled by stack
depth, ((ref_bb / eff_bb) ** stack_exp); calls all-ins with call_pct.
`nash=1` uses the push/fold equilibrium tables at min(eff_bb, 25) BB for
every stack depth. Postflop (rare: only after a call) it shoves made hands
>= post_class and check/folds otherwise.
push_pct=1, call_pct=1 is the pure "jam every hand" bot.
"""
from ..agent_api import FOLD, CHECK, CALL, RAISE
from ..pushfold import load_pushfold
from .base import BaseBot, Ctx


class AllInBot(BaseBot):
    name = "allin"
    DEFAULTS = {"push_pct": 1.0, "call_pct": 1.0, "ref_bb": 15.0, "stack_exp": 0.0,
                "nash": 0, "post_class": 2, "lam": 0.0}

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        BaseBot.__init__(self, params, seed, styles, variant)
        self.pf = load_pushfold((variant or {}).get("name", "nlhe")) if self.p["nash"] else None

    def decide(self, c):
        p, o = self.p, c.o
        if c.si > 0:
            made = self.f.bucket(c.hole, c.board)[0]
            if made >= p["post_class"] and o["can_raise"]:
                return self.shove(c)
            if made >= p["post_class"]:
                return (CALL, 0)
            return (CHECK, 0) if c.to_call == 0 else (FOLD, 0)
        pct = self.f.preflop_pct(c.hole)
        scale = (p["ref_bb"] / max(1.0, c.eff_bb)) ** p["stack_exp"]
        if c.unopened or (c.raises == 0):
            if self.pf is not None:
                go = self.rng.random() < self.pf.push_prob(self.f.preflop_class(c.hole), max(1, c.n_behind),
                                                          min(c.eff_bb, 25), p["lam"])
            else:
                go = pct <= min(1.0, p["push_pct"] * scale)
            if go and o["can_raise"]:
                return self.shove(c)
            return (CHECK, 0) if c.to_call == 0 else (FOLD, 0)
        # facing a raise / jam
        if self.pf is not None:
            ctype = "bb" if c.me == o["bb_seat"] else ("sb" if c.me == o["sb_seat"] else "nb")
            q = self.pf.call_prob(self.f.preflop_class(c.hole), max(1, c.n_behind + 1), min(c.eff_bb, 25),
                                  ctype, p["lam"])
            ok = q >= 0.5
        else:
            ok = pct <= min(1.0, p["call_pct"] * scale)
        if ok:
            return self.shove(c) if o["can_raise"] else (CALL, 0)
        return (FOLD, 0)
