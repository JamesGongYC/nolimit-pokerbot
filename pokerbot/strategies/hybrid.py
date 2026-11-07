"""Hybrid: lookup-table / CFR push-fold preflop + equity & opponent-model
postflop + ICM-aware risk control.

Preflop order: (1) push/fold equilibrium table when short, (2) learned
lookup table when the state has enough visits, (3) parametric ranges.
Postflop: EquityBot EV choice with opponent-model fold rates; when an
opponent's observed tendencies are extreme (station / maniac / nit), the
bot shifts bluff and value thresholds by `adapt`.
"""
from ..agent_api import FOLD, CHECK, CALL, RAISE
from .lookup import LookupBot, state_key, legal_actions, concrete
from .parametric import EquityBot, HeuristicBot


class HybridBot(LookupBot):
    """`post`: 0 = equity/EV postflop, 1 = heuristic postflop."""
    name = "hybrid"
    DEFAULTS = dict(LookupBot.DEFAULTS, **dict(HeuristicBot.DEFAULTS, exploit_shift=0.08, post=0))

    def decide(self, c):
        p = self.p
        if c.si == 0:
            if c.eff_bb <= p["shove_bb"]:
                return self.pushfold(c, self.f.preflop_pct(c.hole))
            if self.table:
                key = state_key(self, c, int(p["res"]))
                row = self.table.get(key)
                if row is not None:
                    q, nvis = row
                    cands = [x for x in legal_actions(c) if nvis[x] >= p["min_visits"]]
                    if cands:
                        return concrete(self, c, max(cands, key=lambda x: q[x]))
            return self.preflop(c)
        return self.exploit_postflop(c)

    def exploit_postflop(self, c):
        p = self.p
        saved = (p["bluff_rate"], p["semibluff_rate"], p["value_eq"], p["call_margin"])
        if self.model is not None and c.opps and p["adapt"] > 0:
            ftb = sum(self.model.rate(s, "fold_post") for s in c.opps) / len(c.opps)
            vpip = sum(self.model.rate(s, "vpip") for s in c.opps) / len(c.opps)
            agg = self.model.rate(c.last_raiser, "aggr") if c.last_raiser >= 0 else 0.4
            sh = p["exploit_shift"] * p["adapt"]
            if ftb < 0.3 or vpip > 0.6:      # station: bluff less, value thinner
                p["bluff_rate"] *= 0.3
                p["semibluff_rate"] *= 0.6
                p["value_eq"] -= sh
            elif ftb > 0.6:                   # folds too much: bluff more
                p["bluff_rate"] = min(0.8, p["bluff_rate"] * 2 + 0.1)
            if agg > 0.6 and c.to_call > 0:   # maniac: call lighter
                p["call_margin"] -= sh
        try:
            if p["post"]:
                return HeuristicBot.postflop(self, c)
            return EquityBot.postflop(self, c)
        finally:
            p["bluff_rate"], p["semibluff_rate"], p["value_eq"], p["call_margin"] = saved
