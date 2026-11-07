"""Compact learned policy: linear action scores over hand/situation features.

48 weights (4 non-fold actions x 12 features), trained with the
cross-entropy method on tournament points (`optimize.py --space linear`).
Default weights encode a sensible pot-odds/equity prior so an untrained
model is playable.
"""
from ..agent_api import FOLD, CHECK, CALL, RAISE
from .parametric import ParametricBot
from .lookup import concrete

N_FEAT = 12
FEATURE_NAMES = ["bias", "eq", "eq_minus_odds", "pot_odds", "commit", "spr_inv", "n_opp",
                 "pos", "icm", "preflop", "facing", "eff_bb"]
ACTS = [1, 2, 3, 4]  # call, raise small, raise big, all-in (fold = 0 score)


def default_weights():
    w = [[0.0] * N_FEAT for _ in ACTS]
    # call: equity over pot odds, discounted by ICM risk premium
    w[0][2], w[0][8] = 6.0, -4.0
    # raise small / big: same pot-odds term plus raw strength
    w[1][0], w[1][1], w[1][2], w[1][6] = -3.2, 5.0, 6.0, -1.0
    w[2][0], w[2][1], w[2][2], w[2][6] = -5.0, 7.0, 6.0, -1.0
    # all-in: strength, short / committed stacks
    w[3][0], w[3][1], w[3][2], w[3][5], w[3][4], w[3][11], w[3][8] = -6.4, 8.0, 6.0, 1.5, 2.0, -1.5, -3.0
    return [x for row in w for x in row]


class LinearBot(ParametricBot):
    name = "linear"
    DEFAULTS = {"w": None, "temp": 0.0}

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        ParametricBot.__init__(self, params, seed, styles, variant)
        w = self.p.get("w") or default_weights()
        self.W = [w[i * N_FEAT:(i + 1) * N_FEAT] for i in range(len(ACTS))]

    def feats(self, c):
        if c.si == 0:
            ri = 0 if c.unopened else min(3, c.raises + 1)
            k = min(2, max(1, c.n_behind)) if c.unopened else max(1, c.n_opp)
            eq = self.f.preflop_equity(c.hole, k, ri)
        else:
            eq = self.equity(c)
        icm = self.icm_premium(c, c.to_call if c.to_call else int(c.pot * 0.7))
        return [1.0, eq, eq - c.pot_odds, c.pot_odds, c.to_call / float(max(1, c.my_total)),
                1.0 / (1.0 + max(0.0, c.spr)), c.n_opp / 7.0, c.pos, icm,
                1.0 if c.si == 0 else 0.0, 1.0 if c.to_call > 0 else 0.0, min(c.eff_bb, 60) / 50.0]

    def decide(self, c):
        x = self.feats(c)
        best_a, best_s = 0, 0.0
        if c.to_call == 0:
            best_s = -1e9  # folding is never better than checking
        legal = [1]
        if c.o["can_raise"]:
            legal += [2, 3, 4]
        for i, a in enumerate(ACTS):
            if a not in legal:
                continue
            row = self.W[i]
            s = 0.0
            for j in range(N_FEAT):
                s += row[j] * x[j]
            if a == 1 and c.to_call == 0:
                s = max(s, 0.0)
            if s > best_s:
                best_a, best_s = a, s
        if best_a == 0:
            return (FOLD, 0)
        return concrete(self, c, best_a)
