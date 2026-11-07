"""Real-time Monte Carlo equity bot (bounded by a hard time budget).

At each postflop decision it samples opponent holdings from ranges inferred
by the opponent model (preflop aggressor -> PFR width, callers -> VPIP
width; a bet facing us narrows the range) and random runouts, evaluating
with the fast evaluator until `mc_budget_us` elapses or `mc_max` samples.
The MC estimate is blended with the precomputed bucket equity as a prior
worth `prior_n` samples, so short budgets degrade gracefully. Decisions
then use EquityBot's EV comparison. Preflop uses the equity tables.
"""
import time

from ..cards import hand_class
from .parametric import EquityBot


class MonteCarloBot(EquityBot):
    name = "montecarlo"
    DEFAULTS = {"mc_budget_us": 250.0, "mc_max": 300, "prior_n": 12.0, "range_tighten": 0.6,
                "mc_preflop": 0}

    def _widths(self, c):
        p = self.p
        out = []
        for s in c.opps:
            if s == c.pf_aggressor:
                w = self.opp_width(s, 0.22)
            elif self.model is not None:
                w = (1 - p["adapt"]) * 0.45 + p["adapt"] * self.model.rate(s, "vpip")
            else:
                w = 0.45
            if c.to_call > 0 and s == c.last_raiser:
                w *= p["range_tighten"]
            out.append(max(0.03, min(1.0, w)))
        return out

    def mc_equity(self, c):
        p = self.p
        deadline = time.perf_counter() + p["mc_budget_us"] * 1e-6
        v = self.f.variant
        pct = self.f.pf_pct
        known = set(c.hole) | set(c.board)
        deck = [x for x in v.deck if x not in known]
        widths = self._widths(c)
        need_board = 5 - len(c.board)
        hole, board = list(c.hole), list(c.board)
        hv = v.hand_value
        rng = self.rng
        tot = 0.0
        n = 0
        maxn = int(p["mc_max"])
        k = len(widths)
        tries = 0
        while n < maxn and tries < 3 * maxn:
            tries += 1
            if (tries & 3) == 0 and time.perf_counter() > deadline:
                break
            draw = rng.sample(deck, 2 * k + need_board + 6)
            used = 0
            opps = []
            pool = draw
            ok = True
            for w in widths:
                got = False
                # rejection-sample a hand inside the range from the remaining draw
                for _t in range(3):
                    if used + 2 > len(pool) - need_board:
                        break
                    a, b = pool[used], pool[used + 1]
                    used += 2
                    if pct is None or w >= 1.0 or pct[hand_class(a, b)] <= w:
                        opps.append((a, b))
                        got = True
                        break
                if not got:
                    ok = False
                    break
            if not ok:
                continue  # rejected (narrow range); `tries` bounds the loop
            run = board + pool[used:used + need_board]
            me = hv(hole, run)
            best = max(hv(o, run) for o in opps)
            if me > best:
                tot += 1.0
            elif me == best:
                tot += 0.5
            n += 1
        return tot, n

    def equity(self, c):
        prior = EquityBot.equity(self, c)
        if c.si == 0 and not self.p["mc_preflop"]:
            return prior
        s, n = self.mc_equity(c)
        m = self.p["prior_n"]
        return (s + m * prior) / (n + m)
