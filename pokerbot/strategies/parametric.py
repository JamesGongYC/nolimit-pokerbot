"""Parametric policies: rule-based heuristic and statistical/equity bots.

Both share preflop logic (ranges by percentile, position, steals, 3-bets,
push/fold below `shove_bb`) and differ postflop:
  HeuristicBot - rules over made-hand / draw classes and bet sizes
  EquityBot    - approximate action values from table equity vs estimated
                 ranges, fold probabilities (opponent model) and pot odds
"""
from ..agent_api import FOLD, CHECK, CALL, RAISE
from ..features import (M_TOP_WEAK, M_TOP_GOOD, M_OVERPAIR, M_MID_PAIR, M_TWO_PAIR, M_SET,
                        M_STRAIGHT, D_NONE, D_OESD, D_FLUSH, D_COMBO)
from ..pushfold import load_pushfold
from .base import BaseBot, range_idx_for_width


class ParametricBot(BaseBot):
    name = "parametric"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        BaseBot.__init__(self, params, seed, styles, variant)
        self.pf = load_pushfold((variant or {}).get("name", "nlhe")) if self.p.get("use_cfr", 1) else None

    # ---------------------------------------------------------------- core
    def decide(self, c):
        p = self.p
        if p["mix"] > 0 and self.rng.random() < p["mix"]:
            return self.random_action(c)
        if c.si == 0:
            return self.preflop(c)
        return self.postflop(c)

    def random_action(self, c):
        r = self.rng.random()
        if r < 0.4:
            return (CALL, 0)
        if r < 0.7:
            return (FOLD, 0)
        return (RAISE, self.bet_amount(c, 0.75))

    # ------------------------------------------------------------- preflop
    def lam(self, c):
        """ICM-ish risk aversion grows as the table shrinks / payouts near."""
        paid = sum(1 for x in c.o["payouts"] if x > 0)
        bubble = max(0.0, 1.0 - (c.n_alive - paid) / 5.0)
        return self.p["icm_risk"] * (0.15 + 0.5 * bubble) * (1.0 if c.chip_rank > 0 else 0.6)

    def cover_frac(self, c):
        if not c.opps:
            return 0.0
        return sum(1 for i in c.opps if c.stacks[i] + c.bets[i] < c.my_total) / float(len(c.opps))

    def endgame_shift(self, c):
        """+ = more aggression late with a short stack near the hand cap."""
        if c.hands_left > 30:
            return 0.0
        return self.p["endgame_push"] * (1.0 if c.chip_rank >= c.n_alive // 2 else -0.5)

    def preflop(self, c):
        p = self.p
        pct = self.f.preflop_pct(c.hole)
        o = c.o
        if c.eff_bb <= p["shove_bb"]:
            return self.pushfold(c, pct)
        rnd = self.rng.random()
        cover = self.cover_frac(c)
        if c.unopened:
            w = p["open_pct"] + p["pos_widen"] * c.pos
            if c.limpers == 0 and c.me in (o["button"], o["sb_seat"]) and c.n_alive > 2:
                w += p["steal_widen"]
            if c.n_alive == 2:
                w = max(w, p["open_pct"] + p["pos_widen"] + p["steal_widen"])
            w *= 1.0 + p["bigstack_pressure"] * cover * 0.6 + self.endgame_shift(c) * 0.3
            w *= 1.0 - min(0.6, 0.25 * self.lam(c))
            if pct <= w:
                size = (p["open_size"] + c.limpers) * c.bb
                return (RAISE, size)
            if c.to_call == 0:
                return (CHECK, 0)
            if p["limp_rate"] > 0 and pct <= w * 1.8 and rnd < p["limp_rate"] and c.to_call <= c.bb:
                return (CALL, 0)
            return (FOLD, 0)
        opener = c.last_raiser
        if c.raises == 1:
            ow = self.opp_width(opener, 0.22)
            v3 = p["threebet_pct"] * max(0.5, min(2.0, (ow / 0.22) ** 0.5))
            amt = int(p["threebet_mult"] * o["current_bet"] + c.limpers * c.bb)
            if pct <= v3:
                return self.raise_or_shove(c, amt)
            late_opener = opener in (o["button"], o["sb_seat"]) or c.n_alive <= 4
            if late_opener and pct <= 0.4 and rnd < p["resteal_rate"] * (1 + p["bigstack_pressure"] * cover):
                return self.raise_or_shove(c, amt)
            k = max(1, sum(1 for i in c.opps if c.bets[i] >= o["current_bet"]))
            ri = range_idx_for_width(ow)
            eq = self.f.preflop_equity(c.hole, k, ri)
            need = c.pot_odds + p["call_margin"] + p["icm_risk"] * self.icm_premium(c, c.to_call)
            if c.to_call <= c.bb * 1.01 and c.me == o["bb_seat"]:
                need -= 0.03
            if eq * (1.0 - 0.05 * c.n_behind) >= need:
                return (CALL, 0)
            return (FOLD, 0)
        # facing 3-bet or more
        if pct <= p["fourbet_pct"]:
            return self.raise_or_shove(c, int(2.3 * o["current_bet"]))
        eq = self.f.preflop_equity(c.hole, max(1, c.n_opp), 3)
        need = c.pot_odds + p["call_margin"] + 0.03 + p["icm_risk"] * self.icm_premium(c, c.to_call)
        if eq >= need:
            return (CALL, 0)
        return (FOLD, 0)

    def raise_or_shove(self, c, amt):
        if amt >= c.my_total * 0.4 or c.eff_bb < 22:
            return self.shove(c)
        return (RAISE, amt)

    def push_width(self, c):
        n = max(1, c.n_behind)
        return min(1.0, 6.0 / max(1.0, c.eff_bb) * n ** -0.75)

    def pushfold(self, c, pct):
        p, o = self.p, c.o
        cls = self.f.preflop_class(c.hole)
        lam = self.lam(c)
        if c.unopened:
            n = max(1, c.n_behind)
            if self.pf is not None:
                if p["push_widen"] == 1.0:
                    go = self.rng.random() < self.pf.push_prob(cls, n, c.eff_bb, lam)
                else:
                    go = pct <= self.pf.push_width(n, c.eff_bb, lam) * p["push_widen"]
            else:
                go = pct <= self.push_width(c) * p["push_widen"]
            w_bonus = p["bigstack_pressure"] * self.cover_frac(c) * 0.3 + self.endgame_shift(c) * 0.2
            if not go and w_bonus > 0 and pct <= self.push_width(c) * (1 + w_bonus):
                go = True
            if go:
                return self.shove(c)
            if c.to_call == 0:
                return (CHECK, 0)
            if p["limp_rate"] > 0 and self.rng.random() < p["limp_rate"] * 0.3 and pct < 0.5:
                return (CALL, 0)
            return (FOLD, 0)
        # facing a raise / shove
        raiser = c.last_raiser
        if self.pf is not None and c.facing_allin:
            # caller type relative to the raiser: n players were behind the pusher
            ctype = "bb" if c.me == o["bb_seat"] else ("sb" if c.me == o["sb_seat"] else "nb")
            n_behind_pusher = max(1, c.n_behind + 1 + (1 if ctype == "nb" else 0))
            q = self.pf.call_prob(cls, n_behind_pusher, c.eff_bb, ctype, lam)
            # blend equilibrium with an exploitative equity check
            ow = self.opp_width(raiser, self.push_width(c) * 1.3)
            eq = self.f.preflop_equity(c.hole, max(1, sum(1 for i in c.opps if c.bets[i] > 0 and
                                                          c.bets[i] >= c.o["current_bet"])),
                                       range_idx_for_width(ow))
            need = c.pot_odds + p["call_shove_margin"] + p["icm_risk"] * self.icm_premium(c, c.to_call)
            a = p["adapt"]
            score = (1 - a) * (q - 0.5) + a * (eq - need) * 4
            return (CALL, 0) if score > 0 else (FOLD, 0)
        ow = self.opp_width(raiser, 0.25)
        eq = self.f.preflop_equity(c.hole, max(1, c.raises), range_idx_for_width(ow))
        shove_need = 0.5 + p["call_shove_margin"] + p["icm_risk"] * self.icm_premium(c, c.my_stack)
        if eq >= shove_need or pct <= p["threebet_pct"]:
            return self.shove(c)
        need = c.pot_odds + p["call_shove_margin"] + p["icm_risk"] * self.icm_premium(c, c.to_call)
        if eq >= need and c.to_call <= c.my_stack * 0.35:
            return (CALL, 0)
        return (FOLD, 0)

    # ------------------------------------------------------------ postflop
    def opp_range_idx(self, c):
        """Estimated range of the main opponent(s) from preflop action."""
        if c.pf_aggressor >= 0 and c.pf_aggressor != c.me:
            w = self.opp_width(c.pf_aggressor, 0.22)
        else:
            w = 0.45
            if self.model is not None and c.opps:
                v = sum(self.model.rate(s, "vpip") for s in c.opps) / len(c.opps)
                w = (1 - self.p["adapt"]) * 0.45 + self.p["adapt"] * v
        ri = range_idx_for_width(w)
        if c.to_call > 0:
            # a bet narrows the range; aggressive bettors narrow it less
            ag = self.model.rate(c.last_raiser, "aggr") if (self.model and c.last_raiser >= 0) else 0.4
            if ag < 0.55 or c.to_call > 0.6 * (c.pot - c.to_call):
                ri += 1
        return min(3, ri)

    def equity(self, c):
        ri = self.opp_range_idx(c)
        lvl = int(self.p["abstraction"])
        e = self.f.post_equity(c.hole, c.board, max(1, c.n_opp), ri, lvl)
        return e

    def postflop(self, c):
        raise NotImplementedError


class EquityBot(ParametricBot):
    """Statistical: chooses the action with the best approximate EV."""

    name = "equity"

    def postflop(self, c):
        p = self.p
        eq = self.equity(c)
        made, draw, _ = self.f.bucket(c.hole, c.board)
        pot = float(c.pot)
        hu = 1.0 / (1.0 + p["multiway_tighten"] * (c.n_opp - 1))
        realize = p["realize"] + (1.0 - p["realize"]) * c.in_pos
        icm_extra = p["icm_risk"] * self.icm_premium(c, max(c.to_call, int(pot * 0.7)))
        drawing = draw != D_NONE and len(c.board) < 5
        options = []
        if c.to_call == 0:
            options.append(((CHECK, 0), eq * pot * realize))
            sizes = [p["bluff_size"], p["value_size"]]
            if c.spr < p["commit_spr"] * 1.5:
                sizes.append(None)
            for fr in sizes:
                amt = c.o["max_raise_to"] if fr is None else self.bet_amount(c, fr)
                b = min(amt, c.my_total) - c.my_bet
                frac = b / max(1.0, pot)
                fp = self.fold_prob(c, frac) * hu
                eqc = max(0.0, eq - p["called_penalty"] * (1 + frac))  # equity when called
                ev = fp * pot + (1 - fp) * (eqc * (pot + 2 * b) - b - icm_extra * b)
                ev += self._style_bias(c, eq, drawing)
                options.append(((RAISE, amt), ev))
        else:
            call = c.to_call
            implied = p["implied"] * pot if drawing else 0.0
            ev_call = eq * (pot + call) - call + implied - (p["call_margin"] + icm_extra) * (pot + call)
            options.append(((FOLD, 0), 0.0))
            options.append(((CALL, 0), ev_call))
            if c.o["can_raise"]:
                for fr in (p["value_size"], None):
                    amt = c.o["max_raise_to"] if (fr is None or c.spr < p["commit_spr"]) else self.bet_amount(c, fr)
                    b = min(amt, c.my_total) - c.my_bet
                    frac = b / max(1.0, pot)
                    fp = self.fold_prob(c, frac) * hu * 0.7
                    eqc = max(0.0, eq - 1.6 * p["called_penalty"])
                    ev = fp * pot + (1 - fp) * (eqc * (pot + b + (b - call)) - b) - icm_extra * b
                    ev += self._style_bias(c, eq, drawing) * 0.5
                    options.append(((RAISE, amt), ev))
        best = max(options, key=lambda x: x[1])
        return best[0]

    def _style_bias(self, c, eq, drawing):
        """Aggression knobs as EV bias in pot units."""
        p = self.p
        pot = float(c.pot)
        bias = 0.0
        if eq >= p["value_eq"]:
            bias += 0.15 * pot
        if drawing:
            bias += (p["semibluff_rate"] - 0.35) * 0.4 * pot
        elif eq < 0.4:
            bias += (p["bluff_rate"] - 0.12) * 0.6 * pot
        if c.pf_aggressor == c.me and c.si == 1:
            bias += (p["cbet_rate"] - 0.6) * 0.3 * pot
        return bias


class HeuristicBot(ParametricBot):
    """Rules over made-hand/draw classes; equity tables only preflop."""

    name = "heuristic"
    DEFAULTS = {"value_class": M_TOP_GOOD, "call_class": M_MID_PAIR, "raise_class": M_TWO_PAIR,
                "big_bet_call_class": M_TOP_GOOD}

    def postflop(self, c):
        p = self.p
        made, draw, tex = self.f.bucket(c.hole, c.board)
        rnd = self.rng.random()
        multi = c.n_opp > 1
        vc = p["value_class"] + (1 if multi and p["multiway_tighten"] > 0.3 else 0)
        wet = tex & 6  # flush/straight-possible board: bet bigger for protection
        strong_draw = draw in (D_OESD, D_FLUSH, D_COMBO) and len(c.board) < 5
        hu_factor = 1.0 if not multi else (1 - p["multiway_tighten"])
        if c.to_call == 0:
            if made >= vc:
                if c.spr < p["commit_spr"] and made >= p["raise_class"]:
                    return self.shove(c)
                return (RAISE, self.bet_amount(c, p["value_size"] * (1.2 if wet else 1.0)))
            if strong_draw and rnd < p["semibluff_rate"] * hu_factor:
                return (RAISE, self.bet_amount(c, p["bluff_size"]))
            if c.pf_aggressor == c.me and c.si == 1 and rnd < p["cbet_rate"] * hu_factor:
                return (RAISE, self.bet_amount(c, p["bluff_size"]))
            if c.n_opp == 1 and c.in_pos and rnd < p["bluff_rate"]:
                return (RAISE, self.bet_amount(c, p["bluff_size"]))
            return (CHECK, 0)
        bet_frac = c.to_call / float(max(1, c.pot - c.to_call))
        if made >= p["raise_class"] or (made >= M_SET):
            if c.o["can_raise"]:
                if c.spr < p["commit_spr"] * 2:
                    return self.shove(c)
                return (RAISE, self.bet_amount(c, p["value_size"]))
            return (CALL, 0)
        if strong_draw and c.o["can_raise"] and rnd < p["bluff_raise_rate"] * hu_factor:
            return self.raise_or_shove(c, self.bet_amount(c, p["value_size"]))
        need_class = p["call_class"] if bet_frac <= 0.6 else p["big_bet_call_class"]
        icm = p["icm_risk"] * self.icm_premium(c, c.to_call)
        if icm > 0.08:
            need_class += 1
        if made >= need_class:
            return (CALL, 0)
        if strong_draw and len(c.board) < 5:
            draw_eq = (0.35 if draw == D_COMBO else 0.2 if draw == D_OESD else 0.19) * \
                      (2 if len(c.board) == 3 else 1)
            if draw_eq + p["implied"] >= c.pot_odds + p["call_margin"] + icm:
                return (CALL, 0)
        if c.pot_odds < 0.15 and made >= M_TOP_WEAK - 2:
            return (CALL, 0)
        return (FOLD, 0)
