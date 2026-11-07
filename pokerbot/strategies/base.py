"""Shared machinery for all strategies: decision context, opponent model,
fast hero-ICM, bet sizing, style presets and the parameter schema.

Only officially observable data (the obs dict) is used. Every bot instance
lives for exactly one table, so all memory here is per-table.
"""
import random

from ..agent_api import Agent, FOLD, CHECK, CALL, RAISE
from ..features import Features, RANGES
from ..variant import Variant

_FEATURES = {}


def features_for(variant_cfg=None):
    key = repr(sorted((variant_cfg or {"name": "nlhe"}).items()))
    f = _FEATURES.get(key)
    if f is None:
        f = Features(Variant.from_config(variant_cfg or {"name": "nlhe"}))
        _FEATURES[key] = f
    return f


# --------------------------------------------------------------------- ICM
def hero_icm(stacks, me, payouts):
    """ICM equity of seat `me` only (Malmuth-Harville), O(n^(paid-1)).

    Exact for up to 4 paid places; deeper places are approximated by
    spreading the remaining probability evenly (rarely matters)."""
    st = [s for s in stacks if s > 0]
    if stacks[me] <= 0:
        return 0.0
    n = len(st)
    pay = list(payouts[:n])
    while pay and pay[-1] == 0:
        pay.pop()
    if not pay:
        return 0.0
    s = float(stacks[me])
    total = float(sum(st))
    others = [float(x) for i, x in enumerate(stacks) if x > 0 and i != me]
    eq = pay[0] * s / total
    if len(pay) == 1:
        return eq
    # place 2
    p_by_place = [s / total]
    p2 = 0.0
    for a in others:
        p2 += a / total * s / (total - a)
    eq += pay[1] * p2
    p_by_place.append(p2)
    if len(pay) >= 3:
        p3 = 0.0
        for i, a in enumerate(others):
            pa = a / total
            ra = total - a
            for j, b in enumerate(others):
                if i != j:
                    p3 += pa * b / ra * s / (ra - b)
        eq += pay[2] * p3
        p_by_place.append(p3)
    if len(pay) >= 4:
        p4 = 0.0
        no = len(others)
        for i in range(no):
            a = others[i]
            pa = a / total
            ra = total - a
            for j in range(no):
                if j == i:
                    continue
                b = others[j]
                pb = pa * b / ra
                rb = ra - b
                for k in range(no):
                    if k == i or k == j:
                        continue
                    c = others[k]
                    p4 += pb * c / rb * s / (rb - c)
        eq += pay[3] * p4
        p_by_place.append(p4)
    if len(pay) > 4:
        rest = max(0.0, 1.0 - sum(p_by_place))
        k = len(pay) - 4
        eq += rest * sum(pay[4:]) / float(max(k, n - 4))
    return eq


# ------------------------------------------------------------- opp model
class OppModel(object):
    """Regularised (Beta-prior) per-seat tendencies within the current table."""

    PRIORS = {"vpip": (0.30, 8.0), "pfr": (0.15, 8.0), "fold_to_bet": (0.45, 6.0),
              "fold_post": (0.40, 6.0),
              "aggr": (0.40, 6.0), "allin": (0.05, 10.0)}

    def __init__(self, n, prior_scale=1.0):
        self.n = n
        self.c = [{"hands": 0, "vpip": 0, "pfr": 0, "face": 0, "fold_face": 0, "agg": 0,
                   "pas": 0, "pf_act": 0, "allin": 0, "face_post": 0, "fold_post": 0}
                  for _ in range(n)]
        self.prior_scale = prior_scale

    def rate(self, seat, name):
        mean, k = self.PRIORS[name]
        k *= self.prior_scale
        c = self.c[seat]
        if name == "vpip":
            return (c["vpip"] + mean * k) / (c["hands"] + k)
        if name == "pfr":
            return (c["pfr"] + mean * k) / (c["hands"] + k)
        if name == "fold_to_bet":
            return (c["fold_face"] + mean * k) / (c["face"] + k)
        if name == "fold_post":
            return (c["fold_post"] + mean * k) / (c["face_post"] + k)
        if name == "aggr":
            return (c["agg"] + mean * k) / (c["agg"] + c["pas"] + k)
        if name == "allin":
            return (c["allin"] + mean * k) / (c["pf_act"] + k)
        return mean

    def update(self, info, stacks_before=None):
        hist = info["history"]
        cur = 0
        street = -1
        bets = {}
        seen = set()
        vp, pr = set(), set()
        for si, seat, kind, amt in hist:
            if si != street:
                street = si
                bets = {}
                cur = 0
            if kind in ("sb", "bb"):
                bets[seat] = amt
                cur = max(cur, amt)
                continue
            seen.add(seat)
            c = self.c[seat]
            before = bets.get(seat, 0)
            facing = cur > before
            if facing:
                c["face"] += 1
                if kind == FOLD:
                    c["fold_face"] += 1
                if si > 0:
                    c["face_post"] += 1
                    if kind == FOLD:
                        c["fold_post"] += 1
            if si == 0:
                c["pf_act"] += 1
                if kind in (CALL, RAISE):
                    vp.add(seat)
                if kind == RAISE:
                    pr.add(seat)
            elif kind == RAISE:
                c["agg"] += 1
            elif kind == CALL:
                c["pas"] += 1
            if kind == RAISE and stacks_before is not None and amt >= stacks_before[seat] * 0.95:
                c["allin"] += 1
            if kind in (CALL, RAISE):
                bets[seat] = amt
                cur = max(cur, amt)
        for s in seen:
            self.c[s]["hands"] += 1
        for s in vp:
            self.c[s]["vpip"] += 1
        for s in pr:
            self.c[s]["pfr"] += 1


def range_idx_for_width(width):
    """Map an estimated range width (fraction of hands) to a RANGES index."""
    best, bi = 9.0, 0
    for i, r in enumerate(RANGES):
        d = abs(r - width)
        if d < best:
            best, bi = d, i
    return bi


# ----------------------------------------------------------------- context
class Ctx(object):
    __slots__ = ("o", "me", "n", "si", "hole", "board", "stacks", "bets", "pot", "to_call", "bb",
                 "my_stack", "my_bet", "my_total", "opps", "n_opp", "eff", "eff_bb", "my_bb",
                 "pot_odds", "n_behind", "pos", "in_pos", "raises", "limpers", "last_raiser",
                 "pf_aggressor", "unopened", "n_alive", "hands_left", "facing_allin", "spr",
                 "avg_bb", "chip_rank", "largest_opp_total", "street_aggressor")

    def __init__(self, o):
        self.o = o
        me = self.me = o["seat"]
        n = self.n = o["num_seats"]
        self.si = o["street_index"]
        self.hole = o["hole"]
        self.board = o["board"]
        st = self.stacks = o["stacks"]
        bets = self.bets = o["bets"]
        self.pot = o["pot"]
        self.to_call = o["to_call"]
        bb = self.bb = max(1, o["bb"])
        self.my_stack = st[me]
        self.my_bet = bets[me]
        self.my_total = st[me] + bets[me]
        in_hand = o["in_hand"]
        alive = o["alive"]
        self.opps = [i for i in range(n) if in_hand[i] and i != me]
        self.n_opp = len(self.opps)
        opp_tot = [st[i] + bets[i] for i in self.opps]
        self.largest_opp_total = max(opp_tot) if opp_tot else 0
        self.eff = min(self.my_total, self.largest_opp_total)
        self.eff_bb = self.eff / float(bb)
        self.my_bb = self.my_total / float(bb)
        self.pot_odds = self.to_call / float(self.pot + self.to_call) if self.to_call else 0.0
        self.n_alive = sum(1 for a in alive if a)
        # action order this street
        if self.si == 0:
            start = o["bb_seat"]
        else:
            start = o["button"]
        order = []
        for k in range(1, n + 1):
            j = (start + k) % n
            if alive[j]:
                order.append(j)
        idx = order.index(me) if me in order else 0
        after = order[idx + 1:]
        self.n_behind = sum(1 for j in after if in_hand[j] and st[j] > 0)
        self.pos = idx / float(max(1, len(order) - 1))  # 0 = first to act, 1 = last
        self.in_pos = self.n_behind == 0
        # betting history
        raises = 0
        limpers = 0
        last_raiser = -1
        pf_aggr = -1
        st_aggr = -1
        for si, seat, kind, amt in o["history"]:
            if si == 0 and kind == RAISE:
                pf_aggr = seat
            if si == self.si:
                if kind == RAISE:
                    raises += 1
                    last_raiser = seat
                    st_aggr = seat
                elif kind == CALL and raises == 0 and si == 0:
                    limpers += 1
        self.raises = raises
        self.limpers = limpers
        self.last_raiser = last_raiser
        self.pf_aggressor = pf_aggr
        self.street_aggressor = st_aggr
        self.unopened = raises == 0
        self.hands_left = o["max_hands"] - o["hand_no"] + 1
        self.facing_allin = (last_raiser >= 0 and st[last_raiser] == 0)
        self.spr = (self.eff - self.my_bet) / float(max(1, self.pot))
        tot = [st[i] + bets[i] for i in range(n) if alive[i]]
        self.avg_bb = sum(tot) / float(len(tot)) / bb if tot else 0.0
        self.chip_rank = sum(1 for t in tot if t > self.my_total)  # 0 = chip leader


# ----------------------------------------------------------------- params
COMMON_PARAMS = {
    # name: (default, lo, hi)
    "open_pct": (0.18, 0.04, 0.6),          # base open range (top fraction), first position
    "pos_widen": (0.22, 0.0, 0.5),          # added open range at the button
    "steal_widen": (0.10, 0.0, 0.4),        # extra range when folded to in BTN/SB
    "limp_rate": (0.0, 0.0, 1.0),           # prob. to limp instead of folding marginal hands
    "open_size": (2.5, 2.0, 4.0),           # open size in BB (+1 per limper)
    "threebet_pct": (0.05, 0.0, 0.2),       # value 3-bet range
    "resteal_rate": (0.08, 0.0, 0.6),       # light 3-bet vs late-position opens
    "threebet_mult": (3.0, 2.2, 4.5),
    "fourbet_pct": (0.025, 0.0, 0.1),
    "call_margin": (0.02, -0.1, 0.15),      # extra equity over pot odds to call
    "value_eq": (0.62, 0.45, 0.85),         # equity to bet/raise for value
    "raise_eq": (0.75, 0.55, 0.95),         # equity to raise a bet
    "value_size": (0.66, 0.25, 1.5),        # pot fraction
    "bluff_size": (0.5, 0.25, 1.2),
    "bluff_rate": (0.12, 0.0, 0.6),         # bluff frequency with air (heads-up-ish)
    "semibluff_rate": (0.35, 0.0, 1.0),
    "cbet_rate": (0.6, 0.0, 1.0),
    "bluff_raise_rate": (0.04, 0.0, 0.3),
    "implied": (0.04, 0.0, 0.15),           # implied-odds credit for draws
    "shove_bb": (10.0, 0.0, 25.0),          # push/fold mode below this effective stack
    "push_widen": (1.0, 0.5, 2.0),          # scale push ranges
    "call_shove_margin": (0.0, -0.1, 0.15),
    "commit_spr": (1.2, 0.3, 3.0),          # shove instead of betting below this SPR
    "icm_risk": (1.0, 0.0, 2.5),            # weight on ICM risk premium
    "bigstack_pressure": (0.3, 0.0, 1.0),   # widen aggression when covering opponents
    "multiway_tighten": (0.3, 0.0, 1.0),
    "adapt": (0.5, 0.0, 1.0),               # opponent-model weight
    "abstraction": (2, 0, 2),               # postflop table resolution
    "mix": (0.0, 0.0, 0.3),                 # random action noise
    "endgame_push": (0.0, 0.0, 1.0),        # late-hand-cap aggression / caution shift
    # EV-model calibration (equity / hybrid / lookup fallback)
    "fold_prior": (0.35, 0.05, 0.7),        # prior P(fold) to a bet, per opponent
    "fold_slope": (0.18, 0.0, 0.4),         # extra fold prob per pot-sized bet
    "called_penalty": (0.06, 0.0, 0.2),     # equity lost when a bet gets called
    "realize": (0.85, 0.5, 1.0),            # equity realisation when checking OOP
}

STYLE_PRESETS = {
    "tight": {"open_pct": 0.11, "pos_widen": 0.14, "call_margin": 0.06, "threebet_pct": 0.035},
    "loose": {"open_pct": 0.30, "pos_widen": 0.30, "call_margin": -0.03, "threebet_pct": 0.08},
    "passive": {"value_eq": 0.75, "raise_eq": 0.88, "bluff_rate": 0.02, "semibluff_rate": 0.05,
                "cbet_rate": 0.25, "limp_rate": 0.6, "threebet_pct": 0.02, "resteal_rate": 0.0},
    "aggressive": {"value_eq": 0.56, "raise_eq": 0.68, "bluff_rate": 0.25, "semibluff_rate": 0.6,
                   "cbet_rate": 0.8, "value_size": 0.85, "bluff_raise_rate": 0.1},
    "bluffy": {"bluff_rate": 0.4, "semibluff_rate": 0.8, "bluff_raise_rate": 0.15, "cbet_rate": 0.85},
    "value": {"bluff_rate": 0.02, "semibluff_rate": 0.15, "value_eq": 0.58, "value_size": 0.8,
              "cbet_rate": 0.4},
    "stealer": {"steal_widen": 0.3, "pos_widen": 0.35},
    "restealer": {"resteal_rate": 0.4, "threebet_pct": 0.07},
    "shover": {"shove_bb": 18.0, "push_widen": 1.3},
    "icm_survival": {"icm_risk": 2.0, "call_shove_margin": 0.06, "bigstack_pressure": 0.1},
    "bigstack_bully": {"bigstack_pressure": 0.9, "steal_widen": 0.25, "icm_risk": 1.2},
    "exploitative": {"adapt": 1.0},
    "static": {"adapt": 0.0},
}


def resolve_params(spec_params=None, styles=None, defaults=None):
    p = {k: v[0] for k, v in COMMON_PARAMS.items()}
    if defaults:
        p.update(defaults)
    for s in styles or []:
        p.update(STYLE_PRESETS[s])
    if spec_params:
        p.update(spec_params)
    return p


class BaseBot(Agent):
    """Common plumbing: params, RNG, opponent model, features, ICM cache."""

    name = "base"
    DEFAULTS = {}

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = resolve_params(params, styles, self.DEFAULTS)
        self.rng = random.Random(seed)
        self.f = features_for(variant)
        self.model = None
        self._icm_cache = {}
        self._last_stacks = None
        self._hand = -1

    # opponent model is created lazily (seat count known on first act)
    def _ensure(self, o):
        if self.model is None:
            self.model = OppModel(o["num_seats"])
        if o["hand_no"] != self._hand:
            self._hand = o["hand_no"]
            self._icm_cache = {}
            self._hand_start = [s + b for s, b in zip(o["stacks"], o["contrib"])]

    def on_hand_end(self, info):
        if self.model is not None:
            self.model.update(info, getattr(self, "_hand_start", None))

    # ------------------------------------------------------------- helpers
    def act(self, o):
        self._ensure(o)
        c = Ctx(o)
        a = self.decide(c)
        return self.finish(c, a)

    def decide(self, c):
        raise NotImplementedError

    def finish(self, c, a):
        o = c.o
        kind, amt = a
        if kind == RAISE:
            if not o["can_raise"]:
                return (CALL, 0) if c.to_call else (CHECK, 0)
            amt = int(max(o["min_raise_to"], min(o["max_raise_to"], amt)))
            # raising almost everything -> just shove
            if amt >= c.my_total * 0.8:
                amt = o["max_raise_to"]
            return (RAISE, amt)
        if kind == FOLD and c.to_call == 0:
            return (CHECK, 0)
        if kind == CALL and c.to_call == 0:
            return (CHECK, 0)
        return (kind, 0)

    def bet_amount(self, c, frac):
        """raise-to amount for a pot-fraction bet/raise."""
        cur = c.o["current_bet"]
        pot_after_call = c.pot + c.to_call
        return cur + max(c.bb, int(frac * pot_after_call))

    def shove(self, c):
        return (RAISE, c.o["max_raise_to"])

    def opp_width(self, seat, default):
        """Estimated range width of an aggressor, blended by `adapt`."""
        if self.model is None or seat < 0:
            return default
        pfr = self.model.rate(seat, "pfr")
        a = self.p["adapt"]
        return (1 - a) * default + a * max(0.04, min(1.0, pfr * 1.1))

    def fold_prob(self, c, bet_frac):
        """Chance all opponents fold to a bet of bet_frac * pot."""
        base = self.p["fold_prior"] + self.p["fold_slope"] * min(bet_frac, 1.5)
        a = self.p["adapt"]
        f = 1.0
        for s in c.opps:
            obs = self.model.rate(s, "fold_post" if c.si > 0 else "fold_to_bet") if self.model else base
            fs = (1 - a) * base + a * (obs * (0.7 + 0.3 * min(bet_frac, 1.5)) / 0.85)
            f *= max(0.02, min(0.95, fs))
        return f

    def icm_premium(self, c, risk_amt):
        """Extra equity needed (vs chip-EV) to risk `risk_amt` more chips,
        approximating the main opponent as the largest stack in the pot."""
        payouts = c.o["payouts"]
        if self.p["icm_risk"] <= 0 or risk_amt <= 0 or not c.opps:
            return 0.0
        key = (risk_amt, c.pot)
        v = self._icm_cache.get(key)
        if v is not None:
            return v
        me = c.me
        tot = list(c.stacks)
        vil = max(c.opps, key=lambda i: c.stacks[i] + c.bets[i])
        r = min(risk_amt, c.my_stack)
        m = max(0, min(r - c.to_call, c.stacks[vil]))
        r_eff = min(r, c.to_call + m)
        big_pot = c.pot + r_eff + m
        fold = list(tot)
        fold[vil] += c.pot
        win = list(tot)
        win[me] += big_pot - r_eff
        win[vil] -= m
        lose = list(tot)
        lose[me] -= r_eff
        lose[vil] += big_pot - m
        ef = hero_icm(fold, me, payouts)
        ew = hero_icm(win, me, payouts)
        el = hero_icm(lose, me, payouts)
        if ew - el <= 1e-9 or big_pot <= 0:
            v = 0.0
        else:
            req_icm = (ef - el) / (ew - el)
            req_chip = r_eff / float(big_pot)
            v = max(0.0, req_icm - req_chip)
        self._icm_cache[key] = v
        return v
