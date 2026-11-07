"""Cheap population bots modelled on what quick 60-minute builds often do."""
import random

from ..agent_api import Agent, FOLD, CHECK, CALL, RAISE
from .base import features_for


class RandomBot(Agent):
    name = "random"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = dict({"raise": 0.15, "fold": 0.35}, **(params or {}))
        self.rng = random.Random(seed)

    def act(self, o):
        r = self.rng.random()
        if r < self.p["raise"] and o["can_raise"]:
            return (RAISE, self.rng.randint(o["min_raise_to"], o["max_raise_to"]))
        if r < self.p["raise"] + self.p["fold"] and not o["can_check"]:
            return (FOLD, 0)
        return (CALL, 0)


class CallingStation(Agent):
    """Calls almost anything; raises only very strong hands."""
    name = "station"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = dict({"raise_eq": 0.8, "fold_eq": 0.15}, **(params or {}))
        self.f = features_for(variant)
        self.rng = random.Random(seed)

    def act(self, o):
        k = max(1, sum(o["in_hand"]) - 1)
        if o["street_index"] == 0:
            eq = self.f.preflop_equity(o["hole"], 1) ** 0.8
        else:
            eq = self.f.post_equity(o["hole"], o["board"], 1)
        if eq > self.p["raise_eq"] and o["can_raise"]:
            return (RAISE, min(o["max_raise_to"], max(o["min_raise_to"], o["pot"])))
        if eq < self.p["fold_eq"] and o["to_call"] > o["pot"] * 0.5:
            return (FOLD, 0) if not o["can_check"] else (CHECK, 0)
        return (CALL, 0)


class Maniac(Agent):
    """Raises/shoves very often, regardless of the cards much."""
    name = "maniac"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = dict({"aggr": 0.7, "shove": 0.25, "fold": 0.1}, **(params or {}))
        self.rng = random.Random(seed)
        self.f = features_for(variant)

    def act(self, o):
        r = self.rng.random()
        pct = self.f.preflop_pct(o["hole"])
        if o["can_raise"] and r < self.p["shove"] * (1.5 - pct):
            return (RAISE, o["max_raise_to"])
        if o["can_raise"] and r < self.p["aggr"]:
            amt = o["current_bet"] + max(o["bb"], o["pot"])
            return (RAISE, max(o["min_raise_to"], min(o["max_raise_to"], amt)))
        if not o["can_check"] and r > 1 - self.p["fold"] and pct > 0.5:
            return (FOLD, 0)
        return (CALL, 0)


class Shover(Agent):
    """Push/fold only: all-in with the top X% (by percentile), else fold;
    calls shoves with the top Y%."""
    name = "shover"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = dict({"push_pct": 0.2, "call_pct": 0.1}, **(params or {}))
        self.f = features_for(variant)

    def act(self, o):
        pct = self.f.preflop_pct(o["hole"])
        if o["street_index"] > 0:
            return (CHECK, 0) if o["can_check"] else (CALL, 0)
        opened = any(h[0] == 0 and h[2] == RAISE for h in o["history"])
        if not opened and pct <= self.p["push_pct"] and o["can_raise"]:
            return (RAISE, o["max_raise_to"])
        if opened and pct <= self.p["call_pct"]:
            return (CALL, 0)
        return (CHECK, 0) if o["can_check"] else (FOLD, 0)


class Nit(Agent):
    """Plays only premium hands, raises them hard, folds the rest."""
    name = "nit"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = dict({"play_pct": 0.06}, **(params or {}))
        self.f = features_for(variant)

    def act(self, o):
        pct = self.f.preflop_pct(o["hole"])
        if o["street_index"] == 0:
            if pct <= self.p["play_pct"]:
                if o["can_raise"]:
                    return (RAISE, max(o["min_raise_to"], min(o["max_raise_to"], 3 * o["current_bet"])))
                return (CALL, 0)
            return (CHECK, 0) if o["can_check"] else (FOLD, 0)
        eq = self.f.post_equity(o["hole"], o["board"], max(1, sum(o["in_hand"]) - 1), 2)
        if eq > 0.7 and o["can_raise"]:
            return (RAISE, max(o["min_raise_to"], min(o["max_raise_to"], o["current_bet"] + o["pot"] * 2 // 3)))
        if eq > 0.45 or o["can_check"]:
            return (CALL, 0) if not o["can_check"] else (CHECK, 0)
        return (FOLD, 0)


class PotOddsBot(Agent):
    """Classic naive build: call when equity beats pot odds, raise when
    equity is high, never bluff."""
    name = "potodds"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        self.p = dict({"raise_eq": 0.7, "margin": 0.0}, **(params or {}))
        self.f = features_for(variant)

    def act(self, o):
        k = max(1, sum(o["in_hand"]) - 1)
        if o["street_index"] == 0:
            eq = self.f.preflop_equity(o["hole"], k)
        else:
            eq = self.f.post_equity(o["hole"], o["board"], k)
        odds = o["to_call"] / float(o["pot"] + o["to_call"]) if o["to_call"] else 0.0
        if eq > self.p["raise_eq"] and o["can_raise"]:
            amt = o["current_bet"] + max(o["bb"], o["pot"] * 3 // 4)
            return (RAISE, max(o["min_raise_to"], min(o["max_raise_to"], amt)))
        if o["can_check"]:
            return (CHECK, 0)
        return (CALL, 0) if eq >= odds + self.p["margin"] else (FOLD, 0)
