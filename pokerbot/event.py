"""Event configuration (structure, blinds, payouts) loaded from JSON."""
import json
import os

from .variant import Variant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EVENT = os.path.join(ROOT, "configs", "event_provisional.json")


class Event(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg.get("name", "event")
        self.table_size = int(cfg.get("table_size", 8))
        self.starting_stack = int(cfg.get("starting_stack", 2000))
        self.max_hands = int(cfg.get("max_hands", 200))
        self.hands_per_level = int(cfg.get("hands_per_level", 25))
        self.blind_levels = [tuple(int(x) for x in (lv + [0])[:3]) for lv in cfg["blind_levels"]]
        pay = list(cfg.get("payouts", [1] + [0] * (self.table_size - 1)))
        self.payouts = (pay + [0] * self.table_size)[:self.table_size]
        self.bust_tie_rule = cfg.get("bust_tie_rule", "stack")
        self.variant = Variant.from_config(cfg.get("variant", {"name": "nlhe"}))
        self.time_limit_ms = float(cfg.get("decision_time_limit_ms", 1.0))
        self.enforce_time_limit = bool(cfg.get("enforce_time_limit", True))
        self.invalid_raise_policy = cfg.get("invalid_raise_policy", "fallback")  # fallback|clamp
        self.weights = list(cfg.get("tournament_weights", [1, 2, 3, 4, 5, 6]))
        self.iterations = int(cfg.get("iterations_per_tournament", 10000))

    def level(self, hand_no):
        """Blind level for 1-based hand number; the last level repeats."""
        i = (hand_no - 1) // self.hands_per_level
        return min(i, len(self.blind_levels) - 1)

    def blinds(self, hand_no):
        return self.blind_levels[self.level(hand_no)]

    def with_overrides(self, **kw):
        cfg = dict(self.cfg)
        cfg.update(kw)
        return Event(cfg)

    @classmethod
    def load(cls, path=None, overrides=None):
        with open(path or DEFAULT_EVENT) as f:
            cfg = json.load(f)
        if overrides:
            cfg.update(overrides)
        return cls(cfg)
