"""Replaceable poker-variant adapter.

The official variant is announced at the event. Everything variant-specific
(deck, hole cards, streets, betting structure, hand ranking) lives here so a
new variant is one JSON block in the event config, e.g.:

    "variant": {"name": "short_deck", "deck_ranks": [4,5,6,7,8,9,10,11,12],
                "category_order": "short_deck"}
"""
from .cards import Evaluator, STANDARD_ORDER, FLUSH, FULL_HOUSE, TRIPS, STRAIGHT

# Short deck (6+): flush beats full house; trips beat straight (Triton rules).
SHORT_DECK_ORDER = list(STANDARD_ORDER)
SHORT_DECK_ORDER[FLUSH], SHORT_DECK_ORDER[FULL_HOUSE] = STANDARD_ORDER[FULL_HOUSE], STANDARD_ORDER[FLUSH]
SHORT_DECK_ORDER[TRIPS], SHORT_DECK_ORDER[STRAIGHT] = STANDARD_ORDER[STRAIGHT], STANDARD_ORDER[TRIPS]

NAMED_ORDERS = {"standard": STANDARD_ORDER, "short_deck": SHORT_DECK_ORDER}

PRESETS = {
    "nlhe": {},
    "plhe": {"betting": "pot_limit"},
    "flhe": {"betting": "fixed_limit"},
    "short_deck": {"deck_ranks": list(range(4, 13)), "category_order": "short_deck"},
    "plo4": {"hole_cards": 4, "hole_use": 2, "betting": "pot_limit"},
}


class Variant(object):
    def __init__(self, name="nlhe", deck_ranks=None, hole_cards=2, hole_use="any",
                 streets=None, betting="no_limit", category_order="standard",
                 burn=True, raise_cap=None, **_ignored):
        self.name = name
        self.deck_ranks = list(deck_ranks if deck_ranks is not None else range(13))
        self.deck = [r * 4 + s for r in self.deck_ranks for s in range(4)]
        self.hole_cards = hole_cards
        self.hole_use = hole_use  # "any" or an int (exact number of hole cards used)
        self.streets = streets or [["preflop", 0], ["flop", 3], ["turn", 1], ["river", 1]]
        self.betting = betting  # no_limit | pot_limit | fixed_limit
        self.raise_cap = raise_cap if raise_cap is not None else (4 if betting == "fixed_limit" else None)
        self.burn = burn
        order = NAMED_ORDERS[category_order] if isinstance(category_order, str) else category_order
        self.evaluator = Evaluator(self.deck_ranks, order)
        self.is_holdem_like = hole_cards == 2 and hole_use == "any" and len(self.deck_ranks) == 13

    def hand_value(self, hole, board):
        if self.hole_use == "any":
            return self.evaluator.evaluate(list(hole) + list(board))
        return self.evaluator.evaluate_split(tuple(hole), tuple(board), self.hole_use)

    @classmethod
    def from_config(cls, cfg):
        cfg = dict(cfg or {})
        preset = PRESETS.get(cfg.get("name", "nlhe"), {})
        merged = dict(preset)
        merged.update(cfg)
        return cls(**merged)
