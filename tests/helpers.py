from pokerbot.agent_api import Agent, FOLD, CHECK, CALL, RAISE
from pokerbot.cards import cards
from pokerbot.event import Event


def event(**kw):
    return Event.load(overrides=kw)


class Scripted(Agent):
    """Plays a fixed list of actions (then check/fold); records observations."""

    def __init__(self, actions=None):
        self.actions = list(actions or [])
        self.obs = []
        self.ends = []

    def act(self, o):
        self.obs.append(o)
        if self.actions:
            a = self.actions.pop(0)
            if callable(a):
                return a(o)
            return a
        return (CHECK, 0) if o["can_check"] else (FOLD, 0)

    def on_hand_end(self, info):
        self.ends.append(info)


def allin(o):
    """Shove; when raising is not legal (all-in for <= the call) just call."""
    return (RAISE, o["max_raise_to"]) if o["can_raise"] else (CALL, 0)


def make_deck(holes_in_deal_order, board, burn=True, hole_cards=2):
    """Deck (popped from the end) that deals the given holes then board.
    holes_in_deal_order: list of [c1, c2] for seats in dealing order."""
    seq = []
    for r in range(hole_cards):
        for h in holes_in_deal_order:
            seq.append(h[r])
    b = list(board)
    streets = [b[:3], b[3:4], b[4:5]]
    junk = [c for c in range(52) if c not in seq and c not in b]
    for st in streets:
        if burn:
            seq.append(junk.pop())
        seq.extend(st)
    rest = [c for c in range(52) if c not in seq]
    return rest + list(reversed(seq))
