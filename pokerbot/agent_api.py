"""Agent interface used by the local simulator.

PROVISIONAL: the official Agent interface ships with the starter kit. This
interface is deliberately close to what a typical kit exposes; the exporter
(`pokerbot/export.py`) generates an adapter shim, and `OfficialAdapter` below
is the single place to translate the official observation/action format.

An agent is constructed fresh for every table (state isolation) and sees only
officially observable information:

act(obs) -> (kind, amount)
    kind in {"fold", "check", "call", "raise"}; amount = total raise-to
    for this street (ignored otherwise). obs is a dict:
      seat, num_seats, hand_no, max_hands, level, hands_per_level,
      sb, bb, ante, button, sb_seat, bb_seat,
      street ("preflop"/"flop"/...), street_index, hole (list[int]),
      board (list[int]), stacks (behind, list), bets (this street),
      contrib (this hand), pot (total incl. current bets),
      in_hand (not folded), alive (started hand with chips),
      to_call, min_raise_to, max_raise_to, can_check, can_raise,
      history (list of (street_index, seat, kind, amount_to)),
      payouts (placement points)
on_hand_end(info) -> None  (optional)
    info: hand_no, stacks (after), showdown {seat: hole}, board, winners,
    history.
"""

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


class Agent(object):
    """Base class; subclasses override act()."""

    name = "agent"

    def __init__(self, params=None, seed=0):
        self.params = params or {}
        self.seed = seed

    def act(self, obs):
        return (CHECK, 0) if obs["can_check"] else (FOLD, 0)

    def on_hand_end(self, info):
        pass


def fallback_action(obs):
    return (CHECK, 0) if obs["can_check"] else (FOLD, 0)


def legalize(obs, action, policy="fallback"):
    """Return (legal_action, was_illegal). Illegal -> check/fold (official
    rule for errors), or clamp raise sizes if policy == 'clamp'."""
    try:
        kind, amt = action[0], action[1] if len(action) > 1 else 0
    except Exception:
        return fallback_action(obs), True
    if kind == FOLD:
        if obs["can_check"]:
            return (CHECK, 0), True
        return (FOLD, 0), False
    if kind == CHECK:
        if obs["can_check"]:
            return (CHECK, 0), False
        return (FOLD, 0), True
    if kind == CALL:
        if obs["to_call"] <= 0:
            return (CHECK, 0), False  # calling zero is a check
        return (CALL, 0), False
    if kind == RAISE:
        if not obs["can_raise"]:
            return fallback_action(obs), True
        try:
            amt = int(amt)
        except Exception:
            return fallback_action(obs), True
        lo, hi = obs["min_raise_to"], obs["max_raise_to"]
        if lo <= amt <= hi:
            return (RAISE, amt), False
        if policy == "clamp":
            return (RAISE, min(max(amt, lo), hi)), True
        return fallback_action(obs), True
    return fallback_action(obs), True
