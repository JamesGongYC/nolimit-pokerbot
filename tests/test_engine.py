import random
import unittest

from pokerbot.agent_api import Agent, FOLD, CHECK, CALL, RAISE
from pokerbot.cards import cards
from pokerbot.engine import Table, play_table
from tests.helpers import Scripted, allin, make_deck, event

ROYAL = cards("AsKsQsJsTs")
RAGS = cards("2h7d9c8sJh")


def run(ev, agents, stacks, button, holes_by_seat, board, max_hands=1):
    """holes_by_seat: dict seat -> cards; dealt starting left of button."""
    n = len(agents)
    order = [(button + k) % n for k in range(1, n + 1) if stacks[(button + k) % n] > 0]
    deck = make_deck([holes_by_seat[s] for s in order], board)
    ev = ev.with_overrides(max_hands=max_hands)
    t = Table(ev, agents, seed=1, deck_fn=lambda h: deck, stacks=stacks, button=button)
    return t, t.run()


class TestSidePots(unittest.TestCase):
    def test_three_way_allin(self):
        ev = event()
        agents = [Scripted([allin]), Scripted([allin]), Scripted([(CALL, 0)])]
        holes = {0: cards("AcAd"), 1: cards("KcKd"), 2: cards("QcQd")}
        t, r = run(ev, agents, [100, 300, 500], 0, holes, RAGS)
        self.assertEqual(t.stacks, [300, 400, 200])

    def test_uncalled_excess_returned(self):
        ev = event()
        agents = [Scripted([allin]), Scripted([(CALL, 0), (CALL, 0)]), Scripted([allin])]
        holes = {0: cards("AcAd"), 1: cards("KcKd"), 2: cards("QcQd")}
        t, r = run(ev, agents, [100, 300, 500], 0, holes, RAGS)
        # seat2 shoves 500, seat1 calls 300 all-in; 200 excess returns to seat2
        self.assertEqual(t.stacks, [300, 400, 200])
        self.assertEqual(sum(t.stacks), 900)

    def test_best_hand_short_stack_side_pot_to_second(self):
        ev = event()
        agents = [Scripted([allin]), Scripted([allin]), Scripted([(CALL, 0)])]
        holes = {0: cards("QcQd"), 1: cards("AcAd"), 2: cards("KcKd")}
        t, r = run(ev, agents, [100, 300, 500], 0, holes, RAGS)
        self.assertEqual(t.stacks, [0, 700, 200])

    def test_odd_chip_left_of_button(self):
        ev = event(blind_levels=[[5, 10, 1]])
        agents = [Scripted([(FOLD, 0)]), Scripted([(CALL, 0)]), Scripted([])]
        holes = {0: cards("2c3d"), 1: cards("4c5d"), 2: cards("6c7d")}
        t, r = run(ev, agents, [100, 100, 100], 0, holes, ROYAL)
        # pot = 3 antes + 10 + 10 = 23 split between seats 1 and 2; seat 1 gets the odd chip
        self.assertEqual(t.stacks[0], 99)
        self.assertEqual(t.stacks[1] - 100, 1)   # paid 11, got 12
        self.assertEqual(t.stacks[2] - 100, 0)   # paid 11, got 11


class TestBettingRules(unittest.TestCase):
    def test_min_raise_and_incomplete_raise(self):
        ev = event()
        s0 = Scripted([(RAISE, 100), lambda o: (CALL, 0)])
        s1 = Scripted([(CALL, 0), lambda o: (CALL, 0)])
        s2 = Scripted([allin])
        holes = {0: cards("AcAd"), 1: cards("KcKd"), 2: cards("QcQd")}
        t, r = run(ev, [s0, s1, s2], [1000, 1000, 130], 0, holes, RAGS)
        o_first = s0.obs[0]
        self.assertEqual(o_first["min_raise_to"], 40)  # BB 20 + raise of 20
        o_s1 = s1.obs[0]
        self.assertEqual(o_s1["min_raise_to"], 180)  # 100 + last raise 80
        # s2 all-in for 130 is an incomplete raise -> s0, s1 may only call/fold
        back = s0.obs[1]
        self.assertEqual(back["to_call"], 30)
        self.assertFalse(back["can_raise"])

    def test_full_raise_reopens(self):
        ev = event()
        s0 = Scripted([(RAISE, 100), lambda o: (CALL, 0)])
        s1 = Scripted([(CALL, 0), lambda o: (CALL, 0)])
        s2 = Scripted([allin])
        holes = {0: cards("AcAd"), 1: cards("KcKd"), 2: cards("QcQd")}
        t, r = run(ev, [s0, s1, s2], [1000, 1000, 300], 0, holes, RAGS)
        self.assertTrue(s0.obs[1]["can_raise"])

    def test_heads_up_order(self):
        ev = event()
        a, b = Scripted([(CALL, 0)]), Scripted([(CHECK, 0)])
        holes = {0: cards("AcAd"), 1: cards("KcKd")}
        t, r = run(ev, [a, b], [500, 500], 0, holes, RAGS)
        o = a.obs[0]
        self.assertEqual((o["sb_seat"], o["bb_seat"], o["street"]), (0, 1, "preflop"))
        self.assertEqual(o["to_call"], 10)
        flop_first = b.obs[1] if len(b.obs) > 1 else None
        self.assertIsNotNone(flop_first)
        self.assertEqual(flop_first["street"], "flop")
        self.assertEqual(len(a.obs), 4)  # preflop + flop/turn/river after b checks

    def test_illegal_actions_fallback(self):
        ev = event()
        bad = Scripted([(RAISE, 5)])      # below min raise -> fallback fold
        good = Scripted([(CHECK, 0)])
        holes = {0: cards("AcAd"), 1: cards("KcKd")}
        t, r = run(ev, [bad, good], [500, 500], 0, holes, RAGS)
        self.assertEqual(r.stats[0].illegal, 1)
        self.assertEqual(t.stacks, [490, 510])


class Chaos(Agent):
    def __init__(self, seed):
        self.r = random.Random(seed)

    def act(self, o):
        x = self.r.random()
        if x < 0.05:
            raise RuntimeError("boom")
        if x < 0.1:
            return ("raise", -5)
        if x < 0.15:
            return "garbage"
        if x < 0.35 and o["can_raise"]:
            return (RAISE, self.r.randint(o["min_raise_to"], o["max_raise_to"]))
        if x < 0.4:
            return (RAISE, o["max_raise_to"] + 1000)
        if x < 0.75:
            return (CALL, 0)
        return (FOLD, 0)


class Auditor(Chaos):
    """Checks every observation is internally consistent."""

    def act(self, o):
        assert o["to_call"] >= 0
        assert o["stacks"][o["seat"]] >= 0
        if o["can_raise"]:
            assert o["min_raise_to"] <= o["max_raise_to"]
            assert o["max_raise_to"] <= o["stacks"][o["seat"]] + o["bets"][o["seat"]]
            assert o["max_raise_to"] > o["current_bet"]
        assert o["can_check"] == (o["to_call"] == 0)
        assert len(o["hole"]) == 2
        return Chaos.act(self, o)


class TestFuzz(unittest.TestCase):
    def test_chip_conservation_and_points(self):
        for variant in ("nlhe", "plhe", "flhe", "short_deck"):
            ev = event(variant={"name": variant})
            errors = illegal = 0
            for seed in range(40):
                agents = [Auditor(seed * 10 + i) for i in range(8)]
                r = play_table(ev, agents, seed)
                self.assertEqual(r.chip_check_failures, 0)
                self.assertEqual(sum(r.final_stacks), 16000)
                self.assertTrue(all(s >= 0 for s in r.final_stacks))
                self.assertAlmostEqual(sum(r.points), sum(ev.payouts), places=6)
                errors += sum(s.errors for s in r.stats)
                illegal += sum(s.illegal for s in r.stats)
            self.assertGreater(errors, 0)   # exceptions were caught -> fallback
            self.assertGreater(illegal, 0)  # malformed actions were caught

    def test_bust_ordering(self):
        ev = event(payouts=[50, 30, 20])
        agents = [Scripted([allin]), Scripted([allin]), Scripted([(CALL, 0)])]
        holes = {0: cards("2c3d"), 1: cards("4c5h"), 2: cards("AcAd")}
        t, r = run(ev, agents, [100, 300, 500], 0, holes, RAGS, max_hands=5)
        self.assertEqual(r.places, [3, 2, 1])
        self.assertEqual(r.points, [20, 30, 50])

    def test_equal_stack_bust_split(self):
        ev = event(payouts=[50, 30, 20])
        agents = [Scripted([allin]), Scripted([allin]), Scripted([(CALL, 0)])]
        holes = {0: cards("2c3d"), 1: cards("4c5h"), 2: cards("AcAd")}
        t, r = run(ev, agents, [200, 200, 500], 0, holes, RAGS, max_hands=5)
        self.assertEqual(r.points, [25.0, 25.0, 50])


if __name__ == "__main__":
    unittest.main()
