import itertools
import random
import unittest

from pokerbot.scoring import icm, bust_points
from pokerbot.strategies.base import hero_icm


def brute_icm(stacks, payouts):
    n = len(stacks)
    eq = [0.0] * n
    idx = [i for i in range(n) if stacks[i] > 0]
    for perm in itertools.permutations(idx):
        p = 1.0
        rem = float(sum(stacks))
        for i in perm:
            p *= stacks[i] / rem
            rem -= stacks[i]
        for place, i in enumerate(perm):
            if place < len(payouts):
                eq[i] += p * payouts[place]
    return eq


class TestICM(unittest.TestCase):
    def test_vs_bruteforce(self):
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(2, 7)
            stacks = [rng.randint(0, 3000) for _ in range(n)]
            stacks[0] = max(stacks[0], 1)
            stacks[1] = max(stacks[1], 1)
            pays = sorted([rng.randint(0, 50) for _ in range(rng.randint(1, n))], reverse=True)
            a, b = icm(stacks, pays), brute_icm(stacks, pays)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=9)

    def test_sums_and_limits(self):
        pays = [50, 30, 20]
        eq = icm([5000, 5000, 3000, 3000], pays)
        self.assertAlmostEqual(sum(eq), 100)
        self.assertAlmostEqual(eq[0], eq[1])
        self.assertAlmostEqual(icm([100, 0, 0], pays)[0], 50)  # only survivor's place 1 paid
        self.assertEqual(icm([0, 100], pays)[0], 0.0)

    def test_hero_icm_matches(self):
        rng = random.Random(9)
        for _ in range(60):
            n = rng.randint(2, 8)
            stacks = [rng.randint(1, 4000) for _ in range(n)]
            k = rng.randint(1, 4)
            pays = sorted([rng.randint(1, 50) for _ in range(k)], reverse=True)
            full = icm(stacks, pays)
            for me in range(n):
                self.assertAlmostEqual(hero_icm(stacks, me, pays), full[me], places=9)


class TestBustPoints(unittest.TestCase):
    def test_stack_rule(self):
        pays = [50, 30, 20, 0, 0, 0, 0, 0]
        out = bust_points([3, 5], [0, 0, 0, 300, 0, 100, 0, 0], 1, pays, "stack")
        self.assertEqual(out[3], (2, 30))
        self.assertEqual(out[5], (3, 20))

    def test_ties_split(self):
        pays = [50, 30, 20]
        out = bust_points([0, 1], [200, 200, 0], 1, pays, "stack")
        self.assertEqual(out[0], (2, 25.0))
        self.assertEqual(out[1], (2, 25.0))
        out = bust_points([0, 1], [300, 200, 0], 1, pays, "split")
        self.assertEqual(out[0][1], 25.0)


if __name__ == "__main__":
    unittest.main()
