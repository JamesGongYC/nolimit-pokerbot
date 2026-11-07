import random
import unittest
from itertools import combinations

from pokerbot.cards import STANDARD, cards, hand_class, class_name, class_combos
from pokerbot.variant import Variant


def naive5(cs, wheel_low=0, ranks=range(13), flush_over_fh=False):
    rs = sorted([c >> 2 for c in cs], reverse=True)
    suits = [c & 3 for c in cs]
    flush = len(set(suits)) == 1
    uniq = sorted(set(rs), reverse=True)
    lo = sorted(ranks)
    straight = -1
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight = uniq[0]
        elif set(uniq) == set([12] + lo[:4]):
            straight = lo[3]
    cnt = {}
    for r in rs:
        cnt[r] = cnt.get(r, 0) + 1
    groups = sorted(cnt.items(), key=lambda x: (-x[1], -x[0]))
    shape = [g[1] for g in groups]
    order = [g[0] for g in groups]
    if straight >= 0 and flush:
        return (8, [straight])
    if shape[0] == 4:
        return (7, order)
    if shape[:2] == [3, 2]:
        return (6, order)
    if flush:
        return (5, rs)
    if straight >= 0:
        return (4, [straight])
    if shape[0] == 3:
        return (3, order)
    if shape[:2] == [2, 2]:
        return (2, order)
    if shape[0] == 2:
        return (1, order)
    return (0, rs)


def naive7(cs):
    return max(naive5(c) for c in combinations(cs, 5))


class TestEvaluator(unittest.TestCase):
    def test_random_vs_bruteforce(self):
        rng = random.Random(7)
        for _ in range(3000):
            a = rng.sample(range(52), 7)
            b = rng.sample(range(52), 7)
            fa, fb = STANDARD.evaluate(a), STANDARD.evaluate(b)
            na, nb = naive7(a), naive7(b)
            self.assertEqual(na[0], STANDARD.category(fa), (a, na))
            self.assertEqual((fa > fb) - (fa < fb), (na > nb) - (na < nb), (a, b))

    def test_known(self):
        e = STANDARD.evaluate
        self.assertGreater(e(cards("AsKsQsJsTs2c3d")), e(cards("9s8s7s6s5s2c3d")))
        self.assertEqual(STANDARD.category(e(cards("Ac2d3h4s5c9dKh"))), 4)  # wheel
        self.assertGreater(e(cards("6c2d3h4s5c9dKh")), e(cards("Ac2d3h4s5c9dKh")))
        self.assertEqual(e(cards("AcAd9h9s7c7d2h")), e(cards("AcAd9h9s7c7d3h")))  # kicker beyond 5 ignored

    def test_short_deck(self):
        v = Variant.from_config({"name": "short_deck"})
        ev = v.evaluator.evaluate
        self.assertGreater(ev(cards("AsKs9s7s6sTdJd")), ev(cards("AsAdAhKsKcTdJd")))
        self.assertEqual(v.evaluator.category(ev(cards("As6d7h8s9cKdKh"))), 4)
        self.assertEqual(len(v.deck), 36)

    def test_classes(self):
        seen = set()
        for a in range(52):
            for b in range(a + 1, 52):
                seen.add(hand_class(a, b))
        self.assertEqual(len(seen), 169)
        self.assertEqual(sum(len(class_combos(c)) for c in range(169)), 1326)
        self.assertEqual(class_name(hand_class(*cards("AsKs"))), "AKs")
        self.assertEqual(class_name(hand_class(*cards("7c2d"))), "72o")


if __name__ == "__main__":
    unittest.main()
