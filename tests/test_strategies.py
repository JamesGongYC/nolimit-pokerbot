import unittest

from pokerbot.engine import play_table
from pokerbot.event import Event
from pokerbot.sim import latency_pct
from pokerbot.strategies import make_agent, presets, REGISTRY
from pokerbot.strategies.treebot import compile_ensemble


class TestAllStrategies(unittest.TestCase):
    def test_every_preset_plays_legally_and_fast(self):
        ev = Event.load()
        names = sorted(presets())
        for k, name in enumerate(names):
            agents = [make_agent(name, 1)] + [make_agent(x, 10 + i) for i, x in
                                               enumerate(["heur_tag", "station", "maniac", "nit", "eq_tag",
                                                          "shover20", "random"])]
            r = play_table(ev, agents, 100 + k)
            st = r.stats[0]
            self.assertEqual(st.errors, 0, name)
            self.assertEqual(st.illegal, 0, name)
            self.assertLess(latency_pct(st, 0.99), 1000, name)  # p99 < 1 ms
            self.assertLess(st.max_ns, 5.0e6, name)  # max: allow OS-scheduling spikes under load
            self.assertEqual(sum(r.final_stacks), 16000)

    def test_registry_types_covered(self):
        types = {p.get("type") for p in presets().values()}
        for t in REGISTRY:
            if t not in ("collector", "probe", "tree", "mixture"):
                self.assertIn(t, types)


class TestTreeCompiler(unittest.TestCase):
    def test_compiled_tree_matches_manual(self):
        # x[0] <= 0.5 ? (x[1] <= 2 ? 1 : 2) : 3, two trees, base 0.5
        t = [[0, 0.5, 1, 4, False, 0], [1, 2.0, 2, 3, False, 0], [0, 0, 0, 0, True, 1.0],
             [0, 0, 0, 0, True, 2.0], [0, 0, 0, 0, True, 3.0]]
        ns = {}
        exec(compile_ensemble({"base": 0.5, "trees": [t, t]}, "q"), ns)
        self.assertEqual(ns["q"]([0.0, 1.0]), 2.5)
        self.assertEqual(ns["q"]([0.0, 5.0]), 4.5)
        self.assertEqual(ns["q"]([1.0, 0.0]), 6.5)
        ns = {}
        exec(compile_ensemble({"base": 0.0, "trees": [t, t], "mean": True}, "q"), ns)
        self.assertEqual(ns["q"]([1.0, 0.0]), 3.0)


if __name__ == "__main__":
    unittest.main()


class TestTopCycle(unittest.TestCase):
    def test_smith_set(self):
        from pokerbot.evolve import top_cycle
        # a>b>c>a cycle, all beat d
        wins = {("a", "b"), ("b", "c"), ("c", "a"), ("a", "d"), ("b", "d"), ("c", "d")}
        beats = lambda x, y: (x, y) in wins
        self.assertEqual(top_cycle(["a", "b", "c", "d"], beats), ["a", "b", "c"])
        # transitive: single winner
        wins = {("a", "b"), ("a", "c"), ("b", "c")}
        self.assertEqual(top_cycle(["c", "b", "a"], lambda x, y: (x, y) in wins), ["a"])
