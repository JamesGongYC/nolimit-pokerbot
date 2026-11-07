import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from pokerbot import sim
from pokerbot.agent_api import Agent, CHECK, FOLD, CALL
from pokerbot.engine import play_table
from pokerbot.event import Event
from pokerbot import strategies
from tests.helpers import event

ALLOWED_OBS_KEYS = {
    "seat", "num_seats", "hand_no", "max_hands", "hands_per_level", "level", "sb", "bb", "ante",
    "button", "sb_seat", "bb_seat", "street", "street_index", "hole", "board", "stacks", "bets",
    "contrib", "pot", "in_hand", "alive", "to_call", "min_raise_to", "max_raise_to", "can_check",
    "can_raise", "history", "payouts", "current_bet"}

PROBE_LOG = []


class Probe(Agent):
    instances = 0

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        Probe.instances += 1
        self.id = Probe.instances
        self.hands = []

    def act(self, o):
        PROBE_LOG.append((self.id, o["hand_no"]))
        self.hands.append(o["hand_no"])
        assert self.hands == sorted(self.hands), "agent saw hands from another table"
        return (CALL, 0) if not o["can_check"] else (CHECK, 0)


class Peeker(Agent):
    """Asserts that observations contain no hidden information."""

    def __init__(self, test):
        self.t = test
        self.holes = None

    def act(self, o):
        self.t.assertLessEqual(set(o), ALLOWED_OBS_KEYS)
        for h in o["history"]:
            self.t.assertEqual(len(h), 4)
            self.t.assertIsInstance(h[3], int)
        return (CALL, 0) if not o["can_check"] else (CHECK, 0)

    def on_hand_end(self, info):
        # showdown reveals only hands that reached showdown
        in_sd = set(info["showdown"])
        folded = {h[1] for h in info["history"] if h[2] == FOLD}
        self.t.assertFalse(in_sd & folded)


class TestIsolation(unittest.TestCase):
    def test_fresh_agent_per_table(self):
        strategies.REGISTRY["probe"] = Probe
        try:
            sim._init_worker(Event.load().cfg)
            del PROBE_LOG[:]
            before = Probe.instances
            for seed in range(3):
                sim.run_table_job(([{"type": "probe"}] * 8, seed, None))
            self.assertEqual(Probe.instances - before, 24)
            ids = {}
            for i, h in PROBE_LOG:
                ids.setdefault(i, []).append(h)
            for i, hs in ids.items():
                self.assertEqual(hs, sorted(hs))  # each instance lives within one table
        finally:
            del strategies.REGISTRY["probe"]

    def test_observations_have_no_hidden_info(self):
        ev = event()
        agents = [Peeker(self) for _ in range(8)]
        play_table(ev, agents, 3)

    def test_obs_is_a_copy(self):
        """Mutating the observation must not change engine state."""
        class Vandal(Agent):
            def act(self, o):
                o["stacks"][0] = 10 ** 9
                o["bets"][:] = [0] * len(o["bets"])
                o["history"].append((9, 9, "raise", 10 ** 9))
                return (CALL, 0)
        ev = event()
        r = play_table(ev, [Vandal() for _ in range(8)], 4)
        self.assertEqual(sum(r.final_stacks), 16000)


class TestTournamentFormat(unittest.TestCase):
    def test_fillers_and_assignment(self):
        ev = event(max_hands=20)
        entries = [{"type": "station", "name": "s%d" % i} for i in range(10)]
        with sim.Runner(ev, workers=1) as r:
            out = sim.tournament(r, entries, iterations=6, seed=1)
        # 10 entries -> 2 tables, 6 filler seats; every entry scored once per iteration
        for it in range(6):
            pts = [out["points"][e][it] for e in range(10)]
            self.assertLessEqual(sum(pts), 2 * sum(ev.payouts) + 1e-6)
            places = [out["places"][e][it] for e in range(10)]
            self.assertTrue(all(p >= 1 for p in places))

    def test_filler_rules(self):
        """148 entries -> 18 full tables + 1 table of 4 scored + 4 fillers not
        seated at it; a field smaller than a table plays short."""
        class Capture(object):
            def __init__(self):
                self.jobs = []

            def run(self, jobs):
                self.jobs += jobs
                return [{"points": [0.0] * len(j[0]), "places": [1] * len(j[0]),
                         "stats": [sim.SeatStats().as_dict()] * len(j[0])} for j in jobs]

        entries = [{"type": "station", "name": "e%d" % i} for i in range(148)]
        cap = Capture()
        sim.tournament(cap, entries, iterations=3, seed=2)
        self.assertEqual(len(cap.jobs), 3 * 19)
        for it in range(3):
            tabs = [[s["name"] for s in j[0]] for j in cap.jobs[it * 19:(it + 1) * 19]]
            self.assertTrue(all(len(t) == 8 for t in tabs))
            self.assertTrue(all(len(set(t)) == 8 for t in tabs))  # no bot twice at a table
            names = [x for t in tabs for x in t]
            self.assertEqual(len(set(names)), 148)  # everyone seated; fillers are repeats
        cap = Capture()
        sim.tournament(cap, entries[:5], iterations=2, seed=3)
        self.assertTrue(all(len(j[0]) == 5 for j in cap.jobs))


class TestExport(unittest.TestCase):
    def test_export_and_run(self):
        """Upload format: agent.py at the root of the tree and of the zip, under
        the size limit, no forbidden imports, and the zip runs as unpacked by a
        harness from an unrelated working directory."""
        import zipfile
        from pokerbot.export import export, static_checks
        d = tempfile.mkdtemp()
        try:
            out = os.path.join(d, "sub")
            zp = os.path.join(d, "upload.zip")
            chk = export("hybrid_default", out, log=lambda *a: None, zip_path=zp, limit_mb=50)
            self.assertTrue(chk["ok"])
            self.assertEqual(chk["forbidden"], [])
            self.assertTrue(chk["agent_at_root"])
            self.assertLess(chk["zip_mb"], 50)
            names = zipfile.ZipFile(zp).namelist()
            self.assertIn("agent.py", names)
            self.assertFalse(any("__pycache__" in n or n.endswith(".pyc") for n in names))
            self.assertFalse(any(n.startswith("sub/") for n in names))  # no wrapping folder
            self.assertFalse(any(n.startswith("pokerbot/") or n == "bot.py" for n in names))  # no shadowing names
            unz = os.path.join(d, "unzipped")
            zipfile.ZipFile(zp).extractall(unz)
            other = os.path.join(d, "elsewhere")
            os.makedirs(other)
            code = ("import importlib.util, sys\n"
                    "sp = importlib.util.spec_from_file_location('agent', sys.argv[1])\n"
                    "m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)\n"
                    "from jj_nolimit_bot.event import Event\nfrom jj_nolimit_bot.engine import play_table\n"
                    "r = play_table(Event.load(), [m.Agent() for _ in range(8)], 2)\n"
                    "print(sum(r.final_stacks), sum(s.errors + s.illegal for s in r.stats))")
            res = subprocess.check_output([sys.executable, "-c", code, os.path.join(unz, "agent.py")],
                                          cwd=other).decode().split()
            self.assertEqual(res, ["16000", "0"])
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
