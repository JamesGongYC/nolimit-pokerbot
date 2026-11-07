"""Randomised mixture: picks one component policy per table (state resets
per table, so this is a mixed strategy over tables)."""
import random

from ..agent_api import Agent


class MixtureBot(Agent):
    name = "mixture"

    def __init__(self, params=None, seed=0, styles=None, variant=None):
        from . import make_agent
        p = params or {}
        comps = p["components"]
        w = p.get("weights") or [1.0] * len(comps)
        rng = random.Random(seed)
        pick = rng.choices(range(len(comps)), w)[0]
        self.inner = make_agent(comps[pick], seed=seed + 1, variant=variant)
        self.experience = getattr(self.inner, "experience", None)

    def act(self, obs):
        return self.inner.act(obs)

    def on_hand_end(self, info):
        self.inner.on_hand_end(info)
