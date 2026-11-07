"""Configuration-driven strategy registry.

A spec is a dict: {"type": <registry key>, "styles": [...], "params": {...}}
or the name of a preset defined in configs/agents/*.json.
"""
import glob
import json
import os

from .simple import RandomBot, CallingStation, Maniac, Shover, Nit, PotOddsBot
from .allin import AllInBot
from .montecarlo import MonteCarloBot
from .treebot import TreeBot, CollectorBot
from .parametric import HeuristicBot, EquityBot
from .lookup import LookupBot
from .learned import LinearBot
from .hybrid import HybridBot
from .mixture import MixtureBot

REGISTRY = {
    "random": RandomBot,
    "station": CallingStation,
    "maniac": Maniac,
    "shover": Shover,
    "nit": Nit,
    "heuristic": HeuristicBot,
    "equity": EquityBot,
    "lookup": LookupBot,
    "linear": LinearBot,
    "hybrid": HybridBot,
    "mixture": MixtureBot,
    "potodds": PotOddsBot,
    "allin": AllInBot,
    "montecarlo": MonteCarloBot,
    "tree": TreeBot,
    "collector": CollectorBot,
}

MECHANISM = {
    "random": "baseline", "station": "baseline", "maniac": "baseline", "shover": "baseline",
    "nit": "baseline", "heuristic": "heuristic", "equity": "statistical+oppmodel",
    "lookup": "lookup-table (rollout-trained)", "linear": "learned (CEM)",
    "hybrid": "hybrid (CFR push/fold + table + equity + oppmodel)",
    "mixture": "randomized mixture",
    "potodds": "baseline (pot odds)",
    "allin": "all-in / jam-fold",
    "montecarlo": "real-time Monte Carlo + EV",
    "tree": "tree ensemble Q (GBDT / RF, fitted policy iteration)",
    "collector": "data collector (training only)",
}

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PRESETS = None


def presets():
    global _PRESETS
    if _PRESETS is None:
        _PRESETS = {}
        for path in sorted(glob.glob(os.path.join(_ROOT, "configs", "agents", "*.json"))):
            with open(path) as f:
                _PRESETS.update(json.load(f))
    return _PRESETS


def resolve(spec):
    """Turn a preset name / spec into a full spec dict (with 'name')."""
    if isinstance(spec, str):
        pre = presets()
        if spec not in pre:
            raise KeyError("unknown agent preset %r" % spec)
        out = dict(pre[spec])
        out.setdefault("name", spec)
        return out
    out = dict(spec)
    if "preset" in out:
        base = resolve(out.pop("preset"))
        params = dict(base.get("params", {}))
        params.update(out.get("params", {}))
        base.update(out)
        base["params"] = params
        out = base
    out.setdefault("name", out.get("type", "agent"))
    return out


def make_agent(spec, seed=0, variant=None):
    spec = resolve(spec)
    cls = REGISTRY[spec["type"]]
    return cls(params=spec.get("params"), seed=seed, styles=spec.get("styles"), variant=variant)
