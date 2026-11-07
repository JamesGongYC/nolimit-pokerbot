"""Runtime lookup of push/fold equilibrium tables (solved offline by
pfsolve.py). See pfsolve.py for the abstraction and its limits."""
import json
import os

from .features import DATA_DIR

STACKS = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25]
LAMS = [0.0, 0.3, 0.7]
NS = list(range(1, 8))


class PushFoldTable(object):
    """Runtime lookup: nearest solved (n, S, lam) cell."""

    def __init__(self, data):
        self.d = data
        self.stacks = data["stacks"]
        self.lams = data["lams"]
        self._widths = {}

    def _key(self, n, s, lam):
        n = max(1, min(7, n))
        S = min(self.stacks, key=lambda x: abs(x - s))
        L = min(self.lams, key=lambda x: abs(x - lam))
        return "%d|%s|%s" % (n, S, L)

    def push_width(self, n_behind, stack_bb, lam=0.0):
        """Combo-weighted fraction of hands pushed in the nearest cell."""
        key = self._key(n_behind, stack_bb, lam)
        w = self._widths.get(key)
        if w is None:
            v = self.d["push"][key]
            w = sum(_COMBOS[i] * int(v[i]) for i in range(169)) / (9.0 * 1326)
            self._widths[key] = w
        return w

    def push_prob(self, cls, n_behind, stack_bb, lam=0.0):
        return int(self.d["push"][self._key(n_behind, stack_bb, lam)][cls]) / 9.0

    def call_prob(self, cls, n_behind_pusher, stack_bb, caller_type, lam=0.0):
        cell = self.d["call"][self._key(n_behind_pusher, stack_bb, lam)]
        s = cell.get(caller_type) or cell.get("bb")
        return int(s[cls]) / 9.0


_COMBOS = [6 if c < 13 else (4 if c < 91 else 12) for c in range(169)]
_PF = {}


def load_pushfold(variant_name="nlhe"):
    if variant_name not in _PF:
        p = os.path.join(DATA_DIR, "pushfold_%s.json" % variant_name)
        _PF[variant_name] = PushFoldTable(json.load(open(p))) if os.path.exists(p) else None
    return _PF[variant_name]
