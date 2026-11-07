"""Fast, bounded-time hand-strength features backed by precomputed tables.

All runtime functions are table lookups plus a few bit operations, so a
decision costs tens of microseconds. Tables are produced offline by
`python -m pokerbot.cli precompute` (see precompute.py) and stored in
pokerbot/data/tables_<variant>.json.

Preflop: class (169 two-card classes), percentile, equity vs k opponents
holding a range (RANGES = top fraction of combos).
Postflop: bucket = (street, made-hand class, draw class, board texture) at
three abstraction levels, with equity vs k opponents per range.
"""
import json
import os

from .cards import POP, hand_class

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RANGES = [1.0, 0.5, 0.25, 0.12]  # opponent range = top X of combos
K_MAX = 7          # preflop opponents 1..7
POST_K = [1, 2, 3]  # postflop opponents (3 = 3+)

# made-hand classes
(M_AIR, M_OVERS, M_WEAK_PAIR, M_MID_PAIR, M_TOP_WEAK, M_TOP_GOOD, M_OVERPAIR, M_TWO_PAIR_BOARD,
 M_TWO_PAIR, M_TRIPS, M_SET, M_STRAIGHT, M_FLUSH_LOW, M_FLUSH_HIGH, M_FULL, M_MONSTER) = range(16)
MADE_NAMES = ["air", "overcards", "weak_pair", "mid_pair", "top_pair_weak", "top_pair_good",
              "overpair", "two_pair_board", "two_pair", "trips", "set", "straight", "flush_low",
              "flush_high", "full_house", "quads_plus"]
D_NONE, D_GUT, D_OESD, D_FLUSH, D_COMBO = range(5)
DRAW_NAMES = ["none", "gutshot", "oesd", "flush_draw", "combo_draw"]


class Features(object):
    """Variant-bound feature extractor + table lookups."""

    def __init__(self, variant, tables=None):
        self.variant = variant
        self.ev = variant.evaluator
        self.straight = self.ev.straight
        self.low_rank = min(variant.deck_ranks)
        self.tables = tables if tables is not None else load_tables(variant.name)
        t = self.tables or {}
        self.pf_eq = t.get("pf_eq")        # [range][k-1][cls] -> equity (0..1)
        self.pf_pct = t.get("pf_pct")      # [cls] -> percentile 0 (best) .. 1
        self.post = [t.get("post0"), t.get("post1"), t.get("post2")]  # key -> [range][k]
        self.hu = t.get("hu")              # flat 169*169 equity*1000 (row = hero)
        self.straight_board = {}

    # -------------------------------------------------------------- preflop
    def preflop_class(self, hole):
        return hand_class(hole[0], hole[1])

    def preflop_pct(self, hole):
        if self.pf_pct is None:
            return _chen_pct(hole)
        return self.pf_pct[hand_class(hole[0], hole[1])]

    def preflop_equity(self, hole, k, range_idx=0):
        k = max(1, min(K_MAX, k))
        if self.pf_eq is None:
            p = _chen_pct(hole)
            return (1.0 - p) ** 0.8 / (k ** 0.55) + 0.05
        return self.pf_eq[range_idx][k - 1][hand_class(hole[0], hole[1])]

    def hu_equity(self, c1, c2):
        return self.hu[c1 * 169 + c2] / 1000.0

    # ------------------------------------------------------------- postflop
    def bucket(self, hole, board):
        """(made, draw, texture) for 2-card hole + 3..5 board cards."""
        ev = self.ev
        cs = list(hole) + list(board)
        score = self.variant.hand_value(hole, board)
        cat = ev.category(score)
        brs = [c >> 2 for c in board]
        bmask = 0
        bcnt = {}
        for r in brs:
            bmask |= 1 << r
            bcnt[r] = bcnt.get(r, 0) + 1
        h1, h2 = hole[0] >> 2, hole[1] >> 2
        top = max(brs)
        board_paired = len(bcnt) < len(brs)
        # ---- made class
        if cat >= 7:
            made = M_MONSTER
        elif cat == 6:
            made = M_FULL
            if self._board_only_cat(board) == 6:
                made = M_TRIPS
        elif cat == 5:
            fs = _flush_suit(cs)
            hs = [c >> 2 for c in hole if c & 3 == fs]
            if not hs:
                made = M_AIR  # flush on board, no hole card in it
            else:
                made = M_FLUSH_HIGH if max(hs) >= 10 else M_FLUSH_LOW
        elif cat == 4:
            made = M_STRAIGHT if self.straight[bmask] < 0 else M_WEAK_PAIR
        elif cat == 3:
            if h1 == h2:
                made = M_SET
            elif bcnt.get(h1, 0) == 2 or bcnt.get(h2, 0) == 2:
                made = M_TRIPS
            else:
                made = M_AIR  # trips on board
        elif cat == 2:
            board_pairs = [r for r, c in bcnt.items() if c >= 2]
            if (h1 in bcnt and h2 in bcnt and h1 != h2):
                made = M_TWO_PAIR
            elif h1 == h2 and h1 > top:
                made = M_TWO_PAIR_BOARD  # overpair on a paired board
            elif board_pairs and (h1 in bcnt or h2 in bcnt):
                made = M_TWO_PAIR_BOARD
            elif len(board_pairs) >= 2:
                made = M_OVERS if min(h1, h2) > top else M_AIR
            else:
                made = M_MID_PAIR
        elif cat == 1:
            if board_paired:
                made = M_OVERS if min(h1, h2) > top else M_AIR
            elif h1 == h2:
                if h1 > top:
                    made = M_OVERPAIR
                elif h1 > sorted(brs)[-2]:
                    made = M_MID_PAIR
                else:
                    made = M_WEAK_PAIR
            else:
                pr = h1 if h1 in bcnt else h2
                kicker = h2 if pr == h1 else h1
                if pr == top:
                    made = M_TOP_GOOD if kicker >= 9 else M_TOP_WEAK
                elif pr == sorted(set(brs))[0]:
                    made = M_WEAK_PAIR
                else:
                    made = M_MID_PAIR
        else:
            made = M_OVERS if min(h1, h2) > top else M_AIR
        # ---- draws (flop/turn only)
        draw = D_NONE
        if len(board) < 5 and made < M_STRAIGHT:
            fd = False
            for s in range(4):
                n_all = sum(1 for c in cs if c & 3 == s)
                n_hole = sum(1 for c in hole if c & 3 == s)
                if n_all == 4 and n_hole >= 1:
                    fd = True
            hmask = bmask | (1 << h1) | (1 << h2)
            outs = 0
            st = self.straight
            for r in self.variant.deck_ranks:
                bit = 1 << r
                if hmask & bit:
                    continue
                if st[hmask | bit] >= 0 and st[bmask | bit] < 0:
                    outs += 1
            sd = D_OESD if outs >= 2 else (D_GUT if outs == 1 else D_NONE)
            if fd and sd:
                draw = D_COMBO
            elif fd:
                draw = D_FLUSH
            else:
                draw = sd
        # ---- texture bits
        tex = 1 if board_paired else 0
        suits = [0, 0, 0, 0]
        for c in board:
            suits[c & 3] += 1
        if max(suits) >= 3:
            tex |= 2
        if self._straighty(bmask):
            tex |= 4
        return made, draw, tex

    def _board_only_cat(self, board):
        if len(board) < 5:
            return -1
        return self.ev.category(self.ev.evaluate(list(board)))

    def _straighty(self, bmask):
        v = self.straight_board.get(bmask)
        if v is None:
            # 3+ board ranks inside some 5-rank window -> straight possible
            v = False
            for lo in range(-1, 9):
                w = 0
                for r in range(lo, lo + 5):
                    rr = 12 if r == -1 else r
                    w |= 1 << rr
                if POP[bmask & w] >= 3:
                    v = True
                    break
            self.straight_board[bmask] = v
        return v

    def post_equity(self, hole, board, k, range_idx=0, level=2):
        """Equity vs k opponents holding range RANGES[range_idx]."""
        made, draw, tex = self.bucket(hole, board)
        return self.bucket_equity(len(board), made, draw, tex, k, range_idx, level)

    def bucket_equity(self, nboard, made, draw, tex, k, range_idx=0, level=2):
        s = 1 if nboard == 3 else (2 if nboard == 4 else 3)
        ki = min(max(k, 1), 3) - 1
        keys = ["%d,%d,%d,%d" % (s, made, draw, tex), "%d,%d,%d" % (s, made, draw), "%d,%d" % (s, made)]
        for key, tab in ((keys[0], self.post[2]), (keys[1], self.post[1]), (keys[2], self.post[0])):
            if level < 2 and tab is self.post[2]:
                continue
            if level < 1 and tab is self.post[1]:
                continue
            if tab is not None:
                row = tab.get(key)
                if row is not None:
                    e = row[range_idx][ki]
                    if k > 3:
                        e = e ** (k / 3.0)
                    return e
        e = _fallback_equity(made, draw)
        return e ** k


def _flush_suit(cs):
    cnt = [0, 0, 0, 0]
    for c in cs:
        cnt[c & 3] += 1
    for s in range(4):
        if cnt[s] >= 5:
            return s
    return -1


def _fallback_equity(made, draw):
    base = [0.25, 0.32, 0.4, 0.5, 0.6, 0.68, 0.75, 0.62, 0.8, 0.82, 0.88, 0.9, 0.88, 0.94, 0.97, 0.99]
    return min(0.99, base[made] + [0, 0.05, 0.12, 0.13, 0.2][draw])


def _chen_pct(hole):
    """Crude percentile when no tables are available (0 best .. 1 worst)."""
    r1, r2 = sorted([hole[0] >> 2, hole[1] >> 2], reverse=True)
    suited = (hole[0] & 3) == (hole[1] & 3)
    s = r1 * 2.0 + r2 * 1.0 + (6 if r1 == r2 else 0) + (2 if suited else 0) - max(0, r1 - r2 - 1)
    return max(0.0, min(1.0, 1.0 - s / 48.0))


_CACHE = {}


def table_path(variant_name):
    return os.path.join(DATA_DIR, "tables_%s.json" % variant_name)


def load_tables(variant_name):
    if variant_name in _CACHE:
        return _CACHE[variant_name]
    p = table_path(variant_name)
    t = None
    if os.path.exists(p):
        with open(p) as f:
            t = json.load(f)
    _CACHE[variant_name] = t
    return t
