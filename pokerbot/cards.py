"""Card encoding and a fast, variant-aware hand evaluator (stdlib only).

Card = int: rank * 4 + suit, rank 0 = deuce .. 12 = ace, suit 0..3.
`Evaluator.evaluate(cards)` returns an int; larger is better. It works for
5..7 cards ("any" hole-card usage, i.e. Hold'em style) and supports
short-deck category orders. Omaha-style "exactly k hole cards" is handled
by `Evaluator.evaluate_split` (slower, combinatorial).
"""
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"

HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ["high card", "pair", "two pair", "trips", "straight", "flush",
                  "full house", "quads", "straight flush"]
STANDARD_ORDER = list(range(9))  # index = category, value = strength rank


def card(s):
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def cards(s):
    s = s.replace(" ", "")
    return [card(s[i:i + 2]) for i in range(0, len(s), 2)]


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def cards_str(cs):
    return " ".join(card_str(c) for c in cs)


POP = [bin(m).count("1") for m in range(8192)]
BITS_DESC = [tuple(r for r in range(12, -1, -1) if m >> r & 1) for m in range(8192)]


def _kick(ranks):
    v = 0
    for i in range(5):
        v = (v << 4) | (ranks[i] + 1 if i < len(ranks) else 0)
    return v


TOP5 = [_kick(BITS_DESC[m][:5]) for m in range(8192)]


def _straight_table(deck_ranks):
    lo = sorted(deck_ranks)
    table = [-1] * 8192
    windows = []
    for i in range(len(lo) - 4):
        if lo[i + 4] - lo[i] == 4:
            windows.append((sum(1 << r for r in lo[i:i + 5]), lo[i + 4]))
    if 12 in lo:  # wheel: ace plays low with the four lowest ranks of the deck
        windows.append(((1 << 12) | sum(1 << r for r in lo[:4]), lo[3]))
    windows.sort(key=lambda w: -w[1])
    for m in range(8192):
        for wm, hi in windows:
            if m & wm == wm:
                table[m] = hi
                break
    return table


class Evaluator(object):
    """Variant-aware evaluator. `category_order[c]` = strength of category c."""

    def __init__(self, deck_ranks=tuple(range(13)), category_order=None):
        self.deck_ranks = tuple(deck_ranks)
        self.straight = _straight_table(self.deck_ranks)
        order = category_order or STANDARD_ORDER
        self.cat_base = [order[c] << 20 for c in range(9)]
        self.cat_of_value = {order[c]: c for c in range(9)}

    def category(self, score):
        return self.cat_of_value[score >> 20]

    def evaluate(self, cs):
        cnt = [0] * 13
        sm = [0, 0, 0, 0]
        for c in cs:
            r = c >> 2
            cnt[r] += 1
            sm[c & 3] |= 1 << r
        rm = sm[0] | sm[1] | sm[2] | sm[3]
        base = self.cat_base
        best = -1
        for s in sm:
            if POP[s] >= 5:
                sh = self.straight[s]
                if sh >= 0:
                    return base[STRAIGHT_FLUSH] | (sh + 1)
                best = base[FLUSH] | TOP5[s]
                break
        quad = trip = trip2 = p1 = p2 = -1
        for r in BITS_DESC[rm]:
            n = cnt[r]
            if n == 4:
                quad = r
            elif n == 3:
                if trip < 0:
                    trip = r
                elif trip2 < 0:
                    trip2 = r
            elif n == 2:
                if p1 < 0:
                    p1 = r
                elif p2 < 0:
                    p2 = r
        if quad >= 0:
            k = BITS_DESC[rm & ~(1 << quad)]
            v = base[QUADS] | ((quad + 1) << 16) | ((k[0] + 1) << 12 if k else 0)
            return v if v > best else best
        if trip >= 0:
            pr = max(trip2, p1)
            if pr >= 0:
                v = base[FULL_HOUSE] | ((trip + 1) << 16) | ((pr + 1) << 12)
                if v > best:
                    best = v
        sh = self.straight[rm]
        if sh >= 0:
            v = base[STRAIGHT] | (sh + 1)
            if v > best:
                best = v
        if trip >= 0:
            k = BITS_DESC[rm & ~(1 << trip)]
            v = base[TRIPS] | ((trip + 1) << 16) | (_k2(k) << 8)
            return v if v > best else best
        if best >= 0:
            return best
        if p2 >= 0:
            k = BITS_DESC[rm & ~(1 << p1) & ~(1 << p2)]
            return base[TWO_PAIR] | ((p1 + 1) << 16) | ((p2 + 1) << 12) | ((k[0] + 1) << 8 if k else 0)
        if p1 >= 0:
            k = BITS_DESC[rm & ~(1 << p1)]
            return base[PAIR] | ((p1 + 1) << 16) | (_k3(k) << 4)
        return base[HIGH] | TOP5[rm]

    def evaluate_split(self, hole, board, use_hole):
        """Best hand using exactly `use_hole` hole cards (Omaha-style)."""
        best = -1
        for h in combinations(hole, use_hole):
            for b in combinations(board, 5 - use_hole):
                v = self.evaluate(h + b)
                if v > best:
                    best = v
        return best


def _k2(k):
    return ((k[0] + 1) << 4 if len(k) > 0 else 0) | (k[1] + 1 if len(k) > 1 else 0)


def _k3(k):
    return (((k[0] + 1) << 8 if len(k) > 0 else 0) | ((k[1] + 1) << 4 if len(k) > 1 else 0)
            | (k[2] + 1 if len(k) > 2 else 0))


STANDARD = Evaluator()


def hand_class(c1, c2):
    """Index 0..168 of a two-card starting hand: pairs 0..12, then suited, offsuit."""
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1
    idx = r1 * (r1 - 1) // 2 + r2  # 0..77 for r1>r2
    return 13 + idx if (c1 & 3) == (c2 & 3) else 91 + idx


def class_name(i):
    if i < 13:
        return RANKS[i] * 2
    suited = i < 91
    idx = i - 13 if suited else i - 91
    r1 = 1
    while (r1 + 1) * r1 // 2 <= idx:
        r1 += 1
    r2 = idx - r1 * (r1 - 1) // 2
    return RANKS[r1] + RANKS[r2] + ("s" if suited else "o")


def class_combos(i, deck_ranks=tuple(range(13))):
    """All concrete (c1, c2) combos of starting-hand class i."""
    name = class_name(i)
    r1, r2 = RANKS.index(name[0]), RANKS.index(name[1])
    if r1 not in deck_ranks or r2 not in deck_ranks:
        return []
    out = []
    for s1 in range(4):
        for s2 in range(4):
            a, b = r1 * 4 + s1, r2 * 4 + s2
            if r1 == r2 and s1 >= s2:
                continue
            if name.endswith("s") and s1 != s2:
                continue
            if name.endswith("o") and s1 == s2:
                continue
            out.append((a, b))
    return out
