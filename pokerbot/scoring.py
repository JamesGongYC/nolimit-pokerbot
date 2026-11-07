"""Placement points and the Independent Chip Model (Malmuth-Harville)."""


def icm(stacks, payouts):
    """ICM equity for each stack. `payouts[k]` = points for place k+1.

    Zero stacks get nothing. Exact bitmask DP over the paid places only:
    O(sum_{m<k} C(n,m) * (n-m)), trivial for n <= 9.
    """
    idx = [i for i, s in enumerate(stacks) if s > 0]
    n = len(idx)
    eq = [0.0] * len(stacks)
    if n == 0:
        return eq
    st = [float(stacks[i]) for i in idx]
    total = sum(st)
    k = min(n, len(payouts))
    while k > 0 and payouts[k - 1] == 0:
        k -= 1
    if k == 0:
        return eq
    local = [0.0] * n
    layer = {0: (1.0, 0.0)}  # mask -> (prob, chips already placed)
    for place in range(k):
        pay = payouts[place]
        nxt = {}
        for mask, (p, used) in layer.items():
            rem = total - used
            if rem <= 0:
                continue
            for j in range(n):
                bit = 1 << j
                if mask & bit:
                    continue
                q = p * st[j] / rem
                local[j] += q * pay
                if place + 1 < k:
                    m2 = mask | bit
                    if m2 in nxt:
                        nxt[m2] = (nxt[m2][0] + q, used + st[j])
                    else:
                        nxt[m2] = (q, used + st[j])
        layer = nxt
    for a, i in enumerate(idx):
        eq[i] = local[a]
    return eq


def bust_points(busted, start_stacks, survivors, payouts, rule="stack"):
    """Places/points for players eliminated on the same hand.

    busted: seats eliminated this hand; survivors: number of players with
    chips after the hand. Places survivors+1 .. survivors+len(busted).
    rule 'stack': larger start-of-hand stack places higher, equal stacks
    split; rule 'split': everyone splits the block evenly.
    Returns {seat: (place, points)}; place is the best place in a tie group.
    """
    out = {}
    if not busted:
        return out
    first = survivors + 1
    if rule == "split":
        groups = [list(busted)]
    else:
        order = sorted(busted, key=lambda s: -start_stacks[s])
        groups = []
        for s in order:
            if groups and start_stacks[groups[-1][0]] == start_stacks[s]:
                groups[-1].append(s)
            else:
                groups.append([s])
    place = first
    for g in groups:
        pts = [payouts[p - 1] if p - 1 < len(payouts) else 0 for p in range(place, place + len(g))]
        share = sum(pts) / float(len(g))
        for s in g:
            out[s] = (place, share)
        place += len(g)
    return out
