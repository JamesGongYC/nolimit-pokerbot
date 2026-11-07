"""Offline Monte Carlo table generation (simulation-based lookup tables).

Produces pokerbot/data/tables_<variant>.json with:
  pf_pct  : starting-hand percentile (combo-weighted, 0 = best)
  pf_eq   : preflop equity [range][k-1][class] vs k opponents in range
  hu      : 169x169 heads-up class-vs-class equity (x1000, ints)
  post0/1/2 : postflop bucket equity at 3 abstraction levels [range][k]
Only 2-hole-card variants are supported (NLHE, PL/FL HE, short deck).
"""
import json
import os
import random
import time
from multiprocessing import Pool

from .cards import hand_class, class_combos
from .event import Event
from .features import Features, RANGES, K_MAX, POST_K, table_path, DATA_DIR

_V = None


def _init(variant_cfg):
    global _V
    from .variant import Variant
    _V = Variant.from_config(variant_cfg)


def _share(hero, opps):
    best = max(opps)
    if hero > best:
        return 1.0
    if hero < best:
        return 0.0
    return 1.0 / (1 + sum(1 for o in opps if o == hero))


def _deal_from_range(rng, deck_left, allowed):
    for _ in range(200):
        a, b = rng.sample(deck_left, 2)
        if allowed is None or allowed[hand_class(a, b)]:
            return a, b
    return a, b


def _allowed(pct, frac):
    if frac >= 1.0 or pct is None:
        return None
    return [p < frac for p in pct]


def _rank_job(args):
    cls, n, seed = args
    v = _V
    rng = random.Random(seed)
    combos = class_combos(cls, v.deck_ranks)
    if not combos:
        return cls, 0.0
    tot = 0.0
    for t in range(n):
        h = combos[t % len(combos)]
        deck = [c for c in v.deck if c not in h]
        k = 1 if t % 2 == 0 else 3
        draw = rng.sample(deck, 2 * k + 5)
        board = draw[2 * k:]
        hs = v.hand_value(h, board)
        opp = [v.hand_value(draw[2 * j:2 * j + 2], board) for j in range(k)]
        tot += _share(hs, opp)
    return cls, tot / n


def _pf_job(args):
    cls, ri, k, n, seed, pct = args
    v = _V
    rng = random.Random(seed)
    combos = class_combos(cls, v.deck_ranks)
    if not combos:
        return cls, ri, k, 0.0
    allowed = _allowed(pct, RANGES[ri])
    tot = 0.0
    for t in range(n):
        h = combos[t % len(combos)]
        left = [c for c in v.deck if c not in h]
        opps = []
        for _ in range(k):
            o = _deal_from_range(rng, left, allowed)
            left.remove(o[0])
            left.remove(o[1])
            opps.append(o)
        board = rng.sample(left, 5)
        hs = v.hand_value(h, board)
        tot += _share(hs, [v.hand_value(o, board) for o in opps])
    return cls, ri, k, tot / n


def _hu_job(args):
    a, n, seed = args
    v = _V
    rng = random.Random(seed)
    ca = class_combos(a, v.deck_ranks)
    row = [500] * 169
    if not ca:
        return a, row
    for b in range(169):
        cb = class_combos(b, v.deck_ranks)
        if not cb:
            continue
        tot = 0.0
        cnt = 0
        for t in range(n):
            h = ca[t % len(ca)]
            o = cb[(t * 7 + 3) % len(cb)]
            if h[0] in o or h[1] in o:
                o2 = [x for x in cb if x[0] not in h and x[1] not in h]
                if not o2:
                    continue
                o = o2[t % len(o2)]
            left = [c for c in v.deck if c not in h and c not in o]
            board = rng.sample(left, 5)
            tot += _share(v.hand_value(h, board), [v.hand_value(o, board)])
            cnt += 1
        row[b] = int(round(1000 * tot / cnt)) if cnt else 500
    return a, row


def _post_job(args):
    ri, k, n, seed, pct = args
    v = _V
    feats = Features(v, tables={})
    rng = random.Random(seed)
    allowed = _allowed(pct, RANGES[ri])
    acc = [{}, {}, {}]
    for _ in range(n):
        left = list(v.deck)
        rng.shuffle(left)
        h = (left.pop(), left.pop())
        opps = []
        for _j in range(k):
            o = _deal_from_range(rng, left, allowed)
            left.remove(o[0])
            left.remove(o[1])
            opps.append(o)
        board = left[:5]
        sh = _share(v.hand_value(h, board), [v.hand_value(o, board) for o in opps])
        for s, nb in ((1, 3), (2, 4), (3, 5)):
            made, draw, tex = feats.bucket(h, board[:nb])
            for lv, key in enumerate(("%d,%d" % (s, made), "%d,%d,%d" % (s, made, draw),
                                      "%d,%d,%d,%d" % (s, made, draw, tex))):
                e = acc[lv].get(key)
                if e is None:
                    acc[lv][key] = [sh, 1]
                else:
                    e[0] += sh
                    e[1] += 1
    return ri, k, acc


def run(event, samples=1.0, workers=None, out=None, log=print):
    v = event.variant
    if v.hole_cards != 2:
        raise SystemExit("precompute supports 2-hole-card variants only (got %d)" % v.hole_cards)
    workers = workers or os.cpu_count()
    vcfg = event.cfg.get("variant", {"name": "nlhe"})
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(vcfg,)) as pool:
        # 1. ranking
        n_rank = int(1500 * samples)
        rank = dict(pool.map(_rank_job, [(c, n_rank, 11 + c) for c in range(169)]))
        weights = [len(class_combos(c, v.deck_ranks)) for c in range(169)]
        order = sorted(range(169), key=lambda c: -rank[c])
        total = float(sum(weights))
        pct = [1.0] * 169
        cum = 0.0
        for c in order:
            if weights[c]:
                pct[c] = (cum + weights[c] / 2.0) / total
            cum += weights[c]
        log("ranking done %.1fs" % (time.time() - t0))
        # 2. preflop equity vs ranges
        n_pf = int(600 * samples)
        jobs = [(c, ri, k, n_pf, 1000 * c + 100 * ri + k, pct)
                for c in range(169) for ri in range(len(RANGES)) for k in range(1, K_MAX + 1)]
        pf_eq = [[[0.0] * 169 for _ in range(K_MAX)] for _ in RANGES]
        for c, ri, k, e in pool.imap_unordered(_pf_job, jobs, chunksize=16):
            pf_eq[ri][k - 1][c] = round(e, 4)
        log("preflop equity done %.1fs" % (time.time() - t0))
        # 3. heads-up matrix
        n_hu = int(160 * samples)
        hu = [0] * (169 * 169)
        for a, row in pool.imap_unordered(_hu_job, [(a, n_hu, 7 + a) for a in range(169)]):
            hu[a * 169:(a + 1) * 169] = row
        log("hu matrix done %.1fs" % (time.time() - t0))
        # 4. postflop buckets
        n_post = int(40000 * samples)
        jobs = []
        for ri in range(len(RANGES)):
            for k in POST_K:
                for part in range(workers):
                    jobs.append((ri, k, n_post // workers, 99991 * ri + 101 * k + part, pct))
        merged = {}
        for ri, k, acc in pool.imap_unordered(_post_job, jobs):
            for lv in range(3):
                for key, (s, cnt) in acc[lv].items():
                    m = merged.setdefault((lv, key), {})
                    e = m.setdefault((ri, k), [0.0, 0])
                    e[0] += s
                    e[1] += cnt
        posts = [{}, {}, {}]
        min_n = 30
        for (lv, key), m in merged.items():
            row = []
            ok = True
            for ri in range(len(RANGES)):
                r = []
                for k in POST_K:
                    s, cnt = m.get((ri, k), (0.0, 0))
                    if cnt < min_n:
                        ok = False
                    r.append(round(s / cnt, 4) if cnt else None)
                row.append(r)
            if ok:
                posts[lv][key] = row
        log("postflop buckets done %.1fs (%d/%d/%d keys)" % (time.time() - t0, len(posts[0]),
                                                              len(posts[1]), len(posts[2])))
    tables = {"variant": v.name, "ranges": RANGES, "post_k": POST_K, "pf_pct": [round(p, 4) for p in pct],
              "pf_eq": pf_eq, "hu": hu, "post0": posts[0], "post1": posts[1], "post2": posts[2],
              "samples": samples, "build_seconds": round(time.time() - t0, 1)}
    out = out or table_path(v.name)
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    with open(out, "w") as f:
        json.dump(tables, f, separators=(",", ":"))
    log("wrote %s (%.0f KB)" % (out, os.path.getsize(out) / 1024.0))
    return out


if __name__ == "__main__":
    run(Event.load())
