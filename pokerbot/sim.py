"""Parallel simulation: table jobs, matched-lineup evaluation, official-format
tournaments, and metric aggregation.

Statistical unit = one table (independent seed and lineup). Hands within a
table are never treated as independent samples.
"""
import math
import os
import random
import time
from multiprocessing import Pool

from .engine import play_table, SeatStats, LAT_BUCKET_US
from .event import Event
from .strategies import make_agent, resolve

_EVENT = None


SIM_TIME_SCALE = float(os.environ.get("POKERBOT_SIM_TIME_SCALE", "3.0"))


def _init_worker(event_cfg):
    global _EVENT
    import gc
    from .features import load_tables
    from .pushfold import load_pushfold
    _EVENT = Event(event_cfg)
    load_tables(_EVENT.variant.name)
    load_pushfold(_EVENT.variant.name)
    gc.freeze()


def run_table_job(job):
    """job = (lineup specs, seed, extra). Returns compact per-seat results."""
    lineup, seed, extra = job
    ev = _EVENT
    vcfg = ev.cfg.get("variant")
    agents = [make_agent(spec, seed=(seed * 31 + i) & 0x7FFFFFFF, variant=vcfg) for i, spec in enumerate(lineup)]
    t0 = time.process_time()
    w0 = time.time()
    res = play_table(ev, agents, seed, time_scale=(extra or {}).get("time_scale", SIM_TIME_SCALE))
    wall = time.time() - w0
    if wall > 20:
        import sys
        sys.stderr.write("SLOW TABLE %.1fs cpu %.1fs seed %d lineup %s\n" % (
            wall, time.process_time() - t0, seed, [s.get("name", s.get("type")) if isinstance(s, dict) else s
                                                   for s in lineup]))
        sys.stderr.flush()
    out = {"points": res.points, "places": res.places, "hands": res.hands_played,
           "capped": res.capped, "stats": [s.as_dict() for s in res.stats],
           "chip_fail": res.chip_check_failures, "cpu": time.process_time() - t0}
    if extra and extra.get("collect"):
        out["experience"] = [getattr(a, "experience", None) for a in agents]
    return out


class Runner(object):
    def __init__(self, event, workers=None):
        self.event = event
        self.workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            self.pool = Pool(self.workers, initializer=_init_worker, initargs=(self.event.cfg,))
        else:
            _init_worker(self.event.cfg)
        return self

    def __exit__(self, *a):
        if self.pool:
            self.pool.terminate()
            self.pool = None

    def run(self, jobs):
        if self.pool:
            return self.pool.map(run_table_job, jobs, chunksize=max(1, len(jobs) // (self.workers * 8)))
        return [run_table_job(j) for j in jobs]


# ------------------------------------------------------------- populations
def load_fields(path):
    import json
    with open(path) as f:
        return json.load(f)["fields"]


def sample_opponents(field, k, rng):
    specs = [m[0] for m in field]
    weights = [m[1] for m in field]
    return [rng.choices(specs, weights)[0] for _ in range(k)]


def matched_jobs(n_tables, fields, seed, table_size=8, archive=None, archive_frac=0.0):
    """Deterministic opponent lineups + seeds + candidate seat per table.
    Returns list of (opponents[table_size-1], seed, cand_seat, field_name)."""
    rng = random.Random(seed)
    names = sorted(fields)
    out = []
    for t in range(n_tables):
        fname = names[t % len(names)]
        opps = sample_opponents(fields[fname], table_size - 1, rng)
        if archive and archive_frac > 0:
            opps = [rng.choice(archive) if rng.random() < archive_frac else o for o in opps]
        out.append((opps, rng.randrange(1 << 30), t % table_size, fname))
    return out


def evaluate_candidates(runner, candidates, plan, collect=False):
    """Every candidate plays every planned table (same seed/opponents/seat).
    Returns per-candidate dict with per-table points and merged stats."""
    jobs = []
    for cand in candidates:
        for opps, seed, seat, fname in plan:
            lineup = list(opps)
            lineup.insert(seat, cand)
            jobs.append((lineup, seed, {"collect": collect}))
    results = runner.run(jobs)
    out = []
    k = len(plan)
    for ci, cand in enumerate(candidates):
        rows = results[ci * k:(ci + 1) * k]
        pts, places, by_field = [], [], {}
        st = SeatStats()
        cpu = 0.0
        exp = []
        for (opps, seed, seat, fname), r in zip(plan, rows):
            pts.append(r["points"][seat])
            places.append(r["places"][seat])
            by_field.setdefault(fname, []).append(r["points"][seat])
            st.merge(SeatStats.from_dict(r["stats"][seat]))
            cpu += r["cpu"]
            if collect:
                exp.append(r["experience"][seat])
        out.append({"spec": cand, "points": pts, "places": places, "by_field": by_field,
                    "stats": st, "cpu": cpu, "experience": exp})
    return out


# ------------------------------------------------------- official format
def tournament(runner, entries, iterations, seed, table_size=8):
    """Official-format scored tournament (competition format page): each
    iteration shuffles all entries into tables of `table_size`; the short
    last table is filled with bots not already seated at it, whose finishes
    never score. Fields smaller than a table play short tables. Returns
    per-entry mean points per table and per-iteration samples."""
    rng = random.Random(seed)
    n = len(entries)
    n_tables = int(math.ceil(n / float(table_size)))
    jobs, meta = [], []
    for it in range(iterations):
        order = list(range(n))
        rng.shuffle(order)
        for t in range(n_tables):
            tab = [(e, True) for e in order[t * table_size:(t + 1) * table_size]]
            short = table_size - len(tab)
            if short:
                here = set(e for e, _ in tab)
                pool = [e for e in range(n) if e not in here]
                tab += [(e, False) for e in rng.sample(pool, min(short, len(pool)))]
                rng.shuffle(tab)
            jobs.append(([entries[e] for e, _ in tab], rng.randrange(1 << 30), None))
            meta.append((it, tab))
    results = runner.run(jobs)
    pts = [[0.0] * iterations for _ in range(n)]
    places = [[0] * iterations for _ in range(n)]
    stats = [SeatStats() for _ in range(n)]
    for (it, tab), r in zip(meta, results):
        for seat, (e, scored) in enumerate(tab):
            if scored:
                pts[e][it] = r["points"][seat]
                places[e][it] = r["places"][seat]
                stats[e].merge(SeatStats.from_dict(r["stats"][seat]))
    return {"points": pts, "places": places, "stats": stats}


# ---------------------------------------------------------------- metrics
def mean_ci(xs, z=1.96):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / float(n)
    if n < 2:
        return m, float("inf")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, z * math.sqrt(var / n)


def latency_pct(stats, q):
    tot = sum(stats.hist)
    if not tot:
        return 0.0
    acc = 0
    for i, c in enumerate(stats.hist):
        acc += c
        if acc >= q * tot:
            return (i + 1) * LAT_BUCKET_US
    return len(stats.hist) * LAT_BUCKET_US


def behavior(stats):
    s = stats
    h = float(max(1, s.hands))
    post_aggr = s.bets + s.raises
    return {
        "vpip": round(s.vpip / h, 3),
        "pfr": round(s.pfr / h, 3),
        "open": round(s.pf_opens / float(max(1, s.pf_open_opps)), 3),
        "steal": round(s.steals / float(max(1, s.steal_opps)), 3),
        "3bet": round(s.pf_3bets / float(max(1, s.pf_3bet_opps)), 3),
        "af": round(post_aggr / float(max(1, s.calls)), 2),
        "fold_to_bet": round(s.fold_to_bet / float(max(1, s.face_bet)), 3),
        "wtsd": round(s.showdowns / float(max(1, s.saw_flop)), 3),
        "allin_per_hand": round(s.allins / h, 3),
    }


def place_dist(places, size=8):
    n = float(max(1, len(places)))
    return [round(sum(1 for p in places if p == k) / n, 3) for k in range(1, size + 1)]


def summarize(name, points, places, stats, cpu=0.0, train_cost=None, mechanism=None):
    m, ci = mean_ci(points)
    d = {"name": name, "points": round(m, 3), "ci95": round(ci, 3), "tables": len(points),
         "place_dist": place_dist(places),
         "lat_p50_us": latency_pct(stats, 0.5), "lat_p99_us": latency_pct(stats, 0.99),
         "lat_max_us": round(stats.max_ns / 1000.0, 1),
         "mean_us": round(stats.time_ns / 1000.0 / max(1, stats.decisions), 1),
         "decisions": stats.decisions, "errors": stats.errors, "timeouts": stats.timeouts,
         "illegal": stats.illegal, "behavior": behavior(stats), "sim_cpu_s": round(cpu, 1)}
    if train_cost is not None:
        d["train_cpu_s"] = train_cost
    if mechanism:
        d["mechanism"] = mechanism
    return d


def spec_name(spec):
    s = resolve(spec)
    return s.get("name", s.get("type"))
