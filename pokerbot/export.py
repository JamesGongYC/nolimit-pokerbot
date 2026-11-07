"""Export a strategy in the competition's upload format and benchmark the
exported interface path.

Upload format (competition format page): `agent.py`, or a zip with `agent.py`
at its root; 50 MB default limit; follow the starter kit for package and
unpacked-file limits. An upload must then be selected as the active bot
(tools/submit.py).

submission/
  agent.py          entry point + adapter (edit to_internal/from_internal
                    once the official Agent interface is known)
  bot_spec.json     frozen strategy spec
  jj_nolimit_bot/   runtime-only modules + data tables (no offline code),
                    renamed from `pokerbot` so nothing can shadow or be
                    shadowed by the harness's own modules
submission.zip      the same tree, agent.py at the zip root (this is uploaded)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

from .strategies import resolve

PKG = os.path.dirname(os.path.abspath(__file__))
RUNTIME_PKG = "jj_nolimit_bot"  # package name inside the upload
RUNTIME_MODULES = ["__init__.py", "agent_api.py", "cards.py", "variant.py", "features.py", "pushfold.py",
                   "event.py", "engine.py", "scoring.py",
                   "strategies/__init__.py", "strategies/base.py", "strategies/parametric.py",
                   "strategies/lookup.py", "strategies/learned.py", "strategies/hybrid.py",
                   "strategies/mixture.py", "strategies/simple.py",
                   "strategies/allin.py", "strategies/montecarlo.py", "strategies/treebot.py"]
ZIP_SKIP_DIRS = {"__pycache__", ".git", ".venv"}
ZIP_SKIP_FILES = {".DS_Store"}
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(socket|urllib|http|requests|subprocess|multiprocessing|"
                       r"threading|asyncio|ssl|ftplib|smtplib)\b", re.M)

AGENT_PY = '''"""Generated submission entry point (agent.py). Strategy: %(name)s

Adapter notes (PROVISIONAL): the official Agent interface is released with
the starter kit. Map it in `to_internal` / `from_internal`; everything else
is fixed. Decisions never block: any exception -> check/fold.
"""
import json
import os
import random
import sys

# Works whatever the harness's working directory is: everything is loaded
# relative to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from %(pkg)s.strategies import make_agent  # noqa: E402
from %(pkg)s.agent_api import legalize, fallback_action  # noqa: E402
from %(pkg)s.features import load_tables  # noqa: E402
from %(pkg)s.pushfold import load_pushfold  # noqa: E402

with open(os.path.join(HERE, "bot_spec.json")) as _f:
    SPEC = json.load(_f)
VARIANT = SPEC.get("variant") or {"name": "nlhe"}
# Warm all tables at import time so no decision pays the load cost.
load_tables(VARIANT.get("name", "nlhe"))
load_pushfold(VARIANT.get("name", "nlhe"))
make_agent(SPEC, seed=0, variant=VARIANT)
import gc  # noqa: E402
gc.freeze()  # keep the large read-only tables out of GC traversals


def to_internal(obs):
    """Official observation -> internal obs dict (identity for the local sim)."""
    return obs


def from_internal(action, obs):
    """Internal (kind, amount) -> official action format."""
    return action


class Agent(object):
    """One instance per table (the engine resets bot state per table)."""

    def __init__(self, *args, **kwargs):
        self._bot = make_agent(SPEC, seed=random.randrange(1 << 30), variant=VARIANT)

    def act(self, obs):
        o = None
        try:
            o = to_internal(obs)
            a = self._bot.act(o)
            a, _ = legalize(o, a)
            return from_internal(a, obs)
        except Exception:
            try:
                return from_internal(fallback_action(o if o is not None else obs), obs)
            except Exception:
                return ("fold", 0)

    def on_hand_end(self, info):
        try:
            self._bot.on_hand_end(info)
        except Exception:
            pass
'''

BENCH = r'''
import json, os, sys, time, resource, tracemalloc
sys.path.insert(0, os.getcwd())
t0 = time.perf_counter()
import agent as bot
import_ms = (time.perf_counter() - t0) * 1000
from PKGNAME.event import Event
from PKGNAME.engine import play_table, SeatStats
from PKGNAME.strategies import make_agent
ev = Event(json.loads(sys.argv[1]))
tables = int(sys.argv[2])
opp = json.loads(sys.argv[3])
st = SeatStats()
tracemalloc.start()
for t in range(tables):
    seat = t % 8
    agents = [make_agent(opp[(t + i) % len(opp)], t * 8 + i) for i in range(8)]
    agents[seat] = bot.Agent()
    r = play_table(ev, agents, 1000 + t)
    st.merge(r.stats[seat])
    if t == 0:
        # all-8-exported table: every seat through the exported path
        r2 = play_table(ev, [bot.Agent() for _ in range(8)], 5)
        for s in r2.stats:
            st.merge(s)
cur, peak = tracemalloc.get_traced_memory()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / (1024 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
print(json.dumps({"import_ms": round(import_ms, 1), "stats": st.as_dict(), "py_peak_kb": peak // 1024,
                  "rss_mb": round(rss_mb, 1)}))
'''


def export(spec, out_dir="submission", variant=None, log=print, zip_path=None, limit_mb=None):
    """Write the submission tree to `out_dir` and the upload zip to `zip_path`
    (default: `<out_dir>.zip`). Raises SystemExit if a check fails."""
    spec = resolve(spec)
    if variant:
        spec["variant"] = variant
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    pkg_out = os.path.join(out_dir, RUNTIME_PKG)
    os.makedirs(os.path.join(pkg_out, "strategies"))
    os.makedirs(os.path.join(pkg_out, "data"))
    for m in RUNTIME_MODULES:
        shutil.copy(os.path.join(PKG, m), os.path.join(pkg_out, m))
    vname = (spec.get("variant") or {"name": "nlhe"}).get("name", "nlhe")
    for fn in ("tables_%s.json" % vname, "pushfold_%s.json" % vname):
        src = os.path.join(PKG, "data", fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(pkg_out, "data", fn))
        else:
            log("WARNING: missing %s (bot will use fallbacks)" % fn)
    # copy referenced lookup tables (including inside mixtures)
    for t in _tables_in(spec):
        src = t if os.path.isabs(t) else os.path.join(os.path.dirname(PKG), t)
        dst_rel = os.path.join(RUNTIME_PKG, "data", os.path.basename(t))
        shutil.copy(src, os.path.join(out_dir, dst_rel))
        _retarget(spec, t, dst_rel)
    # an empty event config so the bundled local simulator runs for benchmarking
    cfgdir = os.path.join(out_dir, "configs")
    os.makedirs(cfgdir)
    shutil.copy(os.path.join(os.path.dirname(PKG), "configs", "event_provisional.json"), cfgdir)
    spec_out = {k: v for k, v in spec.items() if k not in ("train_cpu_s", "train_wall_s")}
    json.dump(spec_out, open(os.path.join(out_dir, "bot_spec.json"), "w"), indent=1)
    open(os.path.join(out_dir, "agent.py"), "w").write(AGENT_PY % {"name": spec.get("name"), "pkg": RUNTIME_PKG})
    zip_path = zip_path or out_dir.rstrip("/\\") + ".zip"
    package(out_dir, zip_path)
    check = static_checks(out_dir, zip_path, limit_mb)
    log("exported %s -> %s + %s (zip %.2f MB, unpacked %.2f MB, limit %s MB, agent.py at root: %s, "
        "forbidden imports: %s)" % (spec.get("name"), out_dir, zip_path, check["zip_mb"], check["size_mb"],
                                    check["limit_mb"], check["agent_at_root"], check["forbidden"] or "none"))
    if not check["ok"]:
        raise SystemExit("export FAILED upload checks: %s" % json.dumps(check))
    st = selftest(zip_path, spec)
    check["selftest"] = st
    log("selftest (unzipped, foreign cwd): %s" % json.dumps(st))
    if not st.get("ok"):
        raise SystemExit("export FAILED selftest: %s" % json.dumps(st))
    return check


SELFTEST = r'''
import importlib.util, json, sys
sp = importlib.util.spec_from_file_location("agent", sys.argv[1])
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
need = json.loads(sys.argv[2])
a = m.Agent()
b = a._bot
inner = getattr(b, "inner", b)
out = {"bot": type(inner).__name__,
       "model_loaded": getattr(inner, "m", None) is not None,
       "table_loaded": bool(getattr(inner, "table", None)),
       "leaked_modules": sorted(k for k in sys.modules if k == "pokerbot" or k.startswith("pokerbot."))}
from PKGNAME.event import Event
from PKGNAME.engine import play_table
from PKGNAME.features import load_tables
from PKGNAME.pushfold import load_pushfold
v = (m.VARIANT or {}).get("name", "nlhe")
out["equity_tables"] = load_tables(v) is not None
out["pushfold_tables"] = load_pushfold(v) is not None
r = play_table(Event.load(), [m.Agent() for _ in range(8)], 11)
out["chips_ok"] = sum(r.final_stacks) == sum(Event.load().starting_stack for _ in range(8))
out["errors"] = sum(s.errors + s.illegal for s in r.stats)
out["ok"] = (out["chips_ok"] and out["errors"] == 0 and not out["leaked_modules"]
             and (out["model_loaded"] or not need["model"]) and (out["table_loaded"] or not need["table"])
             and out["equity_tables"] and out["pushfold_tables"])
print(json.dumps(out))
'''


def selftest(zip_path, spec):
    """Unzip the upload into a temp dir and, in a fresh interpreter whose
    working directory is unrelated, import agent.py by path, check every
    referenced model/table actually loaded and play one table of 8 exported
    agents. Catches silent fallbacks (a missing model makes a TreeBot play as
    its heuristic base)."""
    import tempfile
    p = spec.get("params", {})
    need = {"model": bool(p.get("model")), "table": bool(p.get("table"))}
    d = tempfile.mkdtemp()
    try:
        unz = os.path.join(d, "unzipped")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(unz)
        other = os.path.join(d, "cwd")
        os.makedirs(other)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        out = subprocess.check_output([sys.executable, "-c", SELFTEST.replace("PKGNAME", RUNTIME_PKG),
                                       os.path.join(unz, "agent.py"), json.dumps(need)],
                                      cwd=other, env=env, stderr=subprocess.STDOUT, timeout=300)
        return json.loads(out.decode().strip().splitlines()[-1])
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": e.output.decode()[-2000:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def package(out_dir, zip_path):
    """Zip the submission tree with agent.py at the archive root (no wrapping
    folder), skipping caches; entries sorted for reproducible archives."""
    if os.path.exists(zip_path):
        os.remove(zip_path)
    entries = []
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = sorted(d for d in dirs if d not in ZIP_SKIP_DIRS)
        for f in sorted(files):
            if f in ZIP_SKIP_FILES or f.endswith(".pyc"):
                continue
            full = os.path.join(root, f)
            entries.append((full, os.path.relpath(full, out_dir).replace(os.sep, "/")))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arc in entries:
            z.write(full, arc)
    return zip_path


def _tables_in(spec):
    out = []
    p = spec.get("params", {})
    for key in ("table", "model"):
        if p.get(key):
            out.append(p[key])
    for c in p.get("components", []) or []:
        out += _tables_in(resolve(c))
    return out


def _retarget(spec, old, new):
    p = spec.get("params", {})
    for key in ("table", "model"):
        if p.get(key) == old:
            p[key] = new
    comps = p.get("components")
    if comps:
        p["components"] = [resolve(c) for c in comps]
        for c in p["components"]:
            _retarget(c, old, new)


def static_checks(out_dir, zip_path=None, limit_mb=None):
    """Upload-format checks: agent.py at the root (directory and zip), sizes
    under the limit (default: event config, 50 MB), no forbidden imports."""
    if limit_mb is None:
        try:
            from .event import Event
            limit_mb = float(Event.load().cfg.get("submission_size_limit_mb", 50))
        except Exception:
            limit_mb = 50.0
    size = 0
    forbidden = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            path = os.path.join(root, f)
            size += os.path.getsize(path)
            if f.endswith(".py"):
                src = open(path).read()
                for m in FORBIDDEN.finditer(src):
                    forbidden.append("%s: %s" % (os.path.relpath(path, out_dir), m.group(0).strip()))
    out = {"size_mb": round(size / 1048576.0, 3), "forbidden": forbidden, "limit_mb": limit_mb,
           "agent_at_root": os.path.isfile(os.path.join(out_dir, "agent.py"))}
    if zip_path and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            unpacked = sum(i.file_size for i in z.infolist())
        out["zip_mb"] = round(os.path.getsize(zip_path) / 1048576.0, 3)
        out["zip_unpacked_mb"] = round(unpacked / 1048576.0, 3)
        out["zip_files"] = len(names)
        out["agent_at_root"] = out["agent_at_root"] and "agent.py" in names
    else:
        out["zip_mb"] = 0.0
    out["ok"] = (out["agent_at_root"] and not forbidden and out["size_mb"] < limit_mb
                 and out["zip_mb"] < limit_mb)
    return out


def bench(out_dir="submission", tables=40, event=None, opponents=None, log=print):
    from .event import Event
    from .engine import SeatStats
    from .sim import latency_pct
    event = event or Event.load()
    opp = opponents or ["station", "maniac", "heur_tag", "eq_default", "shover20", "nit", "random"]
    opp = [resolve(o) for o in opp]
    out = subprocess.check_output([sys.executable, "-c", BENCH.replace("PKGNAME", RUNTIME_PKG), json.dumps(event.cfg),
                                   str(tables), json.dumps(opp)],
                                  cwd=out_dir, stderr=subprocess.STDOUT, timeout=1800)
    d = json.loads(out.decode().strip().splitlines()[-1])
    st = SeatStats.from_dict(d["stats"])
    zp = out_dir.rstrip("/\\") + ".zip"
    check = static_checks(out_dir, zp if os.path.exists(zp) else None,
                          event.cfg.get("submission_size_limit_mb", 50))
    lim = event.time_limit_ms * 1000
    rep = {"decisions": st.decisions, "p50_us": latency_pct(st, 0.5), "p99_us": latency_pct(st, 0.99),
           "p999_us": latency_pct(st, 0.999), "max_us": round(st.max_ns / 1000.0, 1),
           "over_limit": sum(c for i, c in enumerate(st.hist) if (i + 1) * 10 > lim),
           "timeouts": st.timeouts, "errors": st.errors, "illegal": st.illegal,
           "import_ms": d["import_ms"], "py_peak_kb": d["py_peak_kb"], "rss_mb": d["rss_mb"],
           "size_mb": check["size_mb"], "zip_mb": check["zip_mb"], "limit_mb": check["limit_mb"],
           "size_ok": check["ok"], "agent_at_root": check["agent_at_root"],
           "forbidden_imports": check["forbidden"]}
    log(json.dumps(rep, indent=1))
    return rep
