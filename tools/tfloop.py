"""Continuous data -> transformer loop (run with the kit venv). Never uploads.

Every cycle:
  1. list the competition's rounds; download every completed round's hand
     history not yet on disk (results/hands/index.json maps round id -> file)
  2. extract preflop decision rows per round (tools/tfdata.py) -> results/tf/r_<id>.npz
  3. retrain the transformer on ALL rounds with recency weights
     w = decay ** (rounds newer than this one)  (newest round = 1.0)
  4. validate: numpy == torch, forward latency, newest-round held-out fit,
     kit check, pack bots/auto/jj_tf_v<N>.zip, paired sim vs the active bot
  5. log to results/tf/loop.jsonl; sleep until the next round completes
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "starter-kit", ".venv", "bin", "python")
PA = os.path.join(ROOT, "starter-kit", ".venv", "bin", "poker-agent")
API = "https://pokerbot.sequoiacap.com/tournament/api"
COMP = "5f8be52b-8038-4c43-a0fa-971abcacec1b"
TEAM = "6b12d15b-331f-4393-bc9b-5e8e3689d76d"
HANDS = os.path.join(ROOT, "results", "hands")
TF = os.path.join(ROOT, "results", "tf")
KNOWN = {  # files downloaded earlier in the session (round id -> file)
    "5da6d631-2bcf-4fe2-9e7a-01b9e8ecac75": "practice1.jsonl.gz",
    "af2a0969-cd0a-4124-875b-84ff690a2a6d": "practice2.jsonl.gz",
    "4d1fe6fb-7336-4745-bfb4-cd0e565fb552": "practice4.jsonl.gz",
    "3e3dcc02-ca4f-48c5-9acb-6c0e3a32c10e": "practice5.jsonl.gz",      # = scored round 1
    "547666b1-3877-4bac-9925-17e1e78134da": "scored1.jsonl.gz",        # = scored round 2
    "74eb2fd7-8ade-46fb-8e6a-6a099170b812": "practice7_jamtop.jsonl.gz",
    "260d2303-7878-4df1-867e-05f8d2a10d01": "scored3.jsonl.gz",
}


def log(msg):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(TF, "loop.log"), "a") as f:
        f.write(line + "\n")


def key():
    return open(os.path.expanduser("~/.config/no-limit-2026/api-key")).read().strip()


def get(path, out=None):
    req = urllib.request.Request(API + path, headers={"Authorization": "Bearer " + key(),
                                                      "User-Agent": "no-limit-2026-tfloop/1"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if out:
            tmp = out + ".part"
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, out)
            return out
        return json.loads(r.read())


def rounds():
    items = get("/competitions/%s/tournaments" % COMP)["items"]
    return sorted([t for t in items if t.get("completed_at")], key=lambda t: t["created_at"])


def sync(index):
    for t in rounds():
        tid = t["id"]
        f = index.get(tid) or KNOWN.get(tid)
        if f and os.path.exists(os.path.join(HANDS, f)):
            index[tid] = f
            continue
        f = "%s_%s_%s.jsonl.gz" % (t["kind"], t["created_at"][11:19].replace(":", ""), tid[:8])
        log("downloading %s round %s (%s)" % (t["kind"], tid[:8], t["created_at"][11:19]))
        try:
            get("/teams/%s/tournaments/%s/hands" % (TEAM, tid), os.path.join(HANDS, f))
            index[tid] = f
        except Exception as e:
            log("  download failed: %r" % (e,))
    json.dump(index, open(os.path.join(HANDS, "index.json"), "w"), indent=1)
    return index


def extract(index, order):
    out = []
    for tid in order:
        npz = os.path.join(TF, "r_%s.npz" % tid[:8])
        if not os.path.exists(npz):
            log("extracting %s" % index[tid])
            subprocess.run([PY, os.path.join(ROOT, "tools", "tfdata.py"), npz, os.path.join(HANDS, index[tid])],
                           check=True, capture_output=True)
        out.append(npz)
    return out


def train_and_stage(v, npzs, decay, active_dir):
    n = len(npzs)
    args = ["%s@%.4f" % (p, decay ** (n - 1 - i)) for i, p in enumerate(npzs)]  # newest last, weight 1
    model = os.path.join(TF, "model_v%d.npz" % v)
    env = dict(os.environ, TF_EPOCHS=os.environ.get("TF_EPOCHS", "2"))
    r = subprocess.run([PY, "-u", os.path.join(ROOT, "tools", "tftrain.py"), model] + args,
                       capture_output=True, text=True, env=env)
    open(os.path.join(TF, "train_v%d.log" % v), "w").write(r.stdout + r.stderr)
    if r.returncode != 0:
        log("training v%d failed: %s" % (v, (r.stdout + r.stderr)[-400:]))
        return None
    report = [ln for ln in r.stdout.splitlines() if "held-out" in ln or "saved" in ln]
    # stage a bot folder
    bot = os.path.join(ROOT, "bots", "jj_tf_v%d" % v)
    if os.path.isdir(bot):
        shutil.rmtree(bot)
    os.makedirs(bot)
    shutil.copy(os.path.join(ROOT, "bots", "jj_tf", "agent.py"), bot)
    shutil.copy(os.path.join(ROOT, "bots", "jj_tf", "jtbase.py"), bot)
    shutil.copy(model, os.path.join(bot, "model.npz"))
    chk = subprocess.run([PA, "check", os.path.join(bot, "agent.py")], capture_output=True, text=True)
    zp = os.path.join(ROOT, "bots", "auto", "jj_tf_v%d.zip" % v)
    if os.path.exists(zp):
        os.remove(zp)
    pk = subprocess.run([PA, "pack", bot, zp], capture_output=True, text=True)
    # verification + paired sim vs the active bot
    ver = subprocess.run([PY, os.path.join(ROOT, "tools", "tfverify.py"), model, bot, active_dir],
                         capture_output=True, text=True)
    row = {"v": v, "t": time.time(), "rounds": n, "decay": decay, "train": report,
           "kit_check_ok": chk.returncode == 0, "packed": pk.returncode == 0, "zip": zp,
           "verify": ver.stdout.strip().splitlines()[-6:] if ver.stdout else ver.stderr[-400:]}
    with open(os.path.join(TF, "loop.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    log("v%d ready: %s | %s | %s" % (v, " ; ".join(report[-3:]), " ; ".join(row["verify"]) if isinstance(row["verify"], list) else row["verify"], zp))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decay", type=float, default=0.8)
    ap.add_argument("--every", type=float, default=120.0, help="seconds between polls")
    ap.add_argument("--start-v", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(TF, exist_ok=True)
    ip = os.path.join(HANDS, "index.json")
    index = json.load(open(ip)) if os.path.exists(ip) else {}
    v = a.start_v
    trained_on = None
    for _ in range(a.cycles):
        try:
            index = sync(index)
            order = [t["id"] for t in rounds() if t["id"] in index]
            if order and order != trained_on:
                npzs = extract(index, order)
                act = json.load(open(os.path.join(ROOT, "results", "kitevo_main", "active.json")))
                log("training v%d on %d rounds (decay %.2f), newest %s" % (v, len(npzs), a.decay, order[-1][:8]))
                if train_and_stage(v, npzs, a.decay, os.path.join(ROOT, act["dir"])):
                    v += 1
                trained_on = order
        except Exception as e:
            log("cycle error: %r" % (e,))
        time.sleep(a.every)


if __name__ == "__main__":
    main()
