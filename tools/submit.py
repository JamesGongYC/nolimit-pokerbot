"""Upload a submission to the No Limit 2026 site and make it the active bot.

Endpoints are the ones the site's own client uses:
  POST  <api>/teams/<team>/submissions?filename=<name>   body = the raw file
  GET   <api>/teams/<team>/submissions                   list (status validating|ready|rejected|cancelled)
  POST  <api>/teams/<team>/submissions/<id>/cancel
  PATCH <api>/teams/<team>   {"active_submission_id": "<id>"}   select the active bot
An upload alone changes nothing: each scored round snapshots the ACTIVE bot at
its start. The API key is read from ~/.config/no-limit-2026/api-key (never
printed).

usage:
  python3 tools/submit.py check  [submission.zip]          local upload-format checks only
  python3 tools/submit.py upload [submission.zip] [--activate] [--wait 300] [--dry-run]
  python3 tools/submit.py list
  python3 tools/submit.py status
  python3 tools/submit.py activate <submission_id>
  python3 tools/submit.py cancel <submission_id>
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

SITE = "https://pokerbot.sequoiacap.com/tournament"
API = SITE + "/api"
KEY_FILE = os.path.expanduser("~/.config/no-limit-2026/api-key")
STAMP = os.path.expanduser("~/.config/no-limit-2026/last-upload")
COOLDOWN_S = 30          # competition `submission_cooldown_seconds` (API); rules page says 5 min
LIMIT_MB = 50.0          # format page: current upload default


def key():
    if os.environ.get("NO_LIMIT_API_KEY"):
        return os.environ["NO_LIMIT_API_KEY"].strip()
    with open(KEY_FILE) as f:
        return f.read().strip()


def call(method, path, body=None, ctype=None):
    req = urllib.request.Request(API + path, data=body, method=method)
    req.add_header("Authorization", "Bearer " + key())
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "no-limit-2026-submit/1")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw[:300].decode("latin-1")}


def team_id(explicit=None):
    if explicit:
        return explicit
    code, d = call("GET", "/auth/api-key")
    if code != 200:
        sys.exit("API key check failed (%s): %s" % (code, d))
    teams = d.get("teams") or []
    if len(teams) != 1:
        sys.exit("expected exactly one team for this key, got %s; pass --team" %
                 [(t["team_name"], t["team_id"]) for t in teams])
    return teams[0]["team_id"]


def local_check(path):
    """The format page's rules: agent.py, or a zip with agent.py at its root;
    50 MB default upload limit. Returns (ok, report)."""
    rep = {"file": path, "exists": os.path.exists(path)}
    if not rep["exists"]:
        return False, rep
    rep["size_mb"] = round(os.path.getsize(path) / 1048576.0, 3)
    rep["under_limit"] = rep["size_mb"] < LIMIT_MB
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            rep["zip_ok"] = z.testzip() is None
            rep["unpacked_mb"] = round(sum(i.file_size for i in z.infolist()) / 1048576.0, 3)
        rep["agent_at_root"] = "agent.py" in names
        rep["files"] = len(names)
        rep["caches"] = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        ok = rep["zip_ok"] and rep["agent_at_root"] and rep["under_limit"] and not rep["caches"]
    else:
        rep["agent_at_root"] = os.path.basename(path) == "agent.py"
        src = open(path).read() if rep["agent_at_root"] else ""
        rep["needs_package"] = "jj_nolimit_bot" in src or "bot_spec.json" in src
        ok = rep["agent_at_root"] and rep["under_limit"] and not rep["needs_package"]
        if rep["needs_package"]:
            rep["hint"] = "this agent.py imports files next to it; upload submission.zip instead"
    rep["ok"] = ok
    return ok, rep


def submissions(team):
    code, d = call("GET", "/teams/%s/submissions" % team)
    if code != 200:
        sys.exit("list failed (%s): %s" % (code, d))
    return d.get("items", d) if isinstance(d, dict) else d


def show(s):
    print("%s  %-10s  %-28s %8.2f MB  %s%s" % (
        s.get("id"), s.get("status"), s.get("filename"), (s.get("size_bytes") or 0) / 1048576.0,
        s.get("created_at"), ("  REJECTED: %s" % s["rejection_reason"]) if s.get("rejection_reason") else ""))


def activate(team, sid):
    code, d = call("PATCH", "/teams/%s" % team, json.dumps({"active_submission_id": sid}).encode(),
                   "application/json")
    if code != 200 or (d or {}).get("active_submission_id") != sid:
        sys.exit("activation failed (%s): %s" % (code, d))
    print("ACTIVE bot is now %s" % sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "upload", "list", "status", "activate", "cancel"])
    ap.add_argument("arg", nargs="?", default=None)
    ap.add_argument("--team", default=None)
    ap.add_argument("--activate", action="store_true", help="select it as the active bot once it is ready")
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for the server check")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.cmd in ("check", "upload"):
        path = a.arg or "submission.zip"
        ok, rep = local_check(path)
        print(json.dumps(rep))
        if not ok:
            sys.exit("local upload-format check FAILED; not uploading")
        if a.cmd == "check":
            return
        if os.path.exists(STAMP):
            age = time.time() - os.path.getmtime(STAMP)
            if age < COOLDOWN_S:
                sys.exit("cooldown: last upload %.0fs ago (< %ds)" % (age, COOLDOWN_S))
        team = team_id(a.team)
        name = os.path.basename(path)
        ctype = "application/zip" if name.endswith(".zip") else "text/x-python"
        if a.dry_run:
            print("DRY RUN: POST %s/teams/%s/submissions?filename=%s (%s, %d bytes)%s" % (
                API, team, urllib.parse.quote(name), ctype, os.path.getsize(path),
                " then PATCH active_submission_id" if a.activate else ""))
            return
        with open(path, "rb") as f:
            body = f.read()
        code, d = call("POST", "/teams/%s/submissions?filename=%s" % (team, urllib.parse.quote(name)), body, ctype)
        if code not in (200, 201):
            sys.exit("upload failed (%s): %s" % (code, d))
        open(STAMP, "w").close()
        sid = d["id"]
        print("uploaded %s -> submission %s (status %s)" % (name, sid, d.get("status")))
        t0 = time.time()
        s = d
        while s.get("status") == "validating" and time.time() - t0 < a.wait:
            time.sleep(5)
            s = next((x for x in submissions(team) if x.get("id") == sid), s)
            pos = s.get("check_position")
            print("  ... %s%s" % (s.get("status"), (" (queue position %s)" % pos) if pos else ""))
        show(s)
        if s.get("check_log"):
            print("---- server check log ----\n" + s["check_log"][-4000:])
        if s.get("status") != "ready":
            sys.exit("submission is %s; the active bot is unchanged" % s.get("status"))
        if a.activate:
            activate(team, sid)
        else:
            print("ready but NOT active; run: python3 tools/submit.py activate %s" % sid)
        return

    team = team_id(a.team)
    if a.cmd == "list":
        for s in submissions(team):
            show(s)
    elif a.cmd == "status":
        code, d = call("GET", "/teams/%s" % team)
        if code != 200:
            sys.exit("team read failed (%s): %s" % (code, d))
        print("team %s, active submission: %s" % (d.get("name"), d.get("active_submission_id")))
    elif a.cmd == "activate":
        activate(team, a.arg)
    elif a.cmd == "cancel":
        code, d = call("POST", "/teams/%s/submissions/%s/cancel" % (team, a.arg))
        print(code, d)


if __name__ == "__main__":
    main()
