#!/bin/sh
# Robustness of finalists to unknown event parameters (payouts, blind speed). Hold-out fields.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-python3}; W=${W:-7}; TABLES=${TABLES:-2000}
CANDS=${CANDS:?set CANDS to a comma list of finalists}
for e in configs/events/*.json; do
  n=$(basename $e .json)
  $PY -m pokerbot.cli --workers $W --event $e evaluate --candidates "$CANDS" --tables $TABLES --no-memory --out results/sens_$n.json > /dev/null
  echo "done $n"
done
$PY - <<'PY'
import glob, json
rows = {}
for f in sorted(glob.glob("results/sens_*.json")):
    rep = json.load(open(f)); n = f[len("results/sens_"):-5]
    for r in rep["rows"]:
        rows.setdefault(r["name"], {})[n] = (r["points"], r["ci95"])
cfgs = sorted({k for v in rows.values() for k in v})
lines = ["| strategy | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
for name, v in rows.items():
    lines.append("| %s | " % name + " | ".join("%.2f ± %.2f" % v[c] if c in v else "" for c in cfgs) + " |")
md = "# Sensitivity to event parameters (hold-out fields)\n\nPoints are on each config's own payout scale (wta = 100 to the winner, linear = 8..1).\n\n" + "\n".join(lines) + "\n"
open("results/sensitivity.md", "w").write(md); print(md)
PY
