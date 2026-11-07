#!/bin/sh
# Hold-out leaderboard + official-format simulation + export + benchmark.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-python3}
TABLES=${TABLES:-3000}
BESTS=""
for f in results/opt_heur/best.json results/opt_eq/best.json results/opt_hybrid/best.json results/opt_linear/best.json results/opt_heur_random/best.json results/mixture/best.json; do
  [ -f "$f" ] && BESTS="$BESTS,$f"
done
BESTS=${BESTS#,}
$PY -m pokerbot.cli evaluate --tables $TABLES --out results/leaderboard_holdout.json \
  --candidates "$BESTS,heur_tag,heur_default,eq_tag,eq_lag,eq_icm,eq_bully,hybrid_q,hybrid_tag,lookup_pf,linear_default,nit,shover20,station,maniac,random"
