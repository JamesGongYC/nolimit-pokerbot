#!/bin/sh
# Every strategy family against each other, three ways:
#  1. hold-out field leaderboard (matched tables, CI, latency, memory, cost)
#  2. all-vs-all official format (each strategy = one entry, fillers, T1..T6 weighted)
#  3. 1-vs-7 cross-play matrix (+ replicator meta share)
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-python3}; W=${W:-7}
TUNED="results/opt_heur/best.json,results/opt_heur2/best.json,results/opt_heur_league/best.json,results/opt_heur_random/best.json,results/opt_gbdt/best.json,results/opt_rf/best.json,results/opt_linear/best.json,results/opt_eq2/best.json,results/opt_hybrid_h/best.json,results/opt_mc/best.json,results/opt_jam/best.json,results/mixture/best.json"
PRESETS="heur_tag,eq_tag,mc_tag,lookup_pf,hybrid_q,jam_always,jam_50,jam_25,jam_12,jam_nash,jam_nash_icm,jam_stack,potodds,nit,shover20,station,maniac,random"
ALL="$TUNED,$PRESETS"
$PY -m pokerbot.cli --workers $W evaluate --candidates "$ALL" --tables ${HO_TABLES:-3000} --out results/grand_holdout.json > results/grand_holdout.log
echo "holdout done"
$PY -m pokerbot.cli --workers $W tournament --entries "$ALL" --iterations ${IT:-2000} --tournaments 6 --out results/grand_official.json > results/grand_official.log
echo "official done"
$PY -m pokerbot.cli --workers $W crossplay --strategies "$ALL" --tables ${XT:-40} --out results/grand_crossplay.json > results/grand_crossplay.log
echo "crossplay done"
