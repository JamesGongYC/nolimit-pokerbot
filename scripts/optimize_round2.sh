#!/bin/sh
# Round 2: after the opponent-model fix and EV-calibration parameters.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-python3}; W=${W:-7}
$PY -m pokerbot.cli --workers $W optimize --base eq_tag   --method cem --generations 12 --popsize 16 --screen-tables 300 --final-tables 1500 --seed 22 --out results/opt_eq2
$PY -m pokerbot.cli --workers $W optimize --base hybrid_h --method cem --generations 10 --popsize 16 --screen-tables 300 --final-tables 1500 --seed 23 --sigma 0.1 --out results/opt_hybrid_h
$PY -m pokerbot.cli --workers $W optimize --base results/opt_heur/best.json --method cem --generations 8 --popsize 16 --screen-tables 400 --final-tables 2000 --seed 24 --sigma 0.1 --out results/opt_heur2
