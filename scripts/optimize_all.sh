#!/bin/sh
# Reproducible optimisation pipeline (training fields only; hold-out untouched).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
GEN=${GEN:-10}; POP=${POP:-16}; SCREEN=${SCREEN:-300}; FINAL=${FINAL:-1500}
$PY -m pokerbot.cli optimize --base heur_tag  --method cem --generations $GEN --popsize $POP --screen-tables $SCREEN --final-tables $FINAL --seed 11 --out results/opt_heur
$PY -m pokerbot.cli optimize --base eq_tag    --method cem --generations $GEN --popsize $POP --screen-tables $SCREEN --final-tables $FINAL --seed 12 --out results/opt_eq
$PY -m pokerbot.cli optimize --base hybrid_q  --method cem --generations $GEN --popsize $POP --screen-tables $SCREEN --final-tables $FINAL --seed 13 --out results/opt_hybrid
$PY -m pokerbot.cli optimize --base linear_default --method cem --space linear --generations $GEN --popsize $POP --screen-tables $SCREEN --final-tables $FINAL --seed 14 --out results/opt_linear
$PY -m pokerbot.cli optimize --base heur_tag  --method random --n-random 64 --popsize 16 --screen-tables 200 --final-tables $FINAL --seed 15 --out results/opt_heur_random
