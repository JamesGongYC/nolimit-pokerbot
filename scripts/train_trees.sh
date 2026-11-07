#!/bin/sh
# Fitted policy iteration with GBDT and random forests around the tuned heuristic.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-python3}; W=${W:-7}
$PY -m pokerbot.cli --workers $W train-trees --base results/opt_heur/best.json --name gbdt --algo gbdt --rounds 3 --tables 10000 --eps 0.15 --trees 150 --depth 4 --leaf 1500 --lr 0.06 --margin 0.25 --eval-tables 2000 --seed 31
$PY -m pokerbot.cli --workers $W train-trees --base results/opt_heur/best.json --name rf --algo rf --rounds 2 --tables 10000 --eps 0.15 --trees 40 --depth 8 --leaf 1500 --margin 0.25 --eval-tables 2000 --seed 41
