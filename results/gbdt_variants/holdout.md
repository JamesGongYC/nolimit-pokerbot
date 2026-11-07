# Leaderboard (sequoia-nolimit-2026-provisional, 4000 hold-out tables, payouts [50, 30, 20, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gbdt_full | tree ensemble Q (GBDT / RF, fitted policy iteration) | 26.12 | 0.66 | +0.00 (0.00) | 20.73 | 37/17/12 | 120/480/29625 | 0/3393/0 | 18705 | 2468.3 |
| 2 | gbdt_c40_4 | tree ensemble Q (GBDT / RF, fitted policy iteration) | 26.10 | 0.65 | -0.02 (0.65) | 20.21 | 36/17/14 | 60/160/13505 | 0/1689/0 | 8341 | 2468.3 |
| 3 | T-heur | heuristic | 25.02 | 0.64 | -1.09 (0.71) | 20.81 | 34/17/15 | 20/70/17230 | 0/592/0 | 4371 | 1955.4 |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| gbdt_full | 21.52 | 27.94 | 20.73 | 27.38 | 33.01 |
| gbdt_c40_4 | 21.51 | 27.74 | 20.21 | 28.11 | 32.92 |
| T-heur | 21.32 | 24.83 | 20.81 | 27.35 | 30.82 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| gbdt_full | 0.384 | 0.347 | 0.536 | 0.596 | 0.162 | 4.03 | 0.572 | 0.369 | 0.041 |
| gbdt_c40_4 | 0.379 | 0.339 | 0.53 | 0.588 | 0.148 | 3.84 | 0.576 | 0.372 | 0.037 |
| T-heur | 0.339 | 0.294 | 0.51 | 0.561 | 0.035 | 3.35 | 0.609 | 0.352 | 0.037 |
