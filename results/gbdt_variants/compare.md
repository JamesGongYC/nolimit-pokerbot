# Leaderboard (sequoia-nolimit-2026-provisional, 3000 hold-out tables, payouts [50, 30, 20, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gbdt_c40_4 | tree ensemble Q (GBDT / RF, fitted policy iteration) | 26.63 | 0.74 | +0.00 (0.00) | 22.51 | 37/18/14 | 60/200/17482 | 0/1390/0 |  | 2468.3 |
| 2 | gbdt_c60_4 | tree ensemble Q (GBDT / RF, fitted policy iteration) | 26.43 | 0.75 | -0.20 (0.70) | 21.86 | 37/17/14 | 70/180/16727 | 0/1393/0 |  | 2468.3 |
| 3 | gbdt_full | tree ensemble Q (GBDT / RF, fitted policy iteration) | 26.35 | 0.74 | -0.28 (0.75) | 21.62 | 36/18/14 | 120/350/8689 | 0/2183/0 |  | 2468.3 |
| 4 | gbdt_c30_3 | tree ensemble Q (GBDT / RF, fitted policy iteration) | 25.63 | 0.74 | -1.00 (0.79) | 22.03 | 34/18/15 | 50/160/14114 | 0/1113/0 |  | 2468.3 |
| 5 | T-heur | heuristic | 25.14 | 0.73 | -1.49 (0.79) | 22.07 | 34/18/14 | 20/70/16568 | 0/587/0 |  | 1955.4 |

## Field sensitivity (mean points by hold-out field)

| strategy | aggressive | balanced | exploitative | mixed | passive |
|---|---|---|---|---|---|
| gbdt_c40_4 | 28.10 | 23.33 | 22.51 | 26.92 | 32.29 |
| gbdt_c60_4 | 27.58 | 23.71 | 21.86 | 27.17 | 31.86 |
| gbdt_full | 28.76 | 23.33 | 21.62 | 26.78 | 31.26 |
| gbdt_c30_3 | 26.88 | 22.30 | 22.03 | 26.63 | 30.32 |
| T-heur | 26.74 | 22.07 | 22.68 | 24.93 | 29.29 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| gbdt_c40_4 | 0.379 | 0.337 | 0.519 | 0.593 | 0.163 | 3.61 | 0.578 | 0.328 | 0.035 |
| gbdt_c60_4 | 0.385 | 0.346 | 0.524 | 0.599 | 0.181 | 3.76 | 0.574 | 0.329 | 0.038 |
| gbdt_full | 0.382 | 0.343 | 0.522 | 0.594 | 0.177 | 3.76 | 0.575 | 0.329 | 0.039 |
| gbdt_c30_3 | 0.388 | 0.332 | 0.522 | 0.593 | 0.142 | 3.2 | 0.575 | 0.314 | 0.03 |
| T-heur | 0.331 | 0.285 | 0.5 | 0.562 | 0.036 | 3.07 | 0.616 | 0.311 | 0.035 |
