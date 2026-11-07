# Leaderboard (sequoia-nolimit-2026-provisional, 1500 hold-out tables, payouts [50, 30, 20, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag_g9_mean | heuristic | 24.35 | 1.02 | +0.00 (0.00) | 20.69 | 32/17/17 | 20/70/7399 | 0/203/0 |  | 1955.4 |
| 2 | eq_tag_g4_7 | statistical+oppmodel | 20.68 | 0.94 | -3.67 (1.30) | 14.38 | 20/24/19 | 20/90/6476 | 0/219/0 |  | 1722.1 |
| 3 | hybrid_q_g9_mean | hybrid (CFR push/fold + table + equity + oppmodel) | 18.82 | 0.93 | -5.53 (1.30) | 14.19 | 18/23/17 | 20/100/6319 | 0/133/0 |  | 1626.7 |
| 4 | equity_w_heur_params | statistical+oppmodel | 16.83 | 0.98 | -7.52 (1.22) | 12.29 | 19/15/15 | 30/100/8179 | 0/183/0 |  | 1955.4 |

## Field sensitivity (mean points by hold-out field)

| strategy | aggressive | balanced | exploitative | mixed | passive |
|---|---|---|---|---|---|
| heur_tag_g9_mean | 25.82 | 20.69 | 22.26 | 25.72 | 27.25 |
| eq_tag_g4_7 | 25.96 | 14.38 | 18.39 | 21.34 | 23.32 |
| hybrid_q_g9_mean | 22.73 | 14.19 | 16.24 | 20.58 | 20.35 |
| equity_w_heur_params | 21.35 | 12.29 | 13.93 | 19.64 | 16.96 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag_g9_mean | 0.333 | 0.287 | 0.504 | 0.566 | 0.036 | 3.1 | 0.617 | 0.305 | 0.036 |
| eq_tag_g4_7 | 0.22 | 0.169 | 0.287 | 0.385 | 0.034 | 5.29 | 0.748 | 0.358 | 0.085 |
| hybrid_q_g9_mean | 0.17 | 0.122 | 0.15 | 0.202 | 0.117 | 3.13 | 0.791 | 0.358 | 0.034 |
| equity_w_heur_params | 0.322 | 0.276 | 0.487 | 0.558 | 0.033 | 7.85 | 0.631 | 0.285 | 0.056 |
