# Leaderboard (provisional-payout_linear, 2000 hold-out tables, payouts [8, 7, 6, 5, 4, 3, 2, 1])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag_g9_mean_g6_7 | heuristic | 5.87 | 0.10 | +0.00 (0.00) | 5.14 | 38/12/11 | 20/210/8322 | 0/396/0 |  | 2317.4 |
| 2 | heur_tag_g9_mean | heuristic | 5.86 | 0.09 | -0.01 (0.09) | 5.18 | 32/16/13 | 20/200/8360 | 0/422/0 |  | 1955.4 |
| 3 | linear_default_g9_mean | learned (CEM) | 5.83 | 0.08 | -0.04 (0.11) | 4.98 | 21/21/20 | 50/220/16544 | 0/673/0 |  | 1916.7 |
| 4 | mix_g5_4 | randomized mixture | 5.83 | 0.09 | -0.04 (0.08) | 5.12 | 34/14/12 | 30/210/7426 | 0/458/0 |  | 610.3 |
| 5 | heur_tag | heuristic | 5.69 | 0.08 | -0.18 (0.10) | 5.03 | 18/23/19 | 20/210/12376 | 0/493/0 |  | - |
| 6 | eq_tag_g6_8 | statistical+oppmodel | 5.63 | 0.09 | -0.24 (0.11) | 4.77 | 24/17/17 | 20/240/14359 | 0/460/0 |  | 2181.6 |
| 7 | nit | baseline | 5.25 | 0.08 | -0.62 (0.11) | 4.49 | 4/26/24 | 10/30/5350 | 0/19/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| heur_tag_g9_mean_g6_7 | 5.14 | 6.12 | 5.29 | 6.24 | 6.56 |
| heur_tag_g9_mean | 5.27 | 6.08 | 5.18 | 6.46 | 6.31 |
| linear_default_g9_mean | 5.61 | 6.17 | 4.98 | 6.21 | 6.20 |
| mix_g5_4 | 5.22 | 6.18 | 5.12 | 6.17 | 6.44 |
| heur_tag | 5.27 | 6.00 | 5.03 | 6.01 | 6.13 |
| eq_tag_g6_8 | 5.09 | 6.00 | 4.77 | 6.03 | 6.26 |
| nit | 4.49 | 5.71 | 4.65 | 5.67 | 5.73 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag_g9_mean_g6_7 | 0.346 | 0.309 | 0.532 | 0.602 | 0.027 | 3.89 | 0.603 | 0.324 | 0.048 |
| heur_tag_g9_mean | 0.302 | 0.264 | 0.448 | 0.51 | 0.032 | 3.43 | 0.646 | 0.362 | 0.042 |
| linear_default_g9_mean | 0.223 | 0.221 | 0.326 | 0.493 | 0.117 | 33.67 | 0.77 | 0.534 | 0.02 |
| mix_g5_4 | 0.318 | 0.288 | 0.484 | 0.572 | 0.046 | 4.36 | 0.636 | 0.335 | 0.044 |
| heur_tag | 0.198 | 0.145 | 0.231 | 0.288 | 0.043 | 2.45 | 0.762 | 0.394 | 0.047 |
| eq_tag_g6_8 | 0.233 | 0.191 | 0.311 | 0.393 | 0.046 | 5.24 | 0.736 | 0.408 | 0.07 |
| nit | 0.054 | 0.05 | 0.062 | 0.059 | 0.047 | 3.7 | 0.915 | 0.727 | 0.014 |
