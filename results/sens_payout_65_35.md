# Leaderboard (provisional-payout_65_35, 2000 hold-out tables, payouts [65, 35, 0, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag_g9_mean | heuristic | 28.29 | 1.27 | +0.00 (0.00) | 24.34 | 35/16/14 | 20/50/27882 | 0/243/0 |  | 1955.4 |
| 2 | heur_tag_g9_mean_g6_7 | heuristic | 28.07 | 1.29 | -0.22 (1.42) | 22.84 | 36/14/13 | 20/50/7157 | 0/215/0 |  | 2317.4 |
| 3 | mix_g5_4 | randomized mixture | 27.03 | 1.27 | -1.26 (1.48) | 21.90 | 34/15/14 | 20/60/13024 | 0/229/0 |  | 610.3 |
| 4 | linear_default_g9_mean | learned (CEM) | 23.26 | 1.15 | -5.02 (1.61) | 18.64 | 24/24/19 | 30/60/12502 | 0/239/0 |  | 1916.7 |
| 5 | heur_tag | heuristic | 22.54 | 1.14 | -5.74 (1.51) | 18.35 | 21/25/18 | 20/50/7885 | 0/212/0 |  | - |
| 6 | eq_tag_g6_8 | statistical+oppmodel | 21.06 | 1.16 | -7.23 (1.53) | 14.56 | 22/19/16 | 20/80/7514 | 0/215/0 |  | 2181.6 |
| 7 | nit | baseline | 15.79 | 0.86 | -12.50 (1.45) | 9.87 | 5/36/26 | 10/30/7384 | 0/21/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| heur_tag_g9_mean | 24.34 | 25.03 | 25.99 | 28.98 | 37.09 |
| heur_tag_g9_mean_g6_7 | 23.80 | 27.18 | 22.84 | 27.25 | 39.26 |
| mix_g5_4 | 21.90 | 27.22 | 22.17 | 27.08 | 36.78 |
| linear_default_g9_mean | 21.51 | 23.68 | 18.64 | 23.90 | 28.57 |
| heur_tag | 20.61 | 22.29 | 18.35 | 23.81 | 27.66 |
| eq_tag_g6_8 | 17.79 | 21.46 | 14.56 | 23.39 | 28.09 |
| nit | 14.68 | 19.87 | 9.87 | 18.32 | 16.20 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag_g9_mean | 0.349 | 0.302 | 0.526 | 0.572 | 0.037 | 3.28 | 0.597 | 0.349 | 0.035 |
| heur_tag_g9_mean_g6_7 | 0.374 | 0.332 | 0.582 | 0.637 | 0.032 | 3.7 | 0.574 | 0.326 | 0.043 |
| mix_g5_4 | 0.348 | 0.313 | 0.538 | 0.607 | 0.052 | 4.1 | 0.602 | 0.337 | 0.038 |
| linear_default_g9_mean | 0.259 | 0.256 | 0.383 | 0.513 | 0.141 | 20.36 | 0.724 | 0.544 | 0.021 |
| heur_tag | 0.228 | 0.161 | 0.263 | 0.306 | 0.049 | 2.24 | 0.724 | 0.374 | 0.04 |
| eq_tag_g6_8 | 0.247 | 0.2 | 0.33 | 0.386 | 0.055 | 4.9 | 0.715 | 0.414 | 0.057 |
| nit | 0.054 | 0.051 | 0.063 | 0.059 | 0.049 | 3.88 | 0.914 | 0.731 | 0.014 |
