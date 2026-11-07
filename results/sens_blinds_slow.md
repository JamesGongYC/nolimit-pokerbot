# Leaderboard (provisional-blinds_slow, 2000 hold-out tables, payouts [50, 30, 20, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag_g9_mean | heuristic | 25.55 | 0.82 | +0.00 (0.00) | 23.64 | 33/21/16 | 20/70/14396 | 0/355/0 |  | 1955.4 |
| 2 | heur_tag_g9_mean_g6_7 | heuristic | 25.05 | 0.87 | -0.50 (0.92) | 21.36 | 36/18/12 | 20/70/30482 | 0/392/0 |  | 2317.4 |
| 3 | mix_g5_4 | randomized mixture | 24.47 | 0.85 | -1.08 (0.94) | 21.03 | 33/20/13 | 20/80/15902 | 0/452/0 |  | 610.3 |
| 4 | heur_tag | heuristic | 22.43 | 0.79 | -3.12 (0.99) | 17.03 | 22/26/18 | 20/60/11200 | 0/276/0 |  | - |
| 5 | linear_default_g9_mean | learned (CEM) | 21.60 | 0.73 | -3.95 (1.03) | 14.69 | 16/31/20 | 30/100/13394 | 0/435/0 |  | 1916.7 |
| 6 | eq_tag_g6_8 | statistical+oppmodel | 19.77 | 0.80 | -5.78 (1.00) | 13.15 | 19/22/18 | 20/100/8694 | 0/289/0 |  | 2181.6 |
| 7 | nit | baseline | 19.04 | 0.67 | -6.50 (0.96) | 14.86 | 8/32/23 | 10/30/4668 | 0/21/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| heur_tag_g9_mean | 24.03 | 25.64 | 23.64 | 27.24 | 27.19 |
| heur_tag_g9_mean_g6_7 | 24.33 | 25.20 | 21.36 | 26.08 | 28.26 |
| mix_g5_4 | 24.28 | 24.76 | 21.03 | 24.86 | 27.41 |
| heur_tag | 22.73 | 24.00 | 17.03 | 23.83 | 24.56 |
| linear_default_g9_mean | 21.54 | 24.70 | 14.69 | 23.06 | 23.99 |
| eq_tag_g6_8 | 19.37 | 22.37 | 13.15 | 21.25 | 22.71 |
| nit | 17.85 | 21.55 | 14.86 | 21.15 | 19.81 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag_g9_mean | 0.339 | 0.295 | 0.505 | 0.549 | 0.038 | 3.21 | 0.604 | 0.312 | 0.019 |
| heur_tag_g9_mean_g6_7 | 0.368 | 0.33 | 0.569 | 0.622 | 0.034 | 3.71 | 0.578 | 0.277 | 0.021 |
| mix_g5_4 | 0.34 | 0.309 | 0.521 | 0.584 | 0.054 | 4.1 | 0.609 | 0.281 | 0.02 |
| heur_tag | 0.216 | 0.15 | 0.238 | 0.272 | 0.051 | 2.14 | 0.735 | 0.33 | 0.021 |
| linear_default_g9_mean | 0.237 | 0.236 | 0.343 | 0.458 | 0.138 | 33.34 | 0.752 | 0.447 | 0.015 |
| eq_tag_g6_8 | 0.227 | 0.186 | 0.298 | 0.343 | 0.059 | 5.25 | 0.736 | 0.339 | 0.029 |
| nit | 0.054 | 0.052 | 0.063 | 0.059 | 0.052 | 4.28 | 0.913 | 0.691 | 0.011 |
