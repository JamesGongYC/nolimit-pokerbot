# Leaderboard (provisional-blinds_fast, 2000 hold-out tables, payouts [50, 30, 20, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag_g9_mean | heuristic | 24.20 | 0.90 | +0.00 (0.00) | 19.73 | 32/19/14 | 20/70/8367 | 0/227/0 |  | 1955.4 |
| 2 | heur_tag_g9_mean_g6_7 | heuristic | 23.74 | 0.94 | -0.46 (0.99) | 19.53 | 33/15/12 | 20/70/10042 | 0/224/0 |  | 2317.4 |
| 3 | mix_g5_4 | randomized mixture | 23.57 | 0.92 | -0.62 (1.05) | 18.92 | 32/16/14 | 30/70/7272 | 0/250/0 |  | 610.3 |
| 4 | linear_default_g9_mean | learned (CEM) | 21.82 | 0.82 | -2.37 (1.14) | 15.13 | 22/23/20 | 30/80/8819 | 0/296/0 |  | 1916.7 |
| 5 | heur_tag | heuristic | 20.71 | 0.82 | -3.48 (1.06) | 14.48 | 20/24/18 | 20/70/21868 | 0/222/0 |  | - |
| 6 | eq_tag_g6_8 | statistical+oppmodel | 20.44 | 0.88 | -3.76 (1.10) | 12.58 | 24/17/16 | 20/100/8553 | 0/228/0 |  | 2181.6 |
| 7 | nit | baseline | 16.41 | 0.61 | -7.78 (1.00) | 11.65 | 2/32/27 | 10/30/7973 | 0/26/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| heur_tag_g9_mean | 19.73 | 23.12 | 21.07 | 26.78 | 30.28 |
| heur_tag_g9_mean_g6_7 | 20.43 | 21.90 | 19.53 | 26.30 | 30.55 |
| mix_g5_4 | 18.92 | 24.44 | 19.33 | 25.01 | 30.16 |
| linear_default_g9_mean | 19.66 | 24.45 | 15.13 | 23.86 | 26.02 |
| heur_tag | 18.87 | 22.17 | 14.48 | 23.18 | 24.86 |
| eq_tag_g6_8 | 17.95 | 22.32 | 12.58 | 23.32 | 26.02 |
| nit | 14.25 | 20.02 | 11.65 | 18.88 | 17.27 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag_g9_mean | 0.337 | 0.293 | 0.51 | 0.567 | 0.032 | 3.36 | 0.613 | 0.374 | 0.048 |
| heur_tag_g9_mean_g6_7 | 0.364 | 0.322 | 0.563 | 0.626 | 0.03 | 3.71 | 0.588 | 0.345 | 0.054 |
| mix_g5_4 | 0.337 | 0.301 | 0.516 | 0.594 | 0.049 | 4.18 | 0.619 | 0.361 | 0.048 |
| linear_default_g9_mean | 0.246 | 0.242 | 0.361 | 0.502 | 0.128 | 20.99 | 0.743 | 0.58 | 0.025 |
| heur_tag | 0.227 | 0.162 | 0.265 | 0.318 | 0.046 | 2.33 | 0.732 | 0.402 | 0.054 |
| eq_tag_g6_8 | 0.25 | 0.2 | 0.332 | 0.4 | 0.051 | 4.68 | 0.717 | 0.432 | 0.074 |
| nit | 0.054 | 0.05 | 0.063 | 0.059 | 0.046 | 3.79 | 0.916 | 0.748 | 0.015 |
