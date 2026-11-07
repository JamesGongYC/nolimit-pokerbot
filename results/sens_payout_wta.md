# Leaderboard (provisional-payout_wta, 2000 hold-out tables, payouts [100, 0, 0, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag_g9_mean_g6_7 | heuristic | 34.10 | 2.05 | +0.00 (0.00) | 18.74 | 35/17/13 | 20/50/7460 | 0/175/0 |  | 2317.4 |
| 2 | heur_tag_g9_mean | heuristic | 33.12 | 2.01 | -0.98 (2.28) | 17.94 | 34/18/13 | 20/50/11671 | 0/176/0 |  | 1955.4 |
| 3 | mix_g5_4 | randomized mixture | 30.82 | 1.98 | -3.28 (2.09) | 19.10 | 31/18/15 | 20/60/10749 | 0/206/0 |  | 610.3 |
| 4 | heur_tag | heuristic | 21.13 | 1.70 | -12.97 (2.38) | 13.74 | 21/26/18 | 20/50/6490 | 0/164/0 |  | - |
| 5 | eq_tag_g6_8 | statistical+oppmodel | 20.52 | 1.74 | -13.57 (2.41) | 14.48 | 21/20/17 | 20/80/6705 | 0/185/0 |  | 2181.6 |
| 6 | linear_default_g9_mean | learned (CEM) | 20.07 | 1.68 | -14.03 (2.47) | 12.14 | 21/26/20 | 30/60/8598 | 0/177/0 |  | 1916.7 |
| 7 | nit | baseline | 5.57 | 0.90 | -28.53 (2.16) | 0.59 | 5/43/22 | 10/30/6093 | 0/28/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| heur_tag_g9_mean_g6_7 | 28.49 | 18.74 | 32.30 | 35.88 | 55.10 |
| heur_tag_g9_mean | 27.27 | 17.94 | 35.21 | 34.61 | 50.57 |
| mix_g5_4 | 22.09 | 19.10 | 27.96 | 33.05 | 51.89 |
| heur_tag | 17.80 | 13.74 | 16.50 | 27.55 | 30.08 |
| eq_tag_g6_8 | 15.77 | 16.28 | 14.48 | 23.96 | 32.13 |
| linear_default_g9_mean | 12.14 | 13.94 | 13.56 | 23.92 | 36.79 |
| nit | 0.59 | 6.14 | 2.43 | 9.72 | 8.95 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag_g9_mean_g6_7 | 0.38 | 0.338 | 0.592 | 0.646 | 0.034 | 3.68 | 0.566 | 0.328 | 0.037 |
| heur_tag_g9_mean | 0.356 | 0.307 | 0.534 | 0.581 | 0.039 | 3.26 | 0.59 | 0.357 | 0.032 |
| mix_g5_4 | 0.352 | 0.317 | 0.544 | 0.616 | 0.054 | 4.15 | 0.598 | 0.342 | 0.034 |
| heur_tag | 0.233 | 0.16 | 0.261 | 0.302 | 0.052 | 2.13 | 0.717 | 0.382 | 0.036 |
| eq_tag_g6_8 | 0.241 | 0.195 | 0.321 | 0.377 | 0.056 | 4.83 | 0.719 | 0.434 | 0.048 |
| linear_default_g9_mean | 0.262 | 0.258 | 0.382 | 0.514 | 0.148 | 17.52 | 0.72 | 0.553 | 0.022 |
| nit | 0.053 | 0.049 | 0.062 | 0.057 | 0.049 | 3.84 | 0.915 | 0.735 | 0.014 |
