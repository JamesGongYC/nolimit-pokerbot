# Leaderboard (provisional-payout_top4, 2000 hold-out tables, payouts [40, 30, 20, 10, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mix_g5_4 | randomized mixture | 22.16 | 0.69 | +0.00 (0.00) | 17.73 | 34/14/17 | 30/200/9352 | 0/407/0 |  | 610.3 |
| 2 | heur_tag_g9_mean | heuristic | 22.09 | 0.69 | -0.06 (0.77) | 18.35 | 33/16/15 | 20/200/13844 | 0/400/0 |  | 1955.4 |
| 3 | heur_tag_g9_mean_g6_7 | heuristic | 22.08 | 0.72 | -0.07 (0.68) | 17.90 | 37/12/13 | 20/200/10058 | 0/393/0 |  | 2317.4 |
| 4 | linear_default_g9_mean | learned (CEM) | 21.25 | 0.62 | -0.91 (0.81) | 15.76 | 22/24/20 | 40/210/11258 | 0/525/0 |  | 1916.7 |
| 5 | heur_tag | heuristic | 19.93 | 0.63 | -2.22 (0.83) | 14.88 | 18/25/19 | 20/200/8607 | 0/355/0 |  | - |
| 6 | eq_tag_g6_8 | statistical+oppmodel | 19.53 | 0.67 | -2.63 (0.84) | 13.37 | 24/18/16 | 20/200/13774 | 0/377/0 |  | 2181.6 |
| 7 | nit | baseline | 16.90 | 0.55 | -5.25 (0.79) | 12.29 | 4/28/26 | 10/30/4449 | 0/19/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | ho_adaptive | ho_aggressive | ho_balanced | ho_mixed | ho_passive |
|---|---|---|---|---|---|
| mix_g5_4 | 20.33 | 22.85 | 17.73 | 24.17 | 25.70 |
| heur_tag_g9_mean | 19.10 | 22.70 | 18.35 | 24.01 | 26.30 |
| heur_tag_g9_mean_g6_7 | 19.33 | 23.26 | 17.90 | 23.94 | 25.97 |
| linear_default_g9_mean | 19.89 | 24.03 | 15.76 | 23.03 | 23.53 |
| heur_tag | 17.68 | 22.66 | 14.88 | 22.07 | 22.38 |
| eq_tag_g6_8 | 16.91 | 21.66 | 13.37 | 21.41 | 24.30 |
| nit | 14.27 | 20.47 | 12.29 | 19.02 | 18.46 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| mix_g5_4 | 0.333 | 0.299 | 0.509 | 0.587 | 0.049 | 4.16 | 0.621 | 0.33 | 0.042 |
| heur_tag_g9_mean | 0.33 | 0.289 | 0.498 | 0.556 | 0.034 | 3.42 | 0.618 | 0.352 | 0.04 |
| heur_tag_g9_mean_g6_7 | 0.361 | 0.322 | 0.559 | 0.624 | 0.031 | 3.83 | 0.59 | 0.321 | 0.046 |
| linear_default_g9_mean | 0.239 | 0.236 | 0.348 | 0.488 | 0.132 | 27.66 | 0.751 | 0.535 | 0.021 |
| heur_tag | 0.215 | 0.155 | 0.249 | 0.3 | 0.047 | 2.36 | 0.743 | 0.375 | 0.044 |
| eq_tag_g6_8 | 0.242 | 0.196 | 0.322 | 0.389 | 0.051 | 5.02 | 0.725 | 0.401 | 0.064 |
| nit | 0.054 | 0.051 | 0.062 | 0.059 | 0.048 | 3.74 | 0.915 | 0.73 | 0.014 |
