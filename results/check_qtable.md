# Leaderboard (sequoia-nolimit-2026-provisional, 1500 hold-out tables, payouts [50, 30, 20, 0, 0, 0, 0, 0])

Points = mean official points per table (one candidate seat, 7 hold-out opponents); CI = 95%% over independent tables; vs leader = paired difference on identical tables.

| # | strategy | mechanism | points | ±95% | vs leader (±) | worst field | 1st/2nd/3rd % | p50/p99/max µs | err/timeout/illegal | mem peak KB | train CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heur_tag | heuristic | 20.92 | 0.89 | +0.00 (0.00) | 14.56 | 17/28/21 | 20/70/11941 | 0/328/0 |  | - |
| 2 | lookup_pf | lookup-table (rollout-trained) | 18.57 | 0.88 | -2.35 (1.14) | 13.48 | 13/27/19 | 20/100/10271 | 0/304/0 |  | - |
| 3 | hybrid_q | hybrid (CFR push/fold + table + equity + oppmodel) | 18.22 | 0.88 | -2.70 (1.13) | 13.75 | 14/26/18 | 20/100/10499 | 0/237/0 |  | - |
| 4 | hybrid_tag | hybrid (CFR push/fold + table + equity + oppmodel) | 15.72 | 0.93 | -5.20 (1.09) | 9.78 | 15/16/16 | 30/100/11815 | 0/334/0 |  | - |
| 5 | eq_default | statistical+oppmodel | 14.60 | 0.93 | -6.32 (1.13) | 9.13 | 15/14/16 | 30/100/10888 | 0/330/0 |  | - |

## Field sensitivity (mean points by hold-out field)

| strategy | aggressive | balanced | exploitative | mixed | passive |
|---|---|---|---|---|---|
| heur_tag | 23.19 | 14.56 | 22.01 | 22.26 | 22.57 |
| lookup_pf | 22.62 | 13.48 | 18.06 | 19.72 | 18.96 |
| hybrid_q | 23.14 | 13.75 | 15.73 | 18.43 | 20.04 |
| hybrid_tag | 21.17 | 9.78 | 14.07 | 16.69 | 16.90 |
| eq_default | 19.63 | 9.13 | 12.37 | 15.68 | 16.18 |

## Behaviour (verifies styles differ)

| strategy | vpip | pfr | open | steal | 3bet | af | fold_to_bet | wtsd | allin_per_hand |
|---|---|---|---|---|---|---|---|---|---|
| heur_tag | 0.22 | 0.151 | 0.245 | 0.297 | 0.05 | 2.13 | 0.741 | 0.323 | 0.042 |
| lookup_pf | 0.171 | 0.122 | 0.157 | 0.205 | 0.109 | 3.27 | 0.785 | 0.411 | 0.04 |
| hybrid_q | 0.168 | 0.123 | 0.158 | 0.207 | 0.109 | 3.42 | 0.787 | 0.405 | 0.045 |
| hybrid_tag | 0.209 | 0.146 | 0.237 | 0.291 | 0.047 | 3.71 | 0.728 | 0.362 | 0.054 |
| eq_default | 0.317 | 0.204 | 0.337 | 0.395 | 0.06 | 3.16 | 0.617 | 0.339 | 0.065 |
