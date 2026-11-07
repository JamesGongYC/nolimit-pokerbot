# Results — round 3: all strategy families head to head

30 strategies: every tuned result (`T-*`) plus untuned presets and
baselines. Full tables: `results/grand_holdout.md`, `results/grand_official.md`,
`results/grand_crossplay.md`. All numbers are under the **provisional** event
config against **synthetic** fields.

| strategy | mechanism | hold-out pts (±95%) | official format, weighted (±) | cross-play mean vs others |
|---|---|---|---|---|
| **T-gbdt** | GBDT Q-trees, preflop override of tuned heuristic | **26.26 (0.76)** | **463 (8)** | **25.5** |
| T-rf | random-forest Q-trees, same scheme | 26.22 (0.73) | 391 (7) | 23.3 |
| T-heur | CEM-tuned heuristic | 24.92 (0.73) | 345 (7) | 21.5 |
| T-heur2 | CEM-tuned heuristic, round 2 | 24.48 (0.76) | 379 (8) | 21.5 |
| T-mixture | CEM mixture | 23.70 (0.75) | 376 (7) | 21.4 |
| T-heur_league | heuristic tuned vs a league of strong bots | 23.47 (0.77) | 400 (8) | 22.3 |
| T-linear | CEM linear policy | 22.36 (0.66) | 399 (7) | 21.8 |
| T-eq2 | tuned EV/equity | 20.42 (0.70) | 364 (8) | 20.0 |
| T-mc | tuned real-time Monte Carlo | 19.98 (0.68) | 339 (7) | 19.5 |
| T-jam | CEM-tuned jam/fold | 19.05 (0.49) | 275 (5) | 16.1 |
| jam_stack | jam top 20% scaled by stack depth | 18.98 (0.55) | 295 (6) | 17.8 |
| jam_nash | push/fold equilibrium at any depth | 15.31 (0.66) | 282 (7) | 15.9 |
| jam_always | jam every hand | 3.53 (0.41) | 52 (4) | 4.3 |
| potodds | naive pot-odds caller | 8.39 (0.52) | 102 (5) | 8.0 |

(12.5 = average seat. Hold-out = 3,000 matched tables per strategy vs unseen
fields. Official = 30 entries, random tables + fillers, 6 × 2,000
iterations, T1..T6 weighted 1–6. Cross-play = one bot vs 7 copies of each
other strategy, 40 tables per cell.)

**Deployed:** compact GBDT (`results/gbdt_variants/c40_4.json`: 40 trees,
depth 4 per action) in `submission/`. Hold-out it is tied with the full model
(26.10 vs 26.12) and 1.1 ± 0.7 above the tuned heuristic. Benchmark through
the exported interface on a loaded machine: p50 100 µs, p99 260 µs,
1 of 15,282 decisions over 1 ms (a scheduler spike; the heuristic fallback in
`submission_heuristic/` had p99 170 µs and max 818 µs in the same run).
0.66 MB.

## Findings from round 3

1. **Tree-based policy improvement works preflop, not postflop.** Fitted
   policy iteration (collect around the tuned heuristic with one random
   deviation per hand, fit per-action GBDT/RF regressors of the hand's ICM
   change, override only when the predicted gain is > 0.25 points) gained
   +1.1 to +1.3 on hold-out once the trees were limited to preflop. Letting the
   trees act on every street lost 4–13 points: postflop single-hand rewards
   are too noisy (validation R² 0.05–0.16 on ~1M samples per round).
   Iterating helped. GBDT preflop scored 24.98 after round 0 and 27.19 after
   round 2 on the same training tables.
2. **The Monte Carlo bot is not better than table lookups under 1 ms.**
   Python fits only about 30 samples per decision (median 32 at 250 µs), so the
   estimate is barely sharper than the bucket table, and the EV layer limits
   it, as with the equity bot. Tuning took it from 15.9 to 20.0 points.
3. **Jam bots: naive versions lose, tuned versions are respectable.**
   Jam-everything scores 3.5. Jamming the top 50% scores 7.1. Stack-scaled or
   CEM-tuned jam/fold reaches about 19, which beats the untuned equity bots.
   That is notable for a policy with no postflop play at all.
4. **The strong bots form a rock-paper-scissors cycle** (cross-play, ±~3 per
   cell). A lone Nash jammer scores 22.2 at a table of seven GBDTs (the tuned
   bots fold too much to relentless jams). A lone linear bot scores 18.9
   against seven GBDTs, while one GBDT among seven linear bots scores 6.4.
   The replicator "meta" of this matrix mixes jam_nash (47 %), GBDT (31 %)
   and linear (19 %). Practical reading: GBDT is best in mixed fields, but if
   the real field turns out to be full of strong tight bots, jam-heavy play
   becomes more valuable.
5. **League (self-play) tuning trades field-specific gains for robustness.**
   Tuned against strong bots only, the heuristic went from 11.8 to 16.1 in that
   league. It ranks 2nd in the all-vs-all official format (400) but 6th on the
   weak-ish hold-out fields (23.5). Which version is better depends on how
   strong the real field is.

Training cost (CPU s): GBDT 3 rounds 2,468 · RF 2 rounds 1,589 · jam CEM
1,398 · MC CEM 1,084 · league CEM 3,465. The tree data (~1M rows per round)
is in `results/treedata/` (549 MB, safe to delete unless refitting).

---

# Results — rounds 1–2 (2026-11-05, provisional event config)

Everything here assumes the **provisional** event config (NLHE, payouts
50/30/20, our guessed blind schedule) and **synthetic** opponent fields. The
numbers rank strategies against those fields. They do not forecast results
against the real competitors. Full tables: `results/leaderboard_holdout.md`,
`results/sensitivity.md`, `results/official_sim.md`.

## Deployed pick: tuned heuristic `results/opt_heur/best.json`

Exported to `submission/` (0.48 MB). Through the exported `bot.Agent` path,
single process: 14,817 decisions, p50 50 µs, p99 150 µs, max 279 µs, 0 over
1 ms, 0 errors, 0 illegal actions, import 31 ms, RSS 18 MB.

### Hold-out leaderboard (4,000 matched tables per strategy, unseen opponent configs + seeds)

Points = mean official points per table (100 per table are awarded, so 12.5
is "average"). ± = 95 % CI over tables. "Gap" = paired difference to the
leader on identical tables.

| strategy | mechanism | points | ± | gap (±) | worst field | win % |
|---|---|---|---|---|---|---|
| **heur tuned (CEM r1)** | heuristic | **24.97** | 0.64 | — | 20.72 | 34 |
| heur tuned (CEM r2) | heuristic | 24.81 | 0.66 | −0.16 (0.70) | 20.28 | 37 |
| mixture (CEM weights) | mixture | 23.72 | 0.65 | −1.25 (0.73) | 18.14 | 33 |
| heur tuned (random search) | heuristic | 23.43 | 0.62 | −1.54 (0.77) | 18.35 | 29 |
| hybrid, heuristic postflop (CEM) | hybrid | 22.49 | 0.55 | −2.48 (0.78) | 16.72 | 21 |
| linear policy (CEM) | learned | 22.42 | 0.57 | −2.55 (0.81) | 15.76 | 24 |
| heur_tag (untuned) | heuristic | 21.71 | 0.57 | −3.26 (0.75) | 16.84 | 21 |
| equity (CEM r2) | statistical | 20.24 | 0.61 | −4.73 (0.77) | 13.46 | 24 |
| hybrid, equity postflop + lookup (CEM) | hybrid | 18.93 | 0.56 | −6.04 (0.79) | 14.05 | 17 |
| lookup_pf | lookup table | 18.51 | 0.55 | −6.46 (0.79) | 13.57 | 14 |
| eq_tag (untuned) | statistical | 17.40 | 0.58 | −7.57 (0.77) | 10.61 | 17 |
| nit | baseline | 17.06 | 0.44 | −7.91 (0.72) | 11.97 | 4 |

The top two are statistically tied. We deploy round 1: it has the best mean,
is tied on worst field, and went through fewer search rounds, so it is less
likely to be overfit to the training fields. Round 2 gained +1.2 on training
fields but nothing on hold-out.

### What the tuned strategy does (vs the untuned TAG preset)

| knob | untuned | tuned | reading |
|---|---|---|---|
| open range (first position → button + steal) | 11 % → 35 % | 35 % → 67 % | open far wider; the field over-folds preflop |
| open size | 2.5 BB | 2.4 BB | small opens |
| value 3-bet range | 3.5 % | 2.1 % | 3-bet only premiums |
| bluff / c-bet rate | 0.25 / 0.80 | 0.33 / 0.74 | keep firing at passive fields |
| call class vs small / big bets | mid pair / top pair good | overcards / **overpair** | float small bets, respect big ones |
| raise class (vs bets) | two pair | **set+** | raise only near-nuts |
| call margin over pot odds | +6 % | +9.5 % | tighter calling |
| push/fold threshold | 10 BB | 9.1 BB | equilibrium push/fold below ~9 BB |
| endgame shift | 0 | 0.18 | a bit more aggression near hand 200 when short |

Behaviour (hold-out): VPIP 34 %, PFR 30 %, steal 56 %, 3-bet 3.5 %, AF 3.3,
fold-to-bet 61 %, WTSD 35 %. A loose-aggressive preflop style combined with
disciplined showdown decisions.

### Sensitivity (hold-out fields, 2,000 tables each)

The tuned heuristics stay first under every alternative tested:
winner-take-all, 65/35, top-4 (40/30/20/10), linear 8..1, faster blinds,
slower blinds. Margins shrink with linear payouts, where survival dominates
and all decent strategies bunch at about 5.6–5.9 points.

### Official format (36 entries: the pick + a 35-bot synthetic regional field)

6 tournaments × 2,000 iterations, random tables, filler seats score nothing:
the pick scored **509 ± 8** weighted points (24.1 per table, 34 % table
wins). Next best were untuned heuristic presets at about 431.

## Findings

1. **Tuning matters more than mechanism choice against these fields.** CEM
   lifted the heuristic from 21.7 to 25.0 on hold-out (+3.3 paired) and the
   equity bot from 17.4 to 20.2. CEM (1,955 CPU s) beat random search
   (666 CPU s) by +1.5 points on hold-out.
2. **Rule-based postflop beat the EV model.** Both share the same preflop
   code. The equity bot's EV approximations (fold probability, equity when
   called) are miscalibrated against bots that call a lot. With
   CEM-calibrated EV constants it still plateaued about 4.7 points below.
   Hand-class thresholds are more robust here.
3. **Opponent-model bug found and fixed.** Mixing preflop folds into
   "fold to bet" inflated postflop fold equity and caused over-bluffing. The
   model now keeps a separate postflop rate.
4. **Lookup tables do not transfer across postflop policies.** The
   rollout-learned preflop table helped the equity bot (+4 on training
   fields), but put in front of heuristic postflop play it *hurt*
   (25.4 → 21.5). Preflop action values depend on how you play later, so
   retrain the table with the deployed postflop policy before combining them.
5. **Mixtures did not beat the best component** against fixed fields (−1.25).
   They can only pay off against adaptive opponents that track your
   tendencies within a table, and these fields do not.
6. **Overfitting is real even with matched seeds.** Round-2 gains on training
   fields vanished on hold-out. Keep the hold-out fields untouched and pick
   finalists on fresh seeds.

## Training cost (CPU seconds on ~6 effective cores; wall clock in parentheses)

equity tables 160 s wall · push/fold tables 38 s wall · lookup table 253 (41) ·
heuristic CEM r1 1,955 (302) · r2 2,317 (509) · random search 666 (103) ·
equity CEM 1,722 (267) / r2 2,182 (363) · hybrid CEM 1,627 (253) / 1,903 (320) ·
linear CEM 1,917 (296) · mixture CEM 610.

## Event-day checklist (only if pre-written code is allowed; see README)

1. Copy the starter kit's payouts, blind schedule and variant into a new event JSON.
2. `precompute` (≈3 min) and `pushfold` (≈40 s) for the announced variant.
3. Map the official observation/action format in `submission/bot.py:to_internal/from_internal`.
4. Re-tune for 5–10 minutes: `optimize --event new.json --base results/opt_heur/best.json --generations 6 --sigma 0.1`.
5. `export` + `bench`, then submit. Submit early; resubmissions are allowed every 5 minutes.
