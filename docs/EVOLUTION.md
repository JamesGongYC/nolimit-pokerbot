# Evolutionary population training (`evolve` / `evo-verdict`)

This replaces the hold-out paradigm for strategy discovery. Code is in
`pokerbot/evolve.py`. Run it with:

```sh
python3 -m pokerbot.cli --workers 7 evolve --name main --generations 60 --minutes 60 \
    --pop 6 --tables 100 --duel-tables 12 --progress-tables 40 --retrain-every 10 --retrain-tables 3000
python3 -m pokerbot.cli --workers 7 evo-verdict --run results/evo_main --iterations 300 --xtables 1
```

State is saved every generation (`results/evo_<name>/state.json`).
Rerunning the same `evolve` command resumes where it stopped.

## Decisions

| # | Decision | Chosen |
|---|---|---|
| 1 | Opponent sources | 50 % hall of fame, 50 % generator; **no current peers** |
| 2 | Hall-of-fame selection | the **top cycle** (Smith set) of the hall-of-fame win graph, as a group; seats drawn uniformly from it |
| 3 | Table make-up | mixed tables: 1 candidate + 7 opponents, alternating **4 HoF / 3 generator** and **3 / 4** |
| 4 | Fitness | **plain mean** official points (all families share identical tables in a generation) |
| 5 | Progress | vs the **previous generation's bests**; frozen anchors (generation-0 champions + baselines, fixed seeds) are also logged |
| 6 | Budget | sized so 50+ generations fit in about 1 hour |
| 7 | Promotion | **every generation's best, one per family** |
| 8 | Verdict | none during training; **final verdict = the whole hall of fame vs itself**: all-vs-all official format + 1-vs-7 cross-play |
| 9–10 | Leakage policy, stopping | unchanged: fixed generation count, with a wall-clock cap |
| — | Equilibrium update | new members duel (4 v 4) **the current group and each other** only |
| — | Tree families | settings evolve every generation; models retrained every 10 generations in the reference run; **every 8 at the event** |
| — | Refit gate (2026-11-06) | a retrained model is adopted only if the family's current mean spec scores higher with it than with the old model on 200 matched pool tables |
| — | Event verdict (2026-11-06) | a **shortlist** verdict is allowed at the event (top members per family + the active bot); full-HoF verdict otherwise |
| — | Families (2026-11-06) | all **8** families at the event: heuristic, equity, montecarlo, linear, allin, hybrid, gbdt, rf |

## One generation

1. **Opponent pool.**
   - The generator draws a fresh random spec per seat. It picks uniformly among the 8 families plus 6 baseline bots, with uniform parameters inside each search space.
   - The hall-of-fame seats come uniformly from the current top cycle.
2. **Candidates.** Each family (heuristic, equity, montecarlo, linear, allin, hybrid, gbdt, rf) samples `pop` vectors from its CEM distribution, one of them being the mean.
3. **Evaluation.** All candidates of all families play the same `tables` pool tables.
4. **Update and promotion.**
   - Each family's CEM mean and std are refit on its top third.
   - Its best candidate is frozen into the hall of fame as `<family>_g<gen>`.
5. **Equilibrium.**
   - Every new member plays a 4 v 4 duel against each current top-cycle member, and against the other new members.
   - "A beats B" means A's seats scored more in total than B's.
   - The new group is the union of source strongly connected components of the win graph over (old group + new members). Members that drop out stay in the hall of fame but are no longer drawn or dueled.
6. **Progress.**
   - The new bests play tables of the previous generation's bests (fresh seeds each generation).
   - They also play tables of frozen anchors (identical seeds every generation, so this column is comparable over time).
7. **Every 10th generation:**
   - collect exploring data around the latest gbdt champion, drawn from the current pool;
   - refit GBDT (40 trees, depth 4) and a random forest (30 trees, depth 7) on the last two collections;
   - point both tree families at the new models. Earlier hall-of-fame tree bots keep their own model files.

## Reading the outputs

- `results/evo_<name>/summary.md` lists, per generation: wall time, mean fitness, vs previous bests, vs anchors, and the members of the top cycle.
- `results/evo_<name>/verdict.md` holds the final ranking of every hall-of-fame player: official weighted score, cross-play mean, worst matchup, and replicator meta share. It also aggregates by family and by generation band.

## Limits of this paradigm

- **Fitness noise.** About 100 tables per candidate gives about ±2 points (one standard error). Selection within a generation is noisy, and progress relies on many generations.
- **"Vs previous bests" is relative.** In a cyclic game it can rise while the population circles. The frozen-anchor column, and the generation-band table in the verdict, are the checks.
- **The top cycle grows with noise.** Noisy duels create spurious cycles, which enlarge the group and the per-generation duel cost. More duel tables shrink it.
- **No holdout by design.** The verdict ranks hall-of-fame players against each other only. It says nothing about weaker or unrelated fields.

## Run `evo_k16` (2026-11-06): 60 generations, top cycle capped at 16

The uncapped top cycle grew to cover nearly the whole hall of fame (207 of 208
members, 226 s per generation). The group is therefore capped at **K = 16**,
keeping the members with the highest Copeland score inside the cycle
(`--group-cap 16`). Long runs need `caffeinate -i` on macOS: a
sleep or nice stall stretched one generation to 1,466 s wall. The uncapped run
is kept in `results/evo_main` (25 generations) for comparison.

**Training trajectory** (`results/evo_k16/summary.md`):
- The aggregate score vs frozen anchors stayed flat (about 16.8–17.5). The score vs previous bests stayed about at parity.
- Per family, vs anchors (first 10 → last 10 generations): gbdt +3.1, MC +1.4, hybrid +1.4, allin +1.3, equity −0.7, linear −1.2, rf −1.4, heuristic −2.9.
- Top-cycle membership over the run: MC 41 %, gbdt 22 %, equity 21 %, others ≤ 7 %.

**Final verdict** (`results/evo_k16/verdict*.md`: 488 players, 300 × 6 official
iterations, 1-table cross-play cells). ± on the official weighted score is about 19,
so the top ~20 entries are statistically tied.

| family | gen 0–10 | 11–20 | 21–30 | 31–40 | 41–50 | 51–60 |
|---|---|---|---|---|---|---|
| gbdt | 9.96 | 12.71 | 14.29 | **16.54** | **16.51** | 15.60 |
| rf | 10.52 | 13.08 | **15.77** | 14.10 | 12.59 | 12.24 |
| montecarlo | 12.79 | 13.17 | 14.20 | 13.84 | 13.89 | 13.79 |
| equity | 12.98 | 13.19 | 13.47 | 13.41 | 13.86 | 13.45 |
| linear | 10.82 | 12.29 | 11.85 | 11.72 | 11.38 | 11.96 |
| heuristic | 10.94 | 11.54 | 11.63 | 11.75 | 11.93 | 11.91 |
| hybrid | 10.70 | 11.48 | 11.68 | 11.73 | 11.65 | 12.04 |
| allin | 10.92 | 11.23 | 10.94 | 10.80 | 11.11 | 11.07 |

(Official mean points per table in the all-vs-all field; 12.5 = average seat.)

Readings:
1. **Evolution produced real absolute gains against the hall-of-fame field.**
   - Every family except allin improved from band 0–10 to its best band.
   - The gbdt family gained +6.6 per table. The generation-0 champions rank low in this field (gbdt_g0 347th, heuristic_g0 398th, MC_g0 163rd).
   - The flat anchor curve during training hid this: the anchors are a weak field where everyone already scores well.
2. **Tree families gain mostly through retraining.**
   - gbdt peaks in generations 31–50; rf peaks in 21–30, then declines after later retrains.
   - The data collected around a stronger pool does not always give a better model, so the later rf refits regressed.
3. **The training-time top cycle disagrees with the verdict.**
   - The final K = 16 group is mostly MC and equity bots, ranked 44th–148th in the verdict; its best member is gbdt_g54 at 27th.
   - The 4 v 4 duels reward bots that beat one specific opponent type. The mixed all-vs-all verdict rewards breadth.
   - The group therefore works as an opponent pool, not as a pick list.
4. **Cross-play with 1 table per cell is too noisy for "worst" and "meta".**
   - Cells are single outcomes (0/20/30/50), so every "worst" is 0 and the replicator converged on an arbitrary player.
   - Row means (≈487 cells each) are usable: the tree bots lead at about 17–18.8.
   - Use `--xtables ≥ 10` on a shortlist if meta shares are needed.

**Pick from this run:** gbdt_g36–g38 / g47 / g49 (official 362–368, tied). They
share the compact 40×4 model size, so export latency should match the deployed
c40_4 bot; benchmark before swapping.
