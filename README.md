# Pokerbot sandbox — Sequoia "No Limit" bot tournament

A local, stdlib-only (Python ≥ 3.9) sandbox for finding strong, tunable
strategies for https://pokerbot.sequoiacap.com/ (rules: `/rules.html`).
Includes a tournament-accurate simulator, a strategy registry covering
several decision mechanisms, an optimizer on official tournament points,
a hold-out leaderboard, and an exporter that writes a fast, self-contained
submission bot.

> **Read before using at the event.** The rules say teams have "60 minutes to
> build a Python bot" (the homepage says "from scratch") and list "copying
> code" among the disqualification triggers. Ask a Gamemaster whether
> pre-written code may be brought. If it can't, use this repo for practice and
> research: the findings in `docs/RESULTS.md` (which mechanisms and parameter
> settings win, push/fold ranges, ICM effects) still carry over to a bot written
> on the day.
>
> The official starter kit (Agent interface, simulator, ICM calculator,
> payouts, blind schedule) and the poker variant are only released at the
> event. Every unknown is a clearly labelled, swappable assumption. See
> [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

## Quick start

```sh
python3 -m pokerbot.cli test                       # 25 unit tests
python3 -m pokerbot.cli precompute --samples 5     # equity lookup tables (~3 min, 10 cores)
python3 -m pokerbot.cli pushfold                   # push/fold equilibrium tables (~40 s)
python3 -m pokerbot.cli train-table --name qtable_pf --generations 6 --tables 1500   # rollout lookup table (~40 s)

# play: 8 presets at one table, seats rotate every table
python3 -m pokerbot.cli simulate --lineup heur_tag,eq_tag,hybrid_q,station,maniac,nit,shover20,random --tables 400 --rotate

# optimize on training fields (CEM or random search; screening -> finalists)
python3 -m pokerbot.cli optimize --base heur_tag --method cem --generations 10 --popsize 16 --screen-tables 300 --final-tables 1500 --out results/opt_heur
./scripts/optimize_all.sh                          # the full search used for docs/RESULTS.md

# hold-out leaderboard (unseen opponent configs + unseen seeds)
python3 -m pokerbot.cli evaluate --candidates results/opt_heur/best.json,heur_tag,eq_tag --tables 3000 --out results/leaderboard.json

# official format: random tables + fillers, 6 tournaments weighted 1..6
python3 -m pokerbot.cli tournament --entries results/opt_heur/best.json,heur_tag,eq_tag,station,maniac,nit,shover20,random,heur_lag,eq_lag --iterations 2000

# tree ensembles (offline venv with scikit-learn; the bot itself stays stdlib-only)
uv venv .venv-train --python 3.9 && uv pip install --python .venv-train/bin/python numpy scikit-learn
python3 -m pokerbot.cli train-trees --base results/opt_heur/best.json --name gbdt --algo gbdt --rounds 3 --tables 10000
./scripts/train_trees.sh; ./scripts/research_round.sh   # GBDT/RF, jam + MC tuning, league tuning

# every strategy against each other: hold-out board, all-vs-all official format, 1-vs-7 cross-play matrix
./scripts/grand_leaderboard.sh
python3 -m pokerbot.cli crossplay --strategies results/opt_gbdt/best.json,results/opt_heur/best.json,jam_nash,mc_tag --tables 40

# export the upload (submission/ + submission.zip with agent.py at the zip root),
# self-test it from an unzipped copy, then benchmark the exact agent.py path
python3 -m pokerbot.cli export --spec results/gbdt_variants/c40_4.json --out submission
python3 -m pokerbot.cli bench --dir submission --tables 60
# upload to the site, wait for its check, select it as the ACTIVE bot
python3 tools/submit.py check submission.zip
python3 tools/submit.py upload submission.zip --activate
```

Global options: `--event <json>` (another event config), `--workers N`.

## Layout

| Path | What |
|---|---|
| `configs/event_provisional.json` | event structure: table size, stacks, blinds, payouts, variant, time limit (**replace with starter-kit values**) |
| `configs/agents/*.json` | named agent presets: mechanism type + style presets + parameter overrides |
| `configs/populations/train.json` / `holdout.json` | opponent fields for tuning / for final evaluation only |
| `pokerbot/engine.py` | table engine: blinds, side pots, NL/PL/FL betting, min-raise/reopen rules, bust placement, ICM at the cap, timing, legal-action fallback, behaviour stats |
| `pokerbot/variant.py` | replaceable variant adapter (deck, hole cards, streets, betting, hand ranking) |
| `pokerbot/scoring.py` | placement points + exact ICM |
| `pokerbot/sim.py` | parallel runner, matched-lineup evaluation, official-format tournament, metrics |
| `pokerbot/features.py` + `precompute.py` | fast hand features and Monte Carlo lookup tables |
| `pokerbot/pfsolve.py` / `pushfold.py` | push/fold fictitious-play solver / runtime lookup |
| `pokerbot/strategies/` | the policies (below) + registry |
| `pokerbot/optimize.py` | random search, CEM, mixture-weight CEM |
| `pokerbot/search.py` | Monte Carlo control → lookup table |
| `pokerbot/evaluate.py` | leaderboard + official-format scoring |
| `pokerbot/export.py` | upload writer (`agent.py` + zip), format checks, unzipped selftest, benchmark |
| `tools/submit.py` | upload / list / activate via the site API (an upload must be selected as the active bot) |

## Strategy mechanisms (`type` in a spec)

| type | mechanism |
|---|---|
| `heuristic` | rules on percentile ranges, position, made-hand/draw classes, bet size, stacks, ICM premium |
| `equity` | statistical: table equity vs estimated ranges, pot odds, opponent-model fold rates → approximate action EVs |
| `lookup` | per-state action values from a table learned by exploring rollouts (`train-table`); falls back to `equity` |
| `linear` | compact learned model: 48 weights scoring actions from 12 features, trained by CEM on tournament points |
| `hybrid` | push/fold equilibrium table when short + lookup-table preflop + equity/opponent-model postflop + exploit shifts |
| `mixture` | randomised mixture over specs, drawn per table (weights tunable by `train-mixture`) |
| `tree` | gradient-boosted trees / random forest Q(state, action) learned by fitted policy iteration (`train-trees`), compiled to plain-Python if/else; overrides the base heuristic only when the predicted gain is above `margin`, with a hard time guard |
| `montecarlo` | real-time Monte Carlo equity vs opponent-model ranges under a hard µs budget, blended with the table equity prior, then EV choice |
| `allin` | jam/fold family: jam everything, jam the top X% (optionally stack-scaled), or the Nash push/fold jammer at any depth |
| `random`, `station`, `maniac`, `shover`, `nit`, `potodds` | cheap population bots |

Styles are independent presets that change parameters: `tight`, `loose`,
`passive`, `aggressive`, `bluffy`, `value`, `stealer`, `restealer`, `shover`,
`icm_survival`, `bigstack_bully`, `exploitative`, `static`. Combine them freely,
e.g. `{"type":"equity","styles":["loose","aggressive","stealer"],"params":{"bluff_rate":0.2}}`.
All tunable parameters and their bounds are in `strategies/base.py:COMMON_PARAMS`.

Opponent modelling (`OppModel`) keeps Beta-prior estimates of VPIP, PFR,
fold-to-bet, aggression and all-in rate per seat, **within the current table
only**. `adapt` sets how much those estimates count.

## Evaluation methodology

- The objective is official points per table, not chips or hands won.
- The statistical unit is the table. CIs are 95 % normal intervals over independent tables. Hands are never treated as independent samples.
- In matched comparisons every candidate plays the same tables (seeds, opponent lineups, seat). The "vs leader" column is a paired difference with its own CI.
- Candidates are tuned on `train.json` fields plus archived elites (self-play) and evaluated on `holdout.json` fields (different parameterisations) with a different seed stream.
- The leaderboard reports points, CI, paired gap, worst field, placement distribution, latency p50/p99/max, errors/timeouts/illegal actions, memory (fresh-process tracemalloc), training CPU, and behaviour metrics (VPIP/PFR/3-bet/AF/WTSD…) that confirm the styles really differ.

## Runtime guarantees of the exported bot

- The default `submission/` is the compact GBDT bot (`results/gbdt_variants/c40_4.json`); `submission_heuristic/` is the faster, slightly weaker fallback. Decisions are table lookups, arithmetic and compiled trees. There is no Monte Carlo in the submitted bots. Local p99 is 260 µs (GBDT) and 170 µs (heuristic) on a loaded Apple M5.
- Tables load and `gc.freeze()` runs at import, so no GC pass walks the tables mid-decision.
- Every exception or illegal action falls back to check/fold. Raise sizes are clamped to the legal range before returning.
- There are no network, subprocess or threading imports (checked statically). The bot is well under 1 MB.
- Each table gets a fresh instance. No cross-table memory and no team identification.
