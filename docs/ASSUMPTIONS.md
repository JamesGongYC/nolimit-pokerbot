# Assumptions and unresolved differences from the official engine

Sources: https://pokerbot.sequoiacap.com/ and /rules.html, read 2026-11-05; the Bay Area competition format page (pasted by the participant) and the tournament API, read 2026-11-06.
The starter kit (agent template, simulator, ICM calculator, payout structure,
blind schedule) is released **at the event** and is not public, and the
variant is announced at the event. Everything below marked PROVISIONAL needs
replacing once the kit is out. Swapping is done in config files; code changes
are only needed where noted.

## Confirmed by the rules

| Item | Value | Where in this repo |
|---|---|---|
| Players per table | 8 | `configs/event_provisional.json: table_size` |
| Starting stack | 2,000 | `starting_stack` |
| Blinds | start 10/20, increase every **20** hands (format page) | `blind_levels[0]`, `hands_per_level` |
| Table end | one bot has all chips, or **250** hands (format page) | `max_hands`; `engine.Table.run` |
| Default payouts | 1st 50, 2nd 30, 3rd 20, 4th–8th 0 (format page) | `payouts` |
| Same-hand busts | more chips at hand start places higher (format page) | `bust_tie_rule: "stack"` |
| Cap scoring | remaining points split among survivors by ICM | `scoring.icm` (Malmuth-Harville, exact) |
| Iterations | ≥ 5,000 per scored round (format page; rules page said 10,000), seats randomized each iteration | `iterations_per_tournament`, `sim.tournament` |
| Fillers | short tables are filled with bots **not already seated at that table**; filler finishes never score; fields smaller than 8 play short tables | `sim.tournament` |
| Team score | average points per table over the iterations | `sim.tournament` |
| Final score | 6 scored rounds, weights 1–6, starting at build end and every 20 min after; each round snapshots the active bot | `evaluate.official` |
| Decision time | 1 ms; on error/timeout the bot folds, or checks if it can | `engine.Table._ask` |
| Submission | upload `agent.py` or a zip with `agent.py` at its root, **50 MB** default; an upload must be selected as the active bot; no network; state reset at each table; anonymized seat IDs | `export.py` checks; new agent instance per table |
| Field size | Bay Area: 148 teams (API standings) → 19 tables per iteration, 4 filler seats | `configs/populations/official_field_148.json` |

## PROVISIONAL (replace from the starter kit)

1. **Variant.** Default is no-limit hold'em. The homepage demos are NLHE, but
   the rules say the variant is announced at the event. `pokerbot/variant.py`
   already supports PL/FL hold'em, short deck (36 cards, flush > full house,
   trips > straight, A-6-7-8-9 wheel) and Omaha-style "exactly k hole cards"
   evaluation. For a new 2-card variant, set `"variant"` in the event JSON, then
   run `precompute` and `pushfold`. For 4-card games the preflop abstraction
   (169 classes) does not apply: bots fall back to a crude percentile and the
   lookup and push/fold tables have to be rebuilt with a new abstraction (code
   change in `features.py`/`precompute.py`).
2. **Payout / placement points.** The format page confirms the *default* `[50, 30, 20, 0, 0, 0, 0, 0]`; the starter kit could still override it per round. This
   choice matters a lot: ICM pressure, the value of survival and the right
   push/fold ranges all depend on it. Rerun the optimizer once the real
   structure is known (`docs/RESULTS.md` notes how sensitive the results are).
3. **Blind schedule after level 1.** We assume 10/20, 15/30, 25/50, 50/100,
   75/150, 100/200, 150/300, 200/400, 300/600, 400/800, 500/1000, 600/1200,
   800/1600 (13 levels of 20 hands, covering the 250-hand cap), with no antes. The homepage demo
   shows 50/100, which fits level 4 here but proves nothing.
4. **Agent interface.** Our `act(obs) -> (kind, amount)` plus `on_hand_end(info)`
   (`pokerbot/agent_api.py`) is a guess. The exported `submission/bot.py` has
   `to_internal` / `from_internal` hooks, which should be the only place that
   needs editing. If the official observation is missing a field we use (for
   example the full action history or revealed showdown cards), the opponent
   model degrades but still works.
5. **Busts on the same hand.** We rank by start-of-hand stack and split equal
   stacks (`bust_tie_rule: "stack"`). `"split"` is also available.
6. **Illegal actions.** Any malformed or out-of-range action becomes check/fold
   (rules: "if bot errors … automatically folds or checks"). A "fold" when a
   check is available becomes a check. A raise when raising is not allowed
   (for example an all-in for no more than the call) becomes check/fold, not a
   call. This is the strictest reading. Our own bots never rely on it:
   `BaseBot.finish` and `legalize` turn such actions into legal ones first.
   `invalid_raise_policy: "clamp"` gives a lenient engine.
7. **Betting rules not specified by the rules page** (standard choices):
   - moving button over live seats (no dead button); heads-up, the button posts SB and acts first preflop
   - a short blind posts all-in; the amount to call preflop is still the full BB
   - NL minimum raise = size of the last full raise (≥ BB); an all-in that is less than a full raise does not reopen raising for players who already acted
   - burn cards; an uncalled excess is returned through side-pot layering
   - odd chips go to the first winner left of the button (the homepage demo gives them to the first winner in seat order)
   - antes, when configured, are dead money
8. **What agents can observe.** Our observation has stacks, bets, pot, seats,
   blinds, hand number, payouts, the betting history of the current hand,
   and (after the hand) showdown hands that were not folded. Hole cards of
   folded players and the deck are never visible. Opponents are known only
   by seat index within one table, so no cross-table memory is possible.
9. **Timing.** We time each `act()` call in-process with `perf_counter_ns`. The
   official harness may count IPC/serialization overhead, so our bot aims to
   stay well under the limit (p99 ≈ 0.1 ms locally on an Apple M5).
   Hardware differs, so keep a 5–10× margin. In the parallel simulator, OS
   scheduling jitter produces occasional multi-ms outliers, so parallel sims
   *count* every >1 ms decision but only enforce the fallback above
   `POKERBOT_SIM_TIME_SCALE` × limit (default 3). The export benchmark
   runs single-process and checks against the strict 1 ms limit.

## Modelling limits (strategy side)

- Equity tables use bucketed Monte Carlo against *preflop* ranges (top 100/50/25/12 %). Postflop range narrowing is a heuristic shift, not Bayesian card-level range tracking.
- The push/fold "CFR" is fictitious play on an abstraction: equal stacks, the first caller only, no card removal, ICM approximated by loss aversion λ. It is **not** a solution of the full game.
- Synthetic opponent fields are guesses at what 60-minute bots look like. They are **not** the real competitors. Treat field-sensitivity tables as a robustness check, not a forecast.
