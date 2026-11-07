# REBUILD SPEC — Sequoia "No Limit" pokerbot stack

**This is a prompt for a fresh Claude Code session.** Read the whole file before writing
code, then build the stack in the order in §0.3. Every module, signature, constant, formula
and edge case below matches a working, tested implementation. Reproduce them **exactly**
unless the event's starter kit says otherwise; the starter kit always wins (§0.2).

Conventions used here:
- "MUST" = behaviour the tests or the results depend on.
- `code` = an exact name to use.
- Pseudocode is Python-like and is meant to be transcribed nearly line for line.

---

## 0. Mission, constraints, build plan

### 0.1 What the event is (CONFIRMED by the Bay Area format page, 2026-11-06)
- **Tables:** 8-max sit-and-go tables. The variant is announced when the build starts; NLHE is the working default.
- **Structure:** 2,000 starting chips. Blinds start at 10/20 and rise **every 20 hands**. A table ends when one bot has all the chips or at the **250-hand cap**.
- **Points per table:** 1st 50, 2nd 30, 3rd 20, 4th–8th 0 (the default; the kit may override it). At the cap, survivors split the remaining points by ICM on their stacks. When several bots bust on one hand, more chips at the start of the hand places higher.
- **Iterations:** each scored round runs **≥ 5,000 iterations**, and every bot plays one table per iteration. Seats are randomized each iteration.
- **Fillers:** a short table is filled with bots **not already seated at it**, and filler finishes never score. Fields smaller than 8 play short tables.
- **Scoring:** a round score = average points per table. Final score = Σ weight × round score.
- **Round schedule:** 6 scored rounds with weights 1..6. Round 1 starts when the build ends, then one every 20 minutes (+20 … +100 min). Round 6 carries 6/21 of the total weight.
- **Snapshots:** each round **freezes a snapshot of every team's active bot at its start**. A team without an active bot scores 0 for that round.
- **Practice runs:** about 1,000 iterations every 10 minutes during the build. They never count, but they are **the only feedback from the real field**.
- **Uploads:** upload `agent.py`, or a zip with `agent.py` at its root. The default limit is 50 MB (follow the kit's package and unpacked-file limits). An upload must then be **selected as the active bot**. The API reports `submission_cooldown_seconds: 30`.
- **Ties:** equal final totals are ordered by countback: weighted share of 1st places, then 2nd places, and so on.
- **Field:** the Bay Area competition has **148 teams**, so each iteration is 19 tables with 4 filler seats.
- **Bot constraints:**
  - 1 ms default decision allowance; a timeout, error or illegal action checks when free, otherwise folds;
  - no network;
  - memory starts empty at every table; opponents have anonymized seat IDs.

### 0.2 First 5 minutes at the event: read the starter kit, then fill in these values
Put them into `configs/event.json` (schema in §2.1):
1. `blind_levels` after level 1, and antes.
2. The variant (deck, hole cards, streets, betting structure, hand ranking).
3. The official Agent interface (observation and action format). Adapt only `to_internal` / `from_internal` in the exported `bot.py` (§9).
4. Any payout or time-allowance override, and the package or unpacked-file limits.
5. The site's round options, if the kit documents them. The site's round schema has `icm_luck_correction` (legacy / all_in_only), `resampling_continuations` / `resampling_threshold`, `time_bank_ms` and `payout_total`. If any is on, it changes scoring or timing, so model it before tuning.

Still PROVISIONAL in `configs/event_provisional.json`: the blinds after level 1 are `[[10,20],[15,30],[25,50],[50,100],[75,150],[100,200],[150,300],[200,400],[300,600],[400,800],[500,1000],[600,1200],[800,1600]]` (13 levels × 20 hands = 260 ≥ 250), with no antes. Equal start-stack busts split their places.

### 0.3 Event timeline: 10 minutes of dev, 50 minutes of compute (~7 CPU cores assumed)
Pre-trained data files cannot be brought in. **All tables and models are regenerated at the event.**
Development is a sprint. The hour belongs to compute: evolution runs from about minute 16 to 52.

#### 0.3.1 Parallel dev (minutes 0–10)
The main session reads §0.2 and fills in `configs/event.json` plus the `to_internal`/`from_internal` mapping. At minute 0 it spawns **5 subagents in one message**, each given this whole file plus its assignment. The interfaces are fixed by this spec, so agents never wait on each other.

| Agent | Writes | Sections | Done when |
|---|---|---|---|
| A | `cards, variant, event, scoring, agent_api, engine`, `tests/helpers.py`, T1–T4 | §1–4 | T1–T4 pass |
| B | `features, precompute, pfsolve, pushfold` | §5 | `precompute --samples 0.2` runs end to end |
| C | `strategies/`: `base, parametric, simple, allin, montecarlo, learned` (linear), `lookup` (abstract actions + `LookupBot`; HybridBot subclasses it), `hybrid`, `treebot`, `__init__` registry and presets JSON (drop only the mixture and Q-table presets) | §7.1–7.5, 7.7 (actions only), 7.10, §10.1, 10.2, 10.4 | T5 passes |
| D | `sim, optimize` (Space + CEM), `treelearn` (collect + fit), `evolve` (incl. gated refits, `evo-best`, `evo-check`, shortlist verdict), `evaluate` (official + crossplay only) | §8.1–8.5, §10.2–10.3, §11 | a 1-generation smoke run (`--pop 2 --tables 8 --duel-tables 2 --progress-tables 4`) finishes |
| E | `export` + `bench`, `cli`, `tools/train_trees.py`, `.venv-train` setup (start the package install **first**), `configs/populations/train.json` | §9, §10.3, §13, §8.6 | `export` + `bench` run on `heur_tag` |

The main session integrates at minute ~8 and runs `python3 -m pokerbot.cli test`.

**Cut from the event build:** the Q-table learner (`search.py`), mixture, the hold-out leaderboard, sensitivity runs and the memory probe. Hybrid runs **without** a Q-table: with `table: ""`, its preflop falls back to ParametricBot ranges.

**Event evolution families (all 8, decided 2026-11-06):** `heuristic, equity, montecarlo, linear, allin, hybrid, gbdt, rf`.

**Start precompute early.** At about minute 5, as soon as A's T1 and B's features import cleanly, launch `precompute --samples 3` in the background (~100 s; measured 160 s at samples 5).

#### 0.3.2 Compute timeline (minutes 10–60)

| Minutes | Run | Output |
|---|---|---|
| 10–13 | `pushfold --iters 300` (~40 s); `export --spec equity_g23.json`, then `bench` | **Submission 1** (`equity_g23`, §12.2) |
| 13–16 | `train-trees --base gbdt_g36_base.json --rounds 1 --tables 3000 --eps .15 --trees 40 --depth 4 --leaf 1500 --lr .12 --margin .35 --eval-tables 400`. Collection takes ~90 s and the fit ~3 s. `gbdt_g36_base.json` = the §12.1 params with `type: heuristic` and without `model/margin/tree_streets`. Then a matched check vs submission 1. | **Submission 2** if it wins |
| 16–52 | `caffeinate -i evolve --name event --generations 40 --minutes 36 --pop 6 --tables 80 --duel-tables 10 --progress-tables 20 --retrain-every 8 --retrain-tables 3000 --group-cap 16`. Seeds: §12 champions, with gbdt/rf pointed at the 13–16 model. | |
| ~28, ~40 | Checkpoint: `evo-best --run results/evo_event --family gbdt --out cand.json`, then `evo-check --run results/evo_event --a cand.json --b submission/bot_spec.json --tables 200` (~30 s, runs alongside evolution). If the candidate wins by more than its CI, export, bench and resubmit. | **Submissions 3–4** |
| 52–57 | `evo-verdict --run results/evo_event --shortlist 3 --extra submission/bot_spec.json --iterations 400 --tournaments 6 --xtables 10`: ~20 entries, ~11k official + ~4k cross-play tables, ≈ 6 min | |
| 57–60 | Export the verdict winner, bench, submit. If the bench fails acceptance (§9.3), keep the previous submission. | **Final** |

Resubmission is allowed every 5 minutes, so the checkpoints cost nothing.

**How the numbers were sized:**
- **Generation time:** a median of 67 s at 100 tables in the reference run; at 80 tables that is ≈ 55 s per generation, so ≈ 35–40 generations.
- **Refits:** GBDT peaked after the generation-30/40 refits. With `--retrain-every 8` there are 4 refits, each gated (§11.4).
- **Verdict:** throughput was ≈ 2,400 tables/min.
- **Calibrate on the event machine:** read the first generation's wall time and set `--generations` to `36 min / that time`.

#### 0.3.3 Cut-lines
- **Dev overruns at minute 10.** As soon as A, B and C pass, ship Submission 1. Start evolution with whatever families work. Trees matter most, then MC/equity.
- **No scikit-learn** (the install fails or there is no internet on the dev machine): use the pure-Python fitter (§10.5), or drop gbdt/rf and evolve the 6 parameter families.
- **Evolution crashes:** it resumes from `state.json`. Rerun the same command.
- **Behind schedule at minute 50:** skip the verdict. Export `evo-best --family gbdt` after an `evo-check` against the deployed bot.

#### 0.3.4 After the build: compute keeps going through the scored rounds
The build is only the first 60 minutes. Scored rounds start at build end and then every 20 minutes. Each round freezes the active bot **at its start**, so the evolution run does not have to stop at minute 60.

| Time after build end | Round (weight) | Action before the round starts |
|---|---|---|
| 0 | 1 (1×) | the best bot from §0.3.2 is uploaded **and selected active** |
| +20 min | 2 (2×) | keep `evolve` running (resume the same `--name`). At +15: `evo-best` → `evo-check` vs active → upload + activate if it wins |
| +40 / +60 / +80 min | 3–5 (3–5×) | the same checkpoint at +35 / +55 / +75 |
| +100 min | 6 (6×) | at +90: a shortlist verdict (§11.6, ≈ 6 min), then upload + activate the winner by +98 |

- **The later rounds matter more:** rounds 4–6 carry 15/21 of the weight. Spend compute so the strongest bot is active at +60, +80 and +100.
- **Never swap in an unchecked bot right before a round.** A bad snapshot is locked in for that whole round. Use `evo-check` (paired CI) and `bench` every time.
- **Real-field feedback:**
  - during the build, practice runs (≈ 1,000 iterations every 10 min) score the active bot against the real field; read them after each one;
  - after the build, each round's results arrive in the next 20-minute window;
  - if the real field punishes a style (for example, our bot finishes 4th–8th far more than expected), add a matching archetype to the generator pool or anchors before the next checkpoint.
- **Uploading means two steps: upload, then select active.** An upload alone changes nothing. The upload is `agent.py`, or a zip with `agent.py` at its root, so the exporter writes a zip with that layout (§9).

### 0.4 Language and runtime rules
- The bot runtime is **Python 3 stdlib only** (no numpy at runtime).
- Offline tree fitting may use numpy + scikit-learn in a separate venv (`.venv-train`). If neither is available, see §10.5.
- Every bot instance lives for exactly one table. Never use module-level mutable state that differs per table. Read-only caches (tables, compiled trees) are fine.
- Forbidden imports in the submission: `socket, urllib, http, requests, subprocess, multiprocessing, threading, asyncio, ssl, ftplib, smtplib`.

### 0.5 Package layout
```
configs/event.json                   # event (§2.1)
configs/agents/{baselines,styles,research}.json   # named presets (§7.10)
configs/populations/{train,holdout}.json          # synthetic fields (§8.6)
pokerbot/__init__.py (empty)
pokerbot/cards.py variant.py event.py scoring.py agent_api.py engine.py
pokerbot/features.py precompute.py pfsolve.py pushfold.py
pokerbot/sim.py optimize.py treelearn.py evolve.py evaluate.py export.py cli.py
pokerbot/strategies/__init__.py base.py parametric.py simple.py allin.py montecarlo.py
                    learned.py lookup.py hybrid.py mixture.py treebot.py
pokerbot/data/                       # generated tables + models
tools/train_trees.py                 # runs in .venv-train
tests/helpers.py test_cards.py test_scoring.py test_engine.py test_isolation.py test_strategies.py
```
Run everything as `python3 -m pokerbot.cli --workers 7 <cmd> ...`.

---

## 1. `cards.py` — encoding and evaluator

- Card = `int` `rank*4 + suit`. Rank 0 = deuce .. 12 = ace. Suit 0..3. `RANKS="23456789TJQKA"`, `SUITS="cdhs"`.
- Helpers:
  - `card("As")`;
  - `cards("AsKd Qh")` (spaces ignored, 2 chars per card);
  - `card_str`, `cards_str` (space-joined).
- Categories: `HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)`; `STANDARD_ORDER = list(range(9))`.
- Precomputed tables over 13-bit rank masks:
  - `POP[m]` = popcount;
  - `BITS_DESC[m]` = tuple of set ranks, high→low;
  - `TOP5[m]` = `_kick(BITS_DESC[m][:5])`, where `_kick(ranks)` packs up to 5 ranks as 4-bit nibbles of `(rank+1)`, first rank in the highest nibble, missing ranks = 0.
- `_straight_table(deck_ranks)` → list of 8192 entries:
  - `lo = sorted(deck_ranks)`;
  - windows = every `lo[i:i+5]` with `lo[i+4]-lo[i]==4`, as (mask, high = `lo[i+4]`);
  - if 12 is in `lo`, add the wheel `(1<<12) | mask(lo[:4])` with high `lo[3]`. For short deck this gives A-6-7-8-9.
  - Sort windows by high, descending. `table[m]` = high of the first window fully contained in m, else -1.
- `class Evaluator(deck_ranks=range(13), category_order=None)`:
  - `self.straight = _straight_table(...)`;
  - `cat_base[c] = order[c] << 20`;
  - `cat_of_value = {order[c]: c}`;
  - `category(score) = cat_of_value[score >> 20]`.
  - `evaluate(cs)` (5..7 cards, larger = better). Exact algorithm:
    ```
    cnt[13]=0; sm[4]=0 (suit rank masks)
    for c: cnt[c>>2]+=1; sm[c&3] |= 1<<(c>>2)
    rm = OR of sm; best=-1
    for s in sm: if POP[s]>=5:
        sh=straight[s]; if sh>=0: return base[SF] | (sh+1)
        best = base[FLUSH] | TOP5[s]; break
    scan r in BITS_DESC[rm]: n=cnt[r]: 4->quad; 3->trip (first), trip2 (second); 2->p1 (first), p2 (second)
    if quad>=0: k=BITS_DESC[rm & ~(1<<quad)]; v=base[QUADS]|((quad+1)<<16)|((k[0]+1)<<12 if k else 0); return max(v,best)
    if trip>=0: pr=max(trip2,p1); if pr>=0: v=base[FULL]|((trip+1)<<16)|((pr+1)<<12); best=max(best,v)
    sh=straight[rm]; if sh>=0: best=max(best, base[STRAIGHT]|(sh+1))
    if trip>=0: k=BITS_DESC[rm&~(1<<trip)]; v=base[TRIPS]|((trip+1)<<16)|(_k2(k)<<8); return max(v,best)
    if best>=0: return best
    if p2>=0: k=BITS_DESC[rm&~(1<<p1)&~(1<<p2)]; return base[TWO_PAIR]|((p1+1)<<16)|((p2+1)<<12)|((k[0]+1)<<8 if k else 0)
    if p1>=0: k=BITS_DESC[rm&~(1<<p1)]; return base[PAIR]|((p1+1)<<16)|(_k3(k)<<4)
    return base[HIGH]|TOP5[rm]
    ```
    `_k2(k)` packs the top 2 kickers as nibbles (`(k0+1)<<4 | (k1+1)`). `_k3` packs the top 3 (`<<8, <<4, <<0`). A missing kicker = 0.
  - `evaluate_split(hole, board, use_hole)` = max over `combinations(hole,use_hole) × combinations(board,5-use_hole)` (Omaha).
- `STANDARD = Evaluator()`.
- `hand_class(c1,c2)` → 0..168:
  - sort ranks so r1 ≥ r2;
  - a pair → `r1`;
  - otherwise `idx = r1*(r1-1)//2 + r2`; suited → `13+idx`, offsuit → `91+idx`.
- `class_name(i)` gives e.g. "AKs" / "72o" / "TT". `class_combos(i, deck_ranks)` lists concrete (a,b) pairs: 6 for a pair, 4 suited, 12 offsuit, [] if a rank is not in the deck.

**T1 tests:**
- (a) `naive5`/`naive7` reference evaluator:
  - categories from the shape of rank counts, plus flush and straight (wheel = A + the 4 lowest ranks);
  - tiebreak tuple ordered by (count desc, rank desc);
  - naive7 = max over the 21 five-card subsets.

  Use `Random(7)` for 3,000 pairs of random 7-card hands. Assert that `category(evaluate(a)) == naive7(a)[0]` and that the comparison sign of (a vs b) matches.
- (b) Known cases:
  - royal flush > 9-high straight flush;
  - `Ac2d3h4s5c9dKh` is a straight (category 4) and loses to the 6-high straight `6c2d3h4s5c9dKh`;
  - `AcAd9h9s7c7d2h == AcAd9h9s7c7d3h` (a kicker beyond 5 cards is ignored).
- (c) Short deck:
  - a flush `AsKs9s7s6sTdJd` beats a full house `AsAdAhKsKcTdJd`;
  - `As6d7h8s9cKdKh` is a straight;
  - the deck has 36 cards.
- (d) All 1,326 combos map onto exactly 169 classes.

---

## 2. `variant.py`, `event.py`, `scoring.py`

### 2.1 Event JSON (exact keys and defaults)
```json
{"name":"...", "table_size":8, "starting_stack":2000, "max_hands":250, "hands_per_level":20,
 "blind_levels":[[10,20,0],[15,30,0],...],   // [sb, bb, ante]; ante optional (pad with 0)
 "payouts":[50,30,20,0,0,0,0,0],             // padded/truncated to table_size
 "bust_tie_rule":"stack",                    // "stack" | "split"
 "variant":{"name":"nlhe"},
 "decision_time_limit_ms":1.0, "enforce_time_limit":true,
 "invalid_raise_policy":"fallback",          // "fallback" | "clamp"
 "submission_size_limit_mb":50, "tournament_weights":[1,2,3,4,5,6],
 "iterations_per_tournament":5000}
```
`Event` class:
- `level(hand_no) = min((hand_no-1)//hands_per_level, len(levels)-1)` (the last level repeats);
- `blinds(hand_no)` → (sb, bb, ante);
- `with_overrides(**kw)` → a new Event;
- `Event.load(path=None, overrides=None)`. The default path is `configs/event.json`, falling back to the provisional file.
- Other attributes: `time_limit_ms`, `weights`, `iterations`, `payouts`, `variant` (a Variant), `cfg` (the raw dict).

### 2.2 Variant
`Variant(name="nlhe", deck_ranks=None, hole_cards=2, hole_use="any", streets=None, betting="no_limit", category_order="standard", burn=True, raise_cap=None, **ignored)`:
- streets default `[["preflop",0],["flop",3],["turn",1],["river",1]]`;
- `deck = [r*4+s for r in deck_ranks for s in 0..3]`;
- `raise_cap` defaults to 4 for fixed_limit, else None;
- `evaluator = Evaluator(deck_ranks, order)`;
- `hand_value(hole, board)` uses `evaluate(hole+board)` when `hole_use=="any"`, else `evaluate_split`.

Presets (the config is merged over the preset):
- `nlhe:{}`;
- `plhe:{betting:pot_limit}`;
- `flhe:{betting:fixed_limit}`;
- `short_deck:{deck_ranks:[4..12], category_order:"short_deck"}`;
- `plo4:{hole_cards:4, hole_use:2, betting:pot_limit}`.

The short-deck order swaps the strengths of FLUSH↔FULL_HOUSE and TRIPS↔STRAIGHT.

### 2.3 Scoring
- `icm(stacks, payouts)` is exact Malmuth–Harville by bitmask DP:
  - keep only positive stacks; zero stacks get 0;
  - k = number of paid places (trailing zeros dropped, capped at the number of live players);
  - `layer = {0:(prob 1, chips used 0)}`. For each place: for each mask and each j not in mask, `q = p*st[j]/(total-used)`; `eq[j] += q*pay[place]`; when more places remain, accumulate `nxt[mask|bit] = (prob sum, used+st[j])`.
- `bust_points(busted, start_stacks, survivors, payouts, rule)` returns `{seat:(place, points)}`:
  - places start at `survivors+1`;
  - rule "stack": sort busted players by start-of-hand stack, descending; group equal stacks; each group splits the sum of its places' payouts evenly and reports its best place;
  - rule "split": one group.

**T2 tests:**
- ICM vs brute-force permutation enumeration (n ≤ 6) to 1e-9;
- Σ icm = Σ paid payouts;
- a zero stack gets 0;
- `hero_icm` (§7.1) matches `icm` exactly for ≤ 4 paid places;
- the stack rule ranks correctly; equal stacks split.

---

## 3. `engine.py` — table engine (MUST match exactly)

Constants: `LAT_BUCKET_US=10`, `LAT_BUCKETS=400` (0..4 ms; the last bucket is overflow).

`SeatStats` (slots, all ints, plus `hist[400]`):
- fields: `decisions, time_ns, max_ns, timeouts, errors, illegal, hist, hands, vpip, pfr, pf_open_opps, pf_opens, bets, raises, calls, checks, folds, face_bet, fold_to_bet, saw_flop, showdowns, won_showdown, allins, pf_3bet_opps, pf_3bets, steal_opps, steals, chips_won`;
- `merge` sums fields (max for `max_ns`, elementwise for hist); `as_dict` / `from_dict` serialize.

`TableResult`: `points[n], places[n], final_stacks, hands_played, capped, stats[n], hand_logs, chip_check_failures`.

`Table(event, agents, seed, record=False, time_scale=1.0, deck_fn=None, stacks=None, button=None)`:
- `rng = random.Random(seed)`;
- `limit_ns = time_limit_ms*1e6`; `enforce_ns = limit_ns*time_scale`;
- stacks default to the starting stack for everyone;
- `button = rng.randrange(n)` unless given;
- `deck_fn(hand_no)` (tests only) returns a deck list that is dealt by `pop()` from the end.

### 3.1 `run()`
```
total = sum(stacks); hand_no = 0
while hand_no < max_hands and live players > 1:
    hand_no += 1; start = copy(stacks); play_hand(hand_no)
    if sum(stacks) != total: chip_check_failures += 1
    busted = [i with start[i]>0 and stacks[i]==0]
    if busted: survivors = count(stacks>0); assign bust_points(...) into places/points
alive = seats with chips
if len(alive)==1: places[a]=1; points[a]=payouts[0]
else: capped=True; eq = icm(stacks, payouts[:len(alive)]); places by chip rank (info only); points = eq
```

### 3.2 `play_hand(hand_no)`
- `(sb, bb, ante) = event.blinds(hand_no)`; `alive[i] = stacks[i] > 0`.
- Button:
  - from hand 2 on, move to the next alive seat after the current button;
  - on hand 1, keep it if that seat is alive, else move to the next alive seat.
- Blind seats:
  - heads-up (2 alive): SB = button, BB = next alive;
  - otherwise SB = next alive after the button, BB = next alive after SB.
- `folded = not alive` (dead seats count as folded). `bets[n]` = 0, `contrib[n]` = 0, `history = []`.
- `pay(i, amt)` takes `min(amt, stack)` and adds it to both bets and contrib.
- Antes: everyone alive pays the ante, then **reset `bets` to 0**. Antes are dead money in contrib only.
- Post SB then BB with `pay`. Append `(0, sb_seat, "sb", bets[sb])` and `(0, bb_seat, "bb", bets[bb])` to history.
- Deck:
  - `hrng = random.Random((seed*1000003 + hand_no*7919) & 0xFFFFFFFFFFFF)`;
  - `deck = copy(variant.deck); hrng.shuffle(deck)` (or `deck_fn`).
- Deal: order = alive seats starting left of the button. For each of `hole_cards` rounds, each seat in order gets `deck.pop()`. Every dealt seat gets `stats.hands += 1`.
- `hand_ctx = {hand_no, sb, bb, ante, button, sb_seat, bb_seat, level, alive}`; `pf_flags = {raised:0, vpip:set(), first_raiser:None}`.
- Streets: for `si, (street, ncards)` in `variant.streets`:
  - if `si > 0`:
    - break if fewer than 2 players are unfolded;
    - burn one card if `burn and ncards`;
    - deal `ncards` to the board;
    - reset `bets` to 0;
    - on the flop, every unfolded seat gets `saw_flop += 1`;
  - break if fewer than 2 are unfolded;
  - first to act = the next alive seat after BB preflop, otherwise the next alive seat after the button;
  - `bets = _betting_round(...)`.
- Finally, `_settle(...)`.

### 3.3 `_betting_round(si, street, first, bets, contrib, folded, holes, board, history, ctx, pf, bb)`
```
current = max(bets) if si==0 else 0;  if si==0: current = max(current, bb)   # short BB: still owe full BB
last_raise = bb
fixed_unit = bb if si<=1 else 2*bb
n_bets = 1 if si==0 else 0
full_version = 0; acted_version = [-1]*n
pending = {i: not folded and stack>0}; ptr = first
while pending:
    if unfolded count < 2: break
    i = first seat in pending scanning ptr, ptr+1, ... (mod n); pending.discard(i); ptr = i+1
    others_can_act = any(j!=i, not folded[j], stacks[j]>0)
    to_call = min(max(current - bets[i], 0), stacks[i])
    if to_call == 0 and not others_can_act: continue
    can_check = to_call == 0
    raise_right = acted_version[i] != full_version      # incomplete all-in raises do not reopen
    can_raise = stacks[i] > to_call and others_can_act and raise_right and (raise_cap is None or n_bets < raise_cap)
    max_to = bets[i] + stacks[i]
    if fixed_limit: min_to = max_to_cap = current + fixed_unit
    else: min_to = current + max(last_raise, bb); max_to_cap = max_to
          if pot_limit: max_to_cap = current + sum(contrib) + (current - bets[i])
    min_raise_to = min(min_to, max_to); max_raise_to = min(max(max_to_cap, min_raise_to), max_to)
    obs = {... see §3.5 ...}
    kind, amt = _ask(i, obs)
    # stats bookkeeping (exact):
    facing = current > bets[i]; if facing: face_bet += 1
    if si==0 and not pf.raised and i != bb_seat: pf_open_opps += 1; if i in (button, sb_seat): steal_opps += 1
    if si==0 and pf.raised==1 and pf.first_raiser != i: pf_3bet_opps += 1
    FOLD: folded[i]=True; folds+=1; if facing: fold_to_bet+=1; pending.discard(i); history.append((si,i,"fold",bets[i]))
    CHECK: checks+=1; history.append((si,i,"check",bets[i]))
    CALL: pay to_call; calls+=1; if si==0: pf.vpip.add(i); if stack==0: allins+=1; history.append((si,i,"call",bets[i]))
    RAISE: pay amt-bets[i]; if stack==0: allins+=1; (bets+=1 if current==0 else raises+=1)
           if si==0: pf.vpip.add(i); if not pf.raised: pf_opens+=1; first_raiser=i; if i in (button,sb): steals+=1
                     elif pf.raised==1: pf_3bets+=1;  pf.raised += 1
           inc = bets[i] - current
           if inc > 0:
               if inc >= last_raise or fixed_limit:
                   if not fixed_limit: last_raise = max(inc, last_raise)
                   full_version += 1; n_bets += 1
               current = bets[i]
               pending = {j != i: not folded, stack>0}
           history.append((si,i,"raise",bets[i]))       # amount = total bet this street ("raise-to")
    acted_version[i] = full_version
if si==0: every seat in pf.vpip gets vpip+=1; every preflop raiser gets pfr+=1
return bets
```

### 3.4 `_ask(i, obs)` and `legalize`
- Time `agent.act(obs)` with `perf_counter_ns`. An exception counts as an error and returns `fallback_action(obs)` (check if possible, else fold).
- Record decisions, `time_ns`, `max_ns`, and the histogram bucket `int(dt/10000)`, capped at 399.
- If `dt > limit_ns`: `timeouts += 1`. If enforcement is on **and** `dt > enforce_ns`, return the fallback.
- Otherwise `legalize(obs, raw, policy)` (in agent_api), counting `illegal` when flagged:
  - a malformed action → fallback, illegal;
  - FOLD when a check is possible → CHECK, illegal;
  - CHECK when facing a bet → FOLD, illegal;
  - CALL with `to_call <= 0` → CHECK, not illegal;
  - RAISE when `not can_raise` → fallback, illegal;
  - RAISE with a non-int amount → fallback, illegal;
  - an amount inside `[min_raise_to, max_raise_to]` is OK;
  - outside the range: clamp if the policy is "clamp" (illegal), else fallback (illegal).

### 3.5 Observation dict (exact keys — nothing else; tests enforce this)
- Seats and hand: `seat, num_seats, hand_no, max_hands, hands_per_level, level, sb, bb, ante, button, sb_seat, bb_seat`.
- Street and cards: `street, street_index, hole (list), board (list)`.
- Chips: `stacks` (chips behind, all seats, a copy), `bets` (this street, a copy), `contrib` (this hand), `pot = sum(contrib)`.
- Players: `in_hand` (not folded), `alive` (started the hand with chips).
- Legal options: `to_call, min_raise_to, max_raise_to, can_check, can_raise`.
- Context: `history` (a copy; list of 4-tuples `(street_index, seat, kind, amount_to:int)`), `payouts`, `current_bet`.

All lists are fresh copies: a mutating agent must not affect the engine.

### 3.6 `_settle(contrib, folded, holes, board, btn, history, hand_no, start)`
- One contender left: they win `sum(contrib)`.
- Otherwise:
  - `scores = {i: hand_value}` and `showdown = {i: hole}`;
  - `levels = sorted(set(c for c in contrib if c > 0))`; `prev = 0`;
  - for each level `lv`:
    - `slice = Σ_i (min(c, lv) - min(c, prev))`;
    - `elig = [i in contenders with contrib[i] >= lv]`. If empty, use `[contenders with contrib >= prev]`, and failing that all contenders. This returns uncalled excess: the excess level is eligible only for its owner.
    - Winners = elig with the best score, sorted by `(s - btn - 1) % n` so the first is left of the button.
    - `share, odd = divmod(slice, len(winners))`; the first `odd` winners get +1.
  - Count showdowns and `won_showdown` (won > contrib).
- Add the winnings to stacks; `chips_won += stack - start`.
- `info = {hand_no, stacks, showdown, board, winners, won, history}`. Call `on_hand_end(info)` for every seat that started with chips; an exception there counts as `errors += 1`.
- If `record`, append a hand log.

`play_table(event, agents, seed, record=False, time_scale=1.0)` returns `Table(...).run()`.

**T3 engine tests** (use `Table(..., deck_fn=lambda h: deck, stacks=..., button=...)`):
- The deck builder `make_deck(holes_in_deal_order, board, burn=True)` builds the sequence: holes round-robin, then for each street a burn (junk card) + street cards. It returns `rest + reversed(seq)`, so `pop()` deals in order.
- Board `RAGS = cards("2h7d9c8sJh")`, `ROYAL = cards("AsKsQsJsTs")`. `allin(o)` = raise to max if `can_raise`, else call.

Cases:
1. Stacks [100,300,500], button 0, holes AA/KK/QQ, all-in, all-in, call → stacks [300,400,200].
2. The same with shove 500 / call 300: the excess is returned, total = 900.
3. Holes QQ/AA/KK → [0,700,200].
4. Blinds [[5,10,1]], 3 players with 100 each, seat 0 folds, seat 1 calls, seat 2 checks, royal board → seat 0 = 99, seat 1 +1 (odd chip), seat 2 +0.
5. Min raise:
   - stacks [1000,1000,130], seat 0 is first preflop with `min_raise_to == 40`;
   - seat 0 raises to 100 → seat 1 sees `min_raise_to == 180`;
   - seat 2 is all-in for 130 (incomplete) → seat 0 back with `to_call == 30` and `can_raise == False`.
6. With seat 2's stack = 300, the full raise reopens: seat 0 can raise.
7. Heads-up: button = SB, acts first preflop with `to_call == 10`, and the BB acts first on the flop.
8. An illegal raise to 5 → `illegal == 1`, stacks [490,510].
9. Fuzz, 4 variants × 40 seeds. An Auditor+Chaos agent raises exceptions, returns garbage, makes out-of-range raises and asserts obs invariants:
   - `to_call >= 0`;
   - `can_check == (to_call == 0)`;
   - when `can_raise`: `min <= max <= stack + bet`, and `max > current_bet`.

   Then check: no chip failures, final total 16000, Σ points = Σ payouts, errors > 0, illegal > 0.
10. Bust ordering:
    - payouts [50,30,20], stacks [100,300,500], seat 2 holds AA → places [3,2,1], points [20,30,50];
    - equal stacks [200,200,500] → [25,25,50].

---

## 4. `agent_api.py`
Constants: `FOLD, CHECK, CALL, RAISE = "fold","check","call","raise"`.

`class Agent(params=None, seed=0)`:
- `act(obs)` returns `(kind, amount)`; the amount is the raise-to total for this street;
- `on_hand_end(info)` is optional.

Also define `fallback_action(obs)` and `legalize(obs, action, policy)` (§3.4).

**T4 isolation tests:**
- a fresh agent instance per table (24 instances for 3 tables × 8), and each instance sees increasing hand numbers only;
- observation keys ⊆ the allowed set; history tuples have length 4 with int amounts;
- showdown ∩ folded = ∅;
- a Vandal agent that mutates its obs cannot change the chip total;
- tournament fillers: 10 entries → 2 tables; every entry is scored once per iteration; Σ points ≤ 2·Σ payouts;
- filler rules, using a capture runner that records the lineups:
  - 148 entries → 19 tables of 8 with no bot twice at a table and all 148 seated;
  - 5 entries → tables of 5 (short).

---

## 5. Hand-strength features and offline tables

### 5.1 `features.py` (runtime, table lookups only)
Constants:
- `RANGES=[1.0,0.5,0.25,0.12]` (opponent range = the top X of combos);
- `K_MAX=7`, `POST_K=[1,2,3]` (3 means 3+).

Made-hand classes (ordered by strength), 0..15:
`M_AIR, M_OVERS, M_WEAK_PAIR, M_MID_PAIR, M_TOP_WEAK, M_TOP_GOOD, M_OVERPAIR, M_TWO_PAIR_BOARD, M_TWO_PAIR, M_TRIPS, M_SET, M_STRAIGHT, M_FLUSH_LOW, M_FLUSH_HIGH, M_FULL, M_MONSTER`.

Draw classes: `D_NONE, D_GUT, D_OESD, D_FLUSH, D_COMBO = 0..4`.

`class Features(variant, tables=None)`. Tables load from `pokerbot/data/tables_<variant>.json` and are cached per variant name. Fields: `pf_eq[range][k-1][cls]`, `pf_pct[cls]` (0 = best .. 1), `post0/1/2` (dict key → `[range][k]`), `hu` (flat 169×169 equity×1000, row = hero).

Methods:
- `preflop_pct(hole)` = `pf_pct[hand_class]`. Fallback `_chen_pct`:
  - `r1 ≥ r2` ranks;
  - `s = 2*r1 + r2 + (6 if pair) + (2 if suited) - max(0, r1-r2-1)`;
  - pct = `clamp(1 - s/48, 0, 1)`.
- `preflop_equity(hole, k, range_idx=0)`:
  - k is clamped to 1..7;
  - result = `pf_eq[ri][k-1][cls]`;
  - fallback `(1-p)**0.8 / k**0.55 + 0.05`.
- `hu_equity(c1, c2)` = `hu[c1*169 + c2] / 1000`.
- `bucket(hole, board)` → `(made, draw, tex)`, exact. Inputs: `cat` = evaluator category of hole+board; `brs` = board ranks; `bcnt` = rank counts; `h1, h2` = hole ranks; `top` = max board rank; `board_paired = len(bcnt) < len(brs)`.
  - Made class:
    - cat ≥ 7 → MONSTER.
    - cat 6 → FULL, except when the board alone is a full house (5 cards) → TRIPS.
    - cat 5 → flush suit fs; hs = hole ranks in fs. None → AIR. Otherwise FLUSH_HIGH if `max(hs) >= 10` (Q+), else FLUSH_LOW.
    - cat 4 → STRAIGHT if the board-only rank mask is not a straight, else WEAK_PAIR (playing the board).
    - cat 3 → SET if pocket pair; TRIPS if one hole rank has count 2 on the board; else AIR (trips on board).
    - cat 2:
      - both hole ranks on the board and different → TWO_PAIR;
      - pocket pair above top → TWO_PAIR_BOARD;
      - a board pair plus a hole rank on the board → TWO_PAIR_BOARD;
      - ≥ 2 board pairs → OVERS if `min(h1,h2) > top`, else AIR;
      - otherwise MID_PAIR.
    - cat 1:
      - board paired → OVERS if both hole ranks > top, else AIR;
      - pocket pair → OVERPAIR if > top; MID_PAIR if > the 2nd-highest board rank; else WEAK_PAIR;
      - otherwise pr = the paired hole rank, kicker = the other. pr == top → TOP_GOOD if kicker ≥ 9 (J+), else TOP_WEAK. pr == lowest distinct board rank → WEAK_PAIR. Else MID_PAIR.
    - cat 0 → OVERS if both hole ranks > top, else AIR.
  - Draws: only if `len(board) < 5` and `made < M_STRAIGHT`.
    - fd = some suit has exactly 4 among all cards with ≥ 1 hole card in it.
    - `hmask = boardmask | hole ranks`. outs = number of ranks r in the deck not in hmask where `straight[hmask|bit] >= 0` and `straight[boardmask|bit] < 0`.
    - sd = OESD if outs ≥ 2, GUT if outs == 1.
    - draw = COMBO if fd and sd; FLUSH if fd; else sd.
  - Texture bits:
    - bit0 = board paired;
    - bit1 = some suit has ≥ 3 board cards;
    - bit2 = "straighty": some 5-rank window (lo = -1..8, with -1 meaning the ace) contains ≥ 3 board ranks. Memoize by board mask.
- `bucket_equity(nboard, made, draw, tex, k, range_idx=0, level=2)`:
  - s = 1/2/3 for flop/turn/river; ki = `min(max(k,1),3) - 1`;
  - keys `"s,made,draw,tex"` (post2), `"s,made,draw"` (post1), `"s,made"` (post0); skip post2 if level < 2 and post1 if level < 1;
  - for the first table that has the key: e = `row[ri][ki]`; if k > 3, `e = e**(k/3)`;
  - fallback: `base=[.25,.32,.4,.5,.6,.68,.75,.62,.8,.82,.88,.9,.88,.94,.97,.99][made] + [0,.05,.12,.13,.2][draw]`, capped at .99, then raised to the power k.
- `post_equity(hole, board, k, ri=0, level=2)` = bucket + bucket_equity.

### 5.2 `precompute.py` (offline, multiprocessing Pool)
- `_share(hero, opps)` = 1 if hero > max; 0 if lower; else `1/(1 + #ties)`.
- `_deal_from_range(rng, deck_left, allowed)`: up to 200 tries of `rng.sample(deck_left, 2)` until `allowed[class]`.
- `allowed = [pct[c] < frac]`, or None if frac ≥ 1.

Steps, with `samples` multiplying all counts:
1. **Ranking.** Per class, 1500 trials. Rotate the combos. Opponents k=1 on even trials, k=3 on odd. Random boards. Score = mean share.
   - pct by combo weight: order classes by score descending; `pct[c] = (cum + w[c]/2)/1326`.
2. **pf_eq.** 600 trials per (class, range, k = 1..7), opponents dealt from the range, a random 5-card board. Job seed `1000c + 100ri + k`.
3. **hu.** For each class pair, 160 trials. Hero combo `ca[t%len]`; opponent combo `cb[(7t+3)%len]` (on overlap, pick another non-conflicting combo). Store `int(round(1000·mean))`; the default is 500.
4. **post buckets.** For each (range, k ∈ POST_K): workers × (40000/workers) random deals. Hero is random; opponents come from the range; there is a 5-card board. For s, nb in ((1,3),(2,4),(3,5)), credit the final share to the keys at all 3 levels using `bucket(hero, board[:nb])`.
   - Merge counts. Keep a key only if every (range, k) cell has ≥ 30 samples. Value = `round(mean, 4)`.
5. Write the JSON `{variant, ranges, post_k, pf_pct, pf_eq, hu, post0, post1, post2, samples, build_seconds}` with compact separators.

`samples=5` takes ≈160 s wall on 7 cores; `samples=2` is fine at the event.

### 5.3 `pfsolve.py` (offline) and `pushfold.py` (runtime): push/fold equilibrium
`pushfold.py` constants:
- `STACKS=[1,1.5,2,3,4,5,6,7,8,10,12,14,17,20,25]`;
- `LAMS=[0.0,0.3,0.7]`;
- `NS=range(1,8)` (n = players behind the pusher).

Abstraction:
- the pusher is first in with n players behind and all stacks = S BB;
- callers decide independently, and only the first caller plays;
- no card removal; equities come from `hu`;
- loss-averse utility `u(x) = x if x ≥ 0 else (1+λ)x`.

Caller types in action order after the pusher: the last is "bb" (posted 1.0), the second-to-last "sb" (0.5), the rest "nb" (0). With n == 1, the pusher is the SB (posted 0.5). `blinds_total = 1.5 + ante·(n+1)`.

Fictitious play, 300 iterations:
```
push = [1]*169; push_avg = 0; call_avg[t] = 0
for it in 1..iters:
  ref_push = push_avg/(it-1) if it>1 else push
  pw[a] = W[a]*ref_push[a] (W = combo counts); tot = Σpw (if 0: pw=W)
  eq_vs[d] = Σ_a pw[a]*eq[a][d]
  for each caller type t (posted p):
     pot = 2S + blinds_total - p - pusher_posted
     e = 1 - eq_vs[d]/tot
     br_call[d] = 1 if e*u(pot - S + p) + (1-e)*u(-(S-p)) > 0 else 0;  call_avg[t] += br
  for each t: q = call_avg[t]/it; qw[d] = W[d]q[d]; cprob[t] = Σqw/ΣW; cval[t][a] = Σ_d eq[a][d]qw[d]/Σqw (or .5)
  for each pusher class a: ev = 0; reach = 1
     for callers in order (t, p): pot = ...; e = cval[t][a]
        ev += reach*cprob[t]*( e*u(pot - S + pusher_posted) + (1-e)*u(-(S - pusher_posted)) ); reach *= 1-cprob[t]
     ev += reach*u(blinds_total)
     br_push[a] = 1 if ev > 0 else 0;  push_avg += br
return push_avg/iters, call_avg/iters
```

Encode each strategy as a 169-character digit string `str(min(9, round(9x)))`. File `pushfold_<variant>.json` = `{stacks, lams, push:{"n|S|lam": str}, call:{"n|S|lam": {nb,sb,bb: str}}}`. Pool over the 7×15×3 jobs; ≈40 s.

Runtime `PushFoldTable(data)`:
- `_key(n, s, lam)`: clamp n to 1..7; nearest S; nearest λ.
- `push_prob(cls, n, S, lam)` = `digit/9`.
- `call_prob(cls, n_behind_pusher, S, caller_type, lam)` (falls back to "bb" if the type is missing).
- `push_width(n, S, lam)` = combo-weighted mean of the digits / 9, cached.
- `load_pushfold(variant_name)` returns None if the file is missing (bots must handle None).

---

## 6. Generated data (rebuild at the event; nothing is carried in)

| File | Command | Wall time (7 workers) | Used by |
|---|---|---|---|
| `pokerbot/data/tables_<v>.json` | `precompute --samples 2..5` | 70–160 s | every bot except random/maniac (fallbacks exist) |
| `pokerbot/data/pushfold_<v>.json` | `pushfold --iters 300` (needs `hu` from the tables) | ~40 s | parametric push/fold, the allin nash variant |
| `results/treedata/*.bin` | `train-trees` / evolution retrain (CollectorBot) | ~3–6 min per 3,000 tables | tree fitting |
| `pokerbot/data/<name>.json` (trees) | `tools/train_trees.py` in `.venv-train` | ~3 s per fit (40×4) | TreeBot |
| `results/opt_*/best.json` | `optimize` | 5–10 min per family | seeds, exports |
| `results/evo_<name>/state.json` | `evolve` | 50–70 s per generation | verdict, exports |

Setting up the offline venv: `uv venv .venv-train && uv pip install --python .venv-train numpy scikit-learn`, or use `python3 -m venv` + pip. XGBoost is not needed; sklearn's HistGradientBoosting is equivalent here.

---

## 7. Strategies (`pokerbot/strategies/`)

A spec is a dict `{"type": <registry key>, "name": str, "styles": [..], "params": {..}}` or a preset name. `make_agent(spec, seed, variant)` constructs `REGISTRY[type](params=, seed=, styles=, variant=)`.

### 7.1 `base.py`
**`features_for(variant_cfg)`** caches one `Features` per variant config.

**`hero_icm(stacks, me, payouts)`** is a fast single-seat Malmuth–Harville:
- live stacks only; pay = payouts truncated to n with trailing zeros dropped;
- `eq = pay0·s/T`;
- place 2: `Σ_a (a/T)·s/(T-a)`;
- place 3: `Σ_{a≠b} (a/T)(b/(T-a))·s/(T-a-b)`;
- place 4: the analogous triple loop;
- more than 4 paid places: `rest = 1 - Σ p_by_place` and `eq += rest·Σpay[4:]/max(k, n-4)`.

**`OppModel(n, prior_scale=1)`**: per-seat Beta-prior rates within this table.
- PRIORS (mean, weight): `vpip (.30,8)`, `pfr (.15,8)`, `fold_to_bet (.45,6)`, `fold_post (.40,6)`, `aggr (.40,6)`, `allin (.05,10)`.
- Counters: `hands, vpip, pfr, face, fold_face, agg, pas, pf_act, allin, face_post, fold_post`.
- `rate(seat, name)` = `(count + mean·k)/(denominator + k)`. Denominators:
  - vpip and pfr use hands;
  - fold_to_bet uses face;
  - fold_post uses face_post;
  - aggr uses agg+pas;
  - allin uses pf_act.
- `update(info, stacks_before)` walks `info["history"]`:
  - reset per-street `bets = {}` and `cur = 0`;
  - sb/bb entries just set bets and cur;
  - for other actions, mark the seat as seen;
  - `facing = cur > bets.get(seat, 0)`:
    - facing → face++, plus fold_face++ on a fold;
    - on postflop streets, facing also → face_post++, plus fold_post++ on a fold;
  - preflop: pf_act++; call/raise → vpip set; raise → pfr set;
  - postflop: raise → agg++; call → pas++;
  - a raise with `amt >= 0.95·stacks_before[seat]` → allin++;
  - a call/raise updates bets and cur;
  - at the end: hands++ for each seen seat; vpip++ / pfr++ per set.
- `range_idx_for_width(w)` = the index of the nearest value in RANGES.

**`Ctx(o)`** precomputes, with exactly these definitions:
- **Seats and stacks:**
  - `me, n, si, hole, board, stacks, bets, pot, to_call`;
  - `bb = max(1, o.bb)`; `my_stack`, `my_bet`, `my_total = stack + bet`;
  - `opps` = in-hand seats other than me; `n_opp`;
  - `largest_opp_total` = max over opps of stack+bet;
  - `eff = min(my_total, largest_opp_total)`; `eff_bb`; `my_bb`;
  - `pot_odds = to_call/(pot+to_call)` (0 if nothing to call);
  - `n_alive`.
- **Action order:** alive seats starting after `bb_seat` preflop, after the button postflop.
  - `n_behind` = seats after me in that order that are in the hand with chips;
  - `pos = idx/max(1, len(order)-1)` (0 = first, 1 = last);
  - `in_pos = n_behind == 0`.
- **History:**
  - `raises` = raises this street; `limpers` = preflop calls before any raise;
  - `last_raiser` = the last raiser this street;
  - `pf_aggressor` = the last preflop raiser; `street_aggressor`;
  - `unopened = raises == 0`.
- **Tournament state:**
  - `hands_left = max_hands - hand_no + 1`;
  - `facing_allin = last_raiser >= 0 and stacks[last_raiser] == 0`;
  - `spr = (eff - my_bet)/max(1, pot)`;
  - `avg_bb`; `chip_rank` = the number of alive players with a bigger total (0 = chip leader).

**COMMON_PARAMS** (name: default, lo, hi). The bounds are also the search box:
```
open_pct .18 .04 .6 | pos_widen .22 0 .5 | steal_widen .10 0 .4 | limp_rate 0 0 1 | open_size 2.5 2 4
threebet_pct .05 0 .2 | resteal_rate .08 0 .6 | threebet_mult 3 2.2 4.5 | fourbet_pct .025 0 .1
call_margin .02 -.1 .15 | value_eq .62 .45 .85 | raise_eq .75 .55 .95 | value_size .66 .25 1.5
bluff_size .5 .25 1.2 | bluff_rate .12 0 .6 | semibluff_rate .35 0 1 | cbet_rate .6 0 1
bluff_raise_rate .04 0 .3 | implied .04 0 .15 | shove_bb 10 0 25 | push_widen 1 .5 2
call_shove_margin 0 -.1 .15 | commit_spr 1.2 .3 3 | icm_risk 1 0 2.5 | bigstack_pressure .3 0 1
multiway_tighten .3 0 1 | adapt .5 0 1 | abstraction 2 0 2 | mix 0 0 .3 | endgame_push 0 0 1
fold_prior .35 .05 .7 | fold_slope .18 0 .4 | called_penalty .06 0 .2 | realize .85 .5 1
```

**STYLE_PRESETS** (overrides applied in order):
```
tight: open_pct .11 pos_widen .14 call_margin .06 threebet_pct .035
loose: open_pct .30 pos_widen .30 call_margin -.03 threebet_pct .08
passive: value_eq .75 raise_eq .88 bluff_rate .02 semibluff_rate .05 cbet_rate .25 limp_rate .6 threebet_pct .02 resteal_rate 0
aggressive: value_eq .56 raise_eq .68 bluff_rate .25 semibluff_rate .6 cbet_rate .8 value_size .85 bluff_raise_rate .1
bluffy: bluff_rate .4 semibluff_rate .8 bluff_raise_rate .15 cbet_rate .85
value: bluff_rate .02 semibluff_rate .15 value_eq .58 value_size .8 cbet_rate .4
stealer: steal_widen .3 pos_widen .35 | restealer: resteal_rate .4 threebet_pct .07
shover: shove_bb 18 push_widen 1.3 | icm_survival: icm_risk 2 call_shove_margin .06 bigstack_pressure .1
bigstack_bully: bigstack_pressure .9 steal_widen .25 icm_risk 1.2 | exploitative: adapt 1 | static: adapt 0
```
`resolve_params(spec_params, styles, defaults)` applies, in order: COMMON defaults → class DEFAULTS → styles in order → spec params.

**`BaseBot(Agent)`**:
- `__init__`: `p = resolve_params(...)`, `rng = Random(seed)`, `f = features_for(variant)`, `model = None`.
- `_ensure(o)`:
  - create `OppModel(num_seats)` lazily;
  - on a new hand_no, reset `_icm_cache` and set `_hand_start = stacks + contrib`.
- `on_hand_end(info)` → `model.update(info, _hand_start)`.
- `act(o)`: `_ensure`; `c = Ctx(o)`; `finish(c, decide(c))`.
- `finish(c, a)`:
  - RAISE when `not can_raise` → CALL if facing a bet, else CHECK;
  - otherwise clamp the amount to [min_raise_to, max_raise_to]; if `amt >= 0.8·my_total`, use max_raise_to;
  - FOLD or CALL with to_call 0 → CHECK.
- `bet_amount(c, frac)` = `current_bet + max(bb, int(frac·(pot + to_call)))`.
- `shove(c)` = `(RAISE, max_raise_to)`.
- `opp_width(seat, default)` = `(1-adapt)·default + adapt·clamp(pfr·1.1, .04, 1)`; returns the default if there is no model.
- `fold_prob(c, bet_frac)`:
  - `base = fold_prior + fold_slope·min(bet_frac, 1.5)`;
  - for each opponent: obs = rate(`fold_post` if postflop else `fold_to_bet`); `fs = (1-a)·base + a·(obs·(0.7 + 0.3·min(bet_frac,1.5))/0.85)`; multiply `clamp(fs, .02, .95)`.
- `icm_premium(c, risk_amt)`: the extra equity needed beyond chip-EV. Return 0 if `icm_risk ≤ 0`, `risk ≤ 0` or there are no opps. Cache by `(risk, pot)`.
  ```
  vil = opp with the largest stack+bet; r = min(risk, my_stack); m = clamp(r - to_call, 0, stacks[vil]); r_eff = min(r, to_call + m)
  big_pot = pot + r_eff + m
  fold: vil += pot;  win: me += big_pot - r_eff, vil -= m;  lose: me -= r_eff, vil += big_pot - m
  ef, ew, el = hero_icm of each; if ew - el <= 1e-9 or big_pot <= 0: 0
  else max(0, (ef-el)/(ew-el) - r_eff/big_pot)
  ```

### 7.2 `parametric.py` — `ParametricBot(BaseBot)`, `EquityBot`, `HeuristicBot`
`ParametricBot.__init__` loads `pf = load_pushfold(variant name)` when `p.get("use_cfr", 1)`.

**`decide(c)`**:
- if `mix > 0` and `rng < mix`, take a random action: r < .4 call; < .7 fold; else raise `bet_amount(.75)`;
- preflop → `preflop(c)`, else `postflop(c)`.

**Helpers:**
- `lam(c)`:
  - `paid` = the number of positive payouts;
  - `bubble = max(0, 1 - (n_alive - paid)/5)`;
  - result `icm_risk·(0.15 + 0.5·bubble)·(1 if chip_rank > 0 else 0.6)`.
- `cover_frac(c)` = the fraction of opps whose stack+bet < my_total.
- `endgame_shift(c)` = 0 if hands_left > 30, else `endgame_push·(1 if chip_rank >= n_alive//2 else -0.5)`.

**`preflop(c)`**, with `pct = preflop_pct`:
1. If `eff_bb <= shove_bb`, return `pushfold(c, pct)`.
2. **Unopened:**
   - `w = open_pct + pos_widen·pos`;
   - add `steal_widen` if there are no limpers, I am the button or SB, and n_alive > 2;
   - heads-up: `w = max(w, open_pct + pos_widen + steal_widen)`;
   - `w *= 1 + bigstack_pressure·cover·0.6 + endgame_shift·0.3`;
   - `w *= 1 - min(0.6, 0.25·lam)`.
   - If pct ≤ w → RAISE `(open_size + limpers)·bb`.
   - Else: CHECK if free. Limp (CALL) if `limp_rate > 0`, `pct ≤ 1.8w`, `rnd < limp_rate` and `to_call ≤ bb`. Otherwise FOLD.
3. **One raise:**
   - `ow = opp_width(last_raiser, .22)`;
   - `v3 = threebet_pct·clamp(sqrt(ow/.22), .5, 2)`;
   - `amt = int(threebet_mult·current_bet + limpers·bb)`.
   - pct ≤ v3 → `raise_or_shove(amt)`.
   - Late opener (button or SB, or n_alive ≤ 4) and pct ≤ .4 and `rnd < resteal_rate·(1 + bigstack_pressure·cover)` → `raise_or_shove(amt)`.
   - Otherwise:
     - `k = max(1, #opps with bet >= current_bet)`;
     - `eq = preflop_equity(hole, k, range_idx_for_width(ow))`;
     - `need = pot_odds + call_margin + icm_risk·icm_premium(c, to_call)`; minus .03 if I am the BB and `to_call ≤ 1.01·bb`.
     - CALL if `eq·(1 - .05·n_behind) ≥ need`, else FOLD.
4. **Facing a 3-bet or more:**
   - pct ≤ fourbet_pct → `raise_or_shove(int(2.3·current_bet))`;
   - otherwise `eq = preflop_equity(hole, max(1, n_opp), 3)` and `need = pot_odds + call_margin + .03 + icm term`; CALL or FOLD.

`raise_or_shove(c, amt)` shoves if `amt ≥ 0.4·my_total` or `eff_bb < 22`, else raises.

`push_width(c)` = `min(1, 6/max(1, eff_bb)·max(1, n_behind)^-0.75)`.

**`pushfold(c, pct)`**, with `cls = class` and `lam = lam(c)`:
- **Unopened:**
  - `n = max(1, n_behind)`.
  - With a pf table: if `push_widen == 1`, go = `rng < pf.push_prob(cls, n, eff_bb, lam)`; else go = `pct ≤ pf.push_width(n, eff_bb, lam)·push_widen`.
  - Without a pf table: go = `pct ≤ push_width(c)·push_widen`.
  - `w_bonus = bigstack_pressure·cover·0.3 + endgame_shift·0.2`; if not go and w_bonus > 0 and `pct ≤ push_width(c)·(1 + w_bonus)`, go = True.
  - go → shove. Else: CHECK if free; limp if `limp_rate > 0` and `rng < .3·limp_rate` and pct < .5; else FOLD.
- **Facing an all-in, with a pf table:**
  - `ctype` = bb / sb / nb by my seat; `n_bp = max(1, n_behind + 1 + (1 if nb))`;
  - `q = pf.call_prob(cls, n_bp, eff_bb, ctype, lam)`;
  - `ow = opp_width(raiser, 1.3·push_width(c))`;
  - `eq = preflop_equity(hole, max(1, #opps with bet ≥ current_bet), ri(ow))`;
  - `need = pot_odds + call_shove_margin + icm term`;
  - `score = (1-adapt)(q - .5) + adapt·4(eq - need)`; CALL if score > 0, else FOLD.
- **Facing a non-all-in raise:**
  - `ow = opp_width(raiser, .25)`; `eq = preflop_equity(hole, max(1, raises), ri(ow))`;
  - shove if `eq ≥ .5 + call_shove_margin + icm_risk·icm_premium(c, my_stack)` or pct ≤ threebet_pct;
  - else CALL if `eq ≥ pot_odds + call_shove_margin + icm term` and `to_call ≤ .35·my_stack`; else FOLD.

**Postflop helpers:**
- `opp_range_idx(c)`:
  - an aggressor other than me → `w = opp_width(pf_aggressor, .22)`;
  - otherwise `w = (1-a)·.45 + a·mean vpip(opps)`;
  - `ri = range_idx_for_width(w)`;
  - if facing a bet: `ag` = rate(last_raiser, aggr) (default .4); if `ag < .55` or `to_call > .6·(pot - to_call)`, ri += 1; cap at 3.
- `equity(c)` = `post_equity(hole, board, max(1, n_opp), opp_range_idx, int(abstraction))`.

**`HeuristicBot.postflop`**. DEFAULTS: `value_class=M_TOP_GOOD(5), call_class=M_MID_PAIR(3), raise_class=M_TWO_PAIR(8), big_bet_call_class=M_TOP_GOOD(5)`.
```
made, draw, tex = bucket; rnd = rng.random(); multi = n_opp > 1
vc = value_class + (1 if multi and multiway_tighten > .3 else 0); wet = tex & 6
strong_draw = draw in (OESD, FLUSH, COMBO) and board < 5; hu = 1 if not multi else 1 - multiway_tighten
if to_call == 0:
  made >= vc: shove if spr < commit_spr and made >= raise_class, else RAISE bet_amount(value_size*(1.2 if wet else 1))
  strong_draw and rnd < semibluff_rate*hu: RAISE bet_amount(bluff_size)
  pf_aggressor == me and flop and rnd < cbet_rate*hu: RAISE bet_amount(bluff_size)
  n_opp == 1 and in_pos and rnd < bluff_rate: RAISE bet_amount(bluff_size)
  else CHECK
bet_frac = to_call/max(1, pot - to_call)
made >= raise_class or made >= SET: if can_raise: shove if spr < 2*commit_spr else RAISE bet_amount(value_size); else CALL
strong_draw and can_raise and rnd < bluff_raise_rate*hu: raise_or_shove(bet_amount(value_size))
need = call_class if bet_frac <= .6 else big_bet_call_class; icm = icm_risk*icm_premium(c, to_call); if icm > .08: need += 1
made >= need: CALL
strong_draw on flop/turn: draw_eq = (.35 combo | .2 oesd | .19 flush)*(2 on flop else 1); CALL if draw_eq + implied >= pot_odds + call_margin + icm
pot_odds < .15 and made >= M_TOP_WEAK - 2: CALL
FOLD
```

**`EquityBot.postflop`** chooses the highest-EV option:
```
eq = equity(c); pot = float(pot); hu = 1/(1 + multiway_tighten*(n_opp-1)); realize = realize + (1-realize)*in_pos
icm_extra = icm_risk*icm_premium(c, max(to_call, int(.7*pot))); drawing = draw != NONE and board < 5
if to_call == 0:
  options: CHECK ev = eq*pot*realize
  for fr in [bluff_size, value_size] (+ None = all-in if spr < 1.5*commit_spr):
     amt = max_raise_to if None else bet_amount(fr); b = min(amt, my_total) - my_bet; frac = b/max(1, pot)
     fp = fold_prob(c, frac)*hu; eqc = max(0, eq - called_penalty*(1+frac))
     ev = fp*pot + (1-fp)*(eqc*(pot+2b) - b - icm_extra*b) + style_bias
else:
  FOLD 0; CALL ev = eq*(pot+call) - call + (implied*pot if drawing) - (call_margin + icm_extra)*(pot+call)
  if can_raise: for fr in [value_size, None]: amt = max if (None or spr < commit_spr) else bet_amount(fr)
     b = min(amt, my_total) - my_bet; frac = b/max(1, pot); fp = .7*fold_prob*hu; eqc = max(0, eq - 1.6*called_penalty)
     ev = fp*pot + (1-fp)*(eqc*(pot + b + (b - call)) - b) - icm_extra*b + .5*style_bias
style_bias = (+.15*pot if eq >= value_eq) + ((semibluff_rate - .35)*.4*pot if drawing elif eq < .4: (bluff_rate - .12)*.6*pot)
             + ((cbet_rate - .6)*.3*pot if pf_aggressor == me and flop)
```

### 7.3 `simple.py` — baselines (the population and generator use these)
All of them clamp raises to [min, max] themselves; `f = features_for(variant)`.
- **RandomBot** {raise .15, fold .35}:
  - r < raise and can_raise → raise to `randint(min, max)`;
  - r < raise+fold and facing a bet → fold;
  - else call.
- **CallingStation** {raise_eq .8, fold_eq .15}:
  - eq = preflop_equity(hole, 1)^.8 preflop, else post_equity(k=1);
  - eq > raise_eq → raise to the pot;
  - eq < fold_eq and to_call > .5·pot → fold (or check if free);
  - else call.
- **Maniac** {aggr .7, shove .25, fold .1}:
  - r < shove·(1.5 - pct) → all-in;
  - r < aggr → raise `current_bet + max(bb, pot)`;
  - facing a bet with r > 1 - fold and pct > .5 → fold;
  - else call.
- **Shover** {push_pct .2, call_pct .1}, preflop only:
  - unopened and pct ≤ push_pct → all-in;
  - opened and pct ≤ call_pct → call;
  - else check/fold;
  - postflop: check/call.
- **Nit** {play_pct .06}:
  - preflop: pct ≤ play_pct → raise `3·current_bet` (or call); else check/fold;
  - postflop: eq vs k opps > .7 → raise `current_bet + 2/3·pot`; eq > .45 or free → call/check; else fold.
- **PotOddsBot** {raise_eq .7, margin 0}:
  - eq > raise_eq → raise `current_bet + max(bb, .75·pot)`;
  - free → check;
  - call if eq ≥ pot_odds + margin, else fold.

### 7.4 `allin.py` — `AllInBot(BaseBot)`
DEFAULTS `push_pct 1, call_pct 1, ref_bb 15, stack_exp 0, nash 0, post_class 2, lam 0`.
- Postflop: made ≥ post_class → shove (or call); else check/fold.
- Preflop: `scale = (ref_bb/max(1, eff_bb))^stack_exp`.
  - **Unopened**, or raises == 0:
    - nash → go = `rng < pf.push_prob(cls, max(1, n_behind), min(eff_bb, 25), lam)`;
    - otherwise go = `pct ≤ min(1, push_pct·scale)`;
    - shove if go and can_raise, else check/fold.
  - **Facing a raise:**
    - nash → `q = call_prob(cls, max(1, n_behind+1), min(eff_bb, 25), ctype, lam)`, ok if q ≥ .5;
    - otherwise ok = `pct ≤ min(1, call_pct·scale)`;
    - ok → shove (or call); else fold.

### 7.5 `montecarlo.py` — `MonteCarloBot(EquityBot)`
DEFAULTS `mc_budget_us 250, mc_max 300, prior_n 12, range_tighten .6, mc_preflop 0`.

`_widths(c)` per opponent:
- the preflop aggressor → `opp_width(s, .22)`;
- otherwise `(1-a)·.45 + a·vpip`;
- the last raiser when we face a bet → `w *= range_tighten`;
- clamp to [.03, 1].

`mc_equity(c)`:
```
deadline = perf_counter() + budget_us*1e-6; deck = cards not in hole/board; need = 5 - len(board)
while n < mc_max and tries < 3*mc_max:
  tries += 1; if tries & 3 == 0 and perf_counter() > deadline: break
  draw = rng.sample(deck, 2k + need + 6); used = 0
  for each width w: up to 3 tries take draw[used:used+2] (used += 2) and accept if w >= 1 or pct[class] <= w; else reject the sample
  run = board + draw[used:used+need]; me vs max(opps): win 1, tie .5; n += 1
return tot, n
```
`equity(c)`: `prior = EquityBot.equity(c)`. Preflop (unless mc_preflop) returns the prior. Otherwise `(s + prior_n·prior)/(n + prior_n)`.

Measured: a median of ~32 samples fit in 250 µs in CPython.

### 7.6 `learned.py` — `LinearBot(ParametricBot)`
- 12 features: `[1, eq, eq - pot_odds, pot_odds, to_call/my_total, 1/(1 + max(0, spr)), n_opp/7, pos, icm, preflop?, facing?, min(eff_bb, 60)/50]`.
  - Preflop eq = `preflop_equity(hole, k, ri)`, with `ri = 0` if unopened else `min(3, raises+1)` and `k = min(2, max(1, n_behind))` if unopened else `max(1, n_opp)`.
  - Postflop eq = `equity(c)`.
  - icm = `icm_premium(c, to_call or int(.7·pot))`.
- 4 action rows (call, raise small, raise big, all-in) × 12 = 48 weights `p["w"]` (flat, row-major). Fold scores 0. When nothing is to call, fold scores -1e9 and the call score is `max(s, 0)`.
- Raises are legal only if can_raise. Pick the argmax, then `concrete(...)` (§7.7).
- Default weights:
  - call: w[2] = 6, w[8] = -4;
  - raise small: w[0] = -3.2, w[1] = 5, w[2] = 6, w[6] = -1;
  - raise big: w[0] = -5, w[1] = 7, w[2] = 6, w[6] = -1;
  - all-in: w[0] = -6.4, w[1] = 8, w[2] = 6, w[5] = 1.5, w[4] = 2, w[11] = -1.5, w[8] = -3.

### 7.7 `lookup.py` — abstract actions plus the Q-table bot (needed by trees)
- Abstract actions: 0 fold, 1 check/call, 2 raise small, 3 raise big, 4 all-in.
- `legal_actions(c)`:
  - always [1]; add 0 if facing a bet;
  - if can_raise: add [2,3,4] if max > min, else [4].
- `concrete(bot, c, a)`:
  - 0 → fold; 1 → call or check; 4 → max_raise_to;
  - preflop unopened → `(2.5 or 4.0)·bb + limpers·bb`;
  - preflop opened → `(2.5 or 3.5)·current_bet`;
  - postflop → `bet_amount(.5 or 1.0)`.
- `_abstract(act, c)`:
  - fold → 0 if facing a bet, else 1;
  - check/call → 1;
  - amt ≥ max_raise_to → 4;
  - otherwise frac = `(amt - current_bet)/(pot + to_call)`: 2 if < .75, else 3.
- `state_key(bot, c, res)`:
  - preflop: `"p{int(pct·(10|20))}.{pos bucket <.34/<.8/else}.{3 if facing_allin else min(raises,2)}.{eff_bb bucket (8,15,30)|(5,8,12,18,30)}.{0 if n_alive<=3 else 1}"`;
  - postflop: `"f{si}.{int(eq·(10|20))}.{facing 0/1 (≤ half pot)/2/3 all-in}.{spr bucket (1,3)|(.7,1.5,3,6)}.{0 if n_opp==1 else 1}.{in_pos}"`.
- `LookupBot(EquityBot)`:
  - DEFAULTS `res 2, min_visits 30, explore 0, table "", preflop_only 0`;
  - plays the best legal action among those with `visits ≥ min_visits`, otherwise EquityBot;
  - when exploring, it records `(key, a)` and credits the hand's `hero_icm` change on hand end.
  - Optional at the event: it scored below the heuristic.

### 7.8 `hybrid.py` and `mixture.py`
- **`HybridBot(LookupBot)`**. DEFAULTS = LookupBot's + Heuristic class defaults + `exploit_shift .08, post 0`.
  - Preflop:
    - `eff_bb ≤ shove_bb` → pushfold;
    - otherwise use the Q-table row if it has enough visits;
    - otherwise ParametricBot.preflop.
  - Postflop, temporary param shifts (restored in `finally`), based on mean fold_post (ftb), mean vpip and the last raiser's aggr:
    - ftb < .3 or vpip > .6 → bluff_rate ×.3, semibluff ×.6, value_eq -= shift·adapt;
    - ftb > .6 → bluff_rate = min(.8, 2·bluff_rate + .1);
    - aggr > .6 and facing a bet → call_margin -= shift·adapt.
    - Then use Heuristic postflop if `post == 1`, else Equity postflop.
- **`MixtureBot`** `{components:[specs], weights}` picks one component per table: `Random(seed).choices`, inner seed = seed + 1. Optional.

### 7.9 `treebot.py` — see §10.

### 7.10 Registry and presets
- REGISTRY keys: `random station maniac shover nit heuristic equity lookup linear hybrid mixture potodds allin montecarlo tree collector`.
- `resolve(spec)`:
  - a string → a preset (from `configs/agents/*.json`), with name = the key;
  - a dict with `"preset"` → merge params over the preset;
  - name defaults to type.
- Presets to define:
  - **baselines:** `random, station, maniac, maniac_soft{aggr .45, shove .1, fold .3}, shover20{.2,.1}, shover40{.4,.2}, nit, nit_loose{play_pct .12}`;
  - **styles:** `heur_tag[tight,aggressive], heur_lag[loose,aggressive], heur_tp[tight,passive], heur_lp[loose,passive], heur_bluffy[loose,bluffy], heur_default, eq_default, eq_tag, eq_lag[loose,aggressive,stealer], eq_value[tight,value], eq_icm[tight,icm_survival], eq_bully[bigstack_bully,stealer,restealer], eq_shover[shover], eq_exploit[exploitative], eq_static[static], linear_default, hybrid_default, hybrid_tag[tight,aggressive,exploitative], hybrid_icm[tight,icm_survival,exploitative]`;
  - **research:** `jam_always, jam_50{.5,.3}, jam_25{.25,.12}, jam_12{.12,.06}, jam_nash{nash 1}, jam_nash_icm{nash 1, lam .7}, jam_stack{.2,.1,ref 15,exp 1}, potodds, mc_default, mc_tag[tight,aggressive]`.

**T5 strategy tests:**
- Every preset in `configs/agents/*.json` plays one table in seat 0 against `heur_tag, station, maniac, nit, eq_tag, shover20, random`, with:
  - 0 errors and 0 illegal actions;
  - p99 < 1 ms;
  - max < 5 ms;
  - final chips = 16000.
- Every REGISTRY type except collector, tree and mixture appears in some preset.
- Tree compiler: tree `t = [[0,.5,1,4,F,0],[1,2.0,2,3,F,0],[0,0,0,0,T,1.0],[0,0,0,0,T,2.0],[0,0,0,0,T,3.0]]`, two copies, base .5:
  - `q([0,1]) == 2.5`, `q([0,5]) == 4.5`, `q([1,0]) == 6.5`;
  - with `mean: True` and base 0: `q([1,0]) == 3.0`.
- Smith set:
  - a>b>c>a with all of them beating d → `[a,b,c]`;
  - transitive a>b>c → `[a]`.

---

## 8. Simulator, search and evaluation

### 8.1 `sim.py` — parallel table jobs
- `SIM_TIME_SCALE = float(env POKERBOT_SIM_TIME_SCALE or 3.0)`. Parallel sims count every decision over 1 ms but only enforce the fallback above 3 ms. Scheduler jitter otherwise produces fake timeouts. The bench (§9) is strict.
- `_init_worker(event_cfg)`: build a global Event, call `load_tables`, `load_pushfold`, then `gc.freeze()`.
- `run_table_job((lineup, seed, extra))`:
  - agents are `make_agent(spec, seed=(seed*31 + i) & 0x7FFFFFFF, variant)`;
  - `play_table(ev, agents, seed, time_scale=extra.get("time_scale", SIM_TIME_SCALE))`;
  - returns `{points, places, hands, capped, stats:[as_dict], chip_fail, cpu}`, plus `experience` (each agent's `.experience`) if `extra.collect`;
  - log `SLOW TABLE` to stderr if wall > 20 s.
- `Runner(event, workers)` is a context manager around `Pool(workers, _init_worker)`. `run(jobs)` = `pool.map(..., chunksize=max(1, len(jobs)//(8·workers)))`. One worker runs in-process.

**The statistical unit is the table, never the hand.**

### 8.2 Matched evaluation (common random numbers)
- `matched_jobs(n_tables, fields, seed, table_size=8, archive=None, archive_frac=0)`:
  - `rng = Random(seed)`;
  - table t uses field `sorted(fields)[t % F]`, with 7 opponents sampled by weight;
  - each opponent is optionally replaced by an archive member with probability `archive_frac`;
  - seed = `rng.randrange(1<<30)`; candidate seat = `t % 8`;
  - returns `(opps, seed, seat, field_name)`.
- `evaluate_candidates(runner, candidates, plan, collect=False)`: every candidate plays every planned table (same opponents, seed and seat). Returns, per candidate, `{spec, points[], places[], by_field{}, stats (merged), cpu, experience[]}`.
- `mean_ci(xs)` = mean, 1.96·sd/√n.
- `latency_pct(stats, q)` reads from the histogram (10 µs buckets).
- `behavior(stats)` = vpip, pfr, open, steal, 3bet, af, fold_to_bet, wtsd, allin_per_hand.

### 8.3 Official format
`tournament(runner, entries, iterations, seed)`:
- per iteration: shuffle all entries and cut them into tables of 8 in order (`n_tables = ceil(n/8)`; only the last table can be short);
- fill a short table with `rng.sample` of entries **not already at that table**, flagged as unscored, then shuffle that table's seats. If too few are available (field < 8), the table plays short;
- each table gets a seed from the rng;
- collect points only for scored seats.

`official(entries, iterations, tournaments=6, seed)`:
- tournament t uses seed + 1000t;
- names are deduplicated with `#k`;
- per entry: per-tournament mean, `final = Σ w_t·mean_t`, the overall mean ± CI, `final_ci95 = ci·Σw`, and the placement distribution;
- writes .json and .md.

### 8.4 Cross-play
`crossplay(specs, tables, seed)`:
- cell (A, B) = A's mean points when seated with 7 copies of B; the seat rotates `t % 8`;
- seeds are shared down each column, `(seed·1000003 + b·7919 + t) & 0x3FFFFFFF`;
- per row: mean vs others, worst opponent;
- replicator dynamics: 5000 iterations of `x ← x·(Mx)/(xᵀMx)`, normalized.

**Use ≥ 10 tables per cell**, or "worst" and the meta shares are pure noise.

### 8.5 `optimize.py` — the search space and CEM
- `INT_PARAMS = {abstraction, value_class, call_class, raise_class, big_bet_call_class, post, post_class}`.
- SPACES:
  - `core` = `open_pct pos_widen steal_widen open_size threebet_pct resteal_rate call_margin value_eq raise_eq value_size bluff_size bluff_rate semibluff_rate cbet_rate shove_bb push_widen call_shove_margin commit_spr icm_risk bigstack_pressure multiway_tighten adapt implied endgame_push`;
  - `allin` = `push_pct call_pct ref_bb stack_exp post_class`.
- EXTRA_BOUNDS (default, lo, hi): `push_pct (1,.03,1)`, `call_pct (1,.02,1)`, `ref_bb (15,3,40)`, `stack_exp (0,0,2)`, `post_class (2,0,10)`, `prior_n (12,0,60)`, `range_tighten (.6,.2,1)`, `margin (.25,0,2)`, `mc_budget_us (250,100,400)`, `value_class (5,3,10)`, `call_class (3,1,7)`, `raise_class (8,6,12)`, `big_bet_call_class (5,3,9)`.
- `Space(base_spec, space)` maps u ∈ [0,1]^d to a spec:
  - **linear** (type linear): names `w0..w47`; `w = w0 + (u - .5)·2·3.0`.
  - **core**: core names, plus
    - the 4 class params for heuristic, tree, and hybrid with post = 1;
    - `margin` for tree;
    - `fold_prior fold_slope called_penalty realize prior_n range_tighten` for montecarlo;
    - `fold_prior fold_slope called_penalty realize` for equity, lookup, linear, and hybrid with post = 0.
  - `center()` = the base's resolved params normalized into the box (clamped).
  - `spec(u, name)` = denormalize (round ints, otherwise 4 decimals) over a copy of base params.
  - `_encode(sp, spec)` is the inverse.
- CEM `run(...)`. Defaults: generations 8, popsize 16, elite 4, screen_tables 240, final_tables 1200, finalists 4, sigma .18, archive_frac .15.
  - Each generation: candidates = [mu] + (pop-1) samples of `clamp(mu + sig·N(0,1))`.
  - The plan is `matched_jobs(screen_tables, train fields, seed·104729 + g)`, with archive opponents.
  - Score by the plain mean.
  - Elites: `mu` = their mean; `sig = max(.03, 1.1·std)`.
  - Archive the best.
  - Finalists (base + top 4 + final mu) rerun on `final_tables` with fresh seeds; pick by mean.
  - Writes `results/opt_<name>/best.json` and `summary.json`.

### 8.6 Synthetic populations (`configs/populations/*.json`, `{"fields": {name: [[preset, weight], ...]}}`)
- `train`:
  - aggressive = maniac 2, maniac_soft 2, heur_lag 2, shover40 1, eq_lag 1, heur_bluffy 1;
  - passive = station 3, heur_lp 2, heur_tp 2, nit 1, random 1;
  - balanced = heur_tag 2, eq_default 2, eq_tag 2, heur_default 1, eq_value 1;
  - exploitative = eq_exploit 2, eq_bully 2, hybrid_default 2, eq_icm 1;
  - mixed = 1 each of random, station, maniac, shover20, nit, heur_tag, heur_lag, eq_default, eq_lag, hybrid_default.
- `holdout`: the same five archetypes but **unseen parameterizations**, e.g. maniac{.55,.18,.2}, heuristic[loose,bluffy,stealer], shover{.3,.15}, station{.85,.25}, nit{.09}.

**The real field is expected to be strong.** For final selection, prefer the evolution verdict (§11.6) over hold-out scores. Hold-out scores (≈25 points) and hall-of-fame scores (≈17 points) are not comparable: each table pays out a fixed 100 points, so absolute scores depend on field strength.

---

## 9. Export and benchmark (`export.py`)

### 9.1 Upload format and layout (format page: "Upload agent.py or a zip with agent.py at its root", 50 MB default)
```
submission/
  agent.py              # entry point + adapter: the upload's root file
  bot_spec.json         # frozen spec (train_* keys removed; model/table paths retargeted)
  jj_nolimit_bot/       # RUNTIME modules only, renamed from `pokerbot` so nothing can shadow or be
                        #   shadowed by the harness's modules; internal imports are all relative
  jj_nolimit_bot/data/  # tables_<v>.json, pushfold_<v>.json, + referenced model/table files
  configs/event_provisional.json   # lets the bundled engine run the bench/selftest
submission.zip          # THE UPLOAD: the same tree, agent.py at the zip root, no wrapping folder,
                        #   no __pycache__/.pyc/.DS_Store, entries sorted
```
- Never upload a bare `agent.py`: ours imports the package next to it. `tools/submit.py` refuses it.
- There is no root-level `bot.py`. Local tools import `agent`.
- `RUNTIME_MODULES`: `__init__ agent_api cards variant features pushfold event engine scoring strategies/{__init__,base,parametric,lookup,learned,hybrid,mixture,simple,allin,montecarlo,treebot}`.
- **Never copy** precompute, pfsolve, sim, optimize, evolve, treelearn or cli: they import multiprocessing or subprocess.
- `strategies/__init__.py` must not import offline modules.

Referenced files: collect `params.table` and `params.model` (recursively through mixture components). Copy them to `jj_nolimit_bot/data/<basename>` and rewrite the spec paths.

`static_checks(dir, zip, limit_mb)` checks:
- `agent.py` is at the root of both the tree and the zip;
- the zip size and unpacked size are each under the limit (`submission_size_limit_mb`, 50);
- a regex scan of every `.py` for `^\s*(import|from)\s+(socket|urllib|http|requests|subprocess|multiprocessing|threading|asyncio|ssl|ftplib|smtplib)\b` finds nothing.

`selftest(zip, spec)` (export fails unless it passes):
- unzip into a temp dir; start a fresh interpreter with an **unrelated working directory** and no PYTHONPATH;
- import `agent.py` by file path;
- assert that every referenced model or table actually loaded (a missing tree model would otherwise silently play as the heuristic), the equity and push/fold tables loaded, and no `pokerbot` module leaked;
- play one table of 8 exported agents: chips conserved, 0 errors, 0 illegal actions.

**The export must fail loudly if anything is wrong.**

### 9.1b Uploading and activating (`tools/submit.py`; endpoints taken from the site's own client)
- `POST <site>/api/teams/<team>/submissions?filename=<name>`, with the raw file as the body (`application/zip`). The server runs its checks: status `validating` → `ready` | `rejected`, plus `check_log`, `rejection_reason` and `check_position`.
- `GET <site>/api/teams/<team>/submissions` lists them. `POST .../submissions/<id>/cancel` cancels one.
- `PATCH <site>/api/teams/<team>` with `{"active_submission_id": "<id>"}` **selects the active bot**. An upload alone changes nothing, and each round snapshots the active bot at its start.
- Usage:
  - `check [zip]`: local format checks;
  - `upload [zip] --activate`: polls until ready, prints the server check log, and activates only if `ready`;
  - `list`, `status`, `activate <id>`, `cancel <id>`, and `--dry-run`.
- The tool enforces the API's 30 s cooldown (`submission_cooldown_seconds`). The key comes from `~/.config/no-limit-2026/api-key`.

### 9.2 `agent.py` template (behaviour MUST match)
```python
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE) if HERE not in sys.path else None
from jj_nolimit_bot.strategies import make_agent
from jj_nolimit_bot.agent_api import legalize, fallback_action
from jj_nolimit_bot.features import load_tables
from jj_nolimit_bot.pushfold import load_pushfold
SPEC = json.load(open(os.path.join(HERE, "bot_spec.json"))); VARIANT = SPEC.get("variant") or {"name": "nlhe"}
load_tables(VARIANT["name"]); load_pushfold(VARIANT["name"]); make_agent(SPEC, seed=0, variant=VARIANT)  # warm caches + compile trees
import gc; gc.freeze()   # big read-only tables out of GC traversals -> removes latency spikes

def to_internal(obs): return obs            # <- map the OFFICIAL observation to §3.5 here
def from_internal(action, obs): return action   # <- map (kind, amount) to the OFFICIAL action here

class Agent:
    def __init__(self, *a, **k): self._bot = make_agent(SPEC, seed=random.randrange(1 << 30), variant=VARIANT)
    def act(self, obs):
        o = None
        try:
            o = to_internal(obs); a = self._bot.act(o); a, _ = legalize(o, a); return from_internal(a, obs)
        except Exception:
            try: return from_internal(fallback_action(o if o is not None else obs), obs)
            except Exception: return ("fold", 0)
    def on_hand_end(self, info):
        try: self._bot.on_hand_end(info)
        except Exception: pass
```

**Adapter checklist** for when the official obs differs:
- reconstruct `to_call, min/max_raise_to, can_check, can_raise, current_bet, in_hand, alive, contrib, history` from whatever the kit provides;
- if the kit has no `on_hand_end`, derive the opponent-model updates from the history seen at the next hand's first `act` (or set `adapt = 0`);
- if the kit's raise amount is "raise by" rather than "raise to", convert it in `from_internal`.

### 9.3 `bench(dir, tables=40)`
Runs a subprocess in the submission dir, single process:
- times `import bot`;
- plays `tables` tables with `bot.Agent()` in seat `t % 8` against `[station, maniac, heur_tag, eq_default, shover20, nit, random]`, plus one table of 8 exported bots;
- measures tracemalloc peak and RSS.

Report: decisions, p50/p99/p99.9/max µs, number over 1 ms, timeouts, errors, illegal, import ms, MB, size_ok, forbidden imports.

**Acceptance:**
- p99 < 300 µs;
- 0 errors and 0 illegal actions;
- ≤ 1 in 10k decisions over 1 ms;
- size < 50 MB (the upload default; check the kit's limits).

Reference results:
- compact GBDT: p50 100 µs, p99 260 µs, 0.66 MB;
- heuristic: p99 170 µs.

The official harness may add IPC overhead, so keep a 3–5× margin.

---

## 10. Tree-ensemble Q-policies (the deployed family)

### 10.1 Features (`treebot.FEATURE_NAMES`, 29, exact order)
```
street, pct, eq, made, draw, texture, pot_odds, commit, spr, eff_bb, n_opp, n_behind, pos, raises,
facing_allin, pf_aggr_me, icm, n_alive, chip_rank, hands_left, level, opp_vpip, opp_pfr,
opp_fold_post, opp_aggr, cover, pot_bb, limpers, in_pos
```
`feature_vector(bot, c)`:
- **preflop:** made = draw = tex = -1; `k = min(3, max(1, n_behind if unopened else n_opp))`; `eq = preflop_equity(hole, k, 0 if unopened else min(3, raises+1))`.
- **postflop:** bucket; `eq = bucket_equity(len(board), made, draw, tex, max(1, n_opp), 0, 2)`.
- **Main opponent:** the last raiser if it isn't me, else `opps[0]`. Its rates (vpip, pfr, fold_post, aggr) default to (.3, .15, .4, .4).
- **Other values:**
  - icm = `icm_premium(c, to_call or int(.7·pot))`;
  - commit = `to_call/max(1, my_total)`;
  - spr is capped at 20 and eff_bb at 100;
  - cover = the fraction of opps with less;
  - pot_bb = pot/bb;
  - booleans are 0/1.

### 10.2 Data collection (`CollectorBot(ParametricBot)`, DEFAULTS `base: heur_tag, eps .12`)
- `inner = make_agent(p["base"], seed+7)`.
- On the first act of a hand, store `(stacks + contrib, seat, payouts)`.
- Each decision:
  - `x = feature_vector`;
  - if no deviation has happened yet this hand and `rng < eps`: set the deviation flag, **clear the rows recorded so far this hand**, and play a uniformly random legal abstract action;
  - otherwise play the inner bot's action and abstract it;
  - record `(x, a)`.
- At hand end: forward `on_hand_end` to inner. `r = hero_icm(final stacks) - hero_icm(start stacks)`, both with `payouts[:alive at start]`. Append `x + [a, round(r, 4)]` to `experience`.

Rationale: every row from the deviation onward is an unbiased sample of Q_base(s, a).

`treelearn.collect(runner, base_spec, tables, eps, seed, fields, path)`:
- `evaluate_candidates([collector], plan, collect=True)`;
- write float32 rows to `<path>` (array('f').tofile) and `{"ncol": 31, "rows": n}` to `<path>.json`.
- Use 3,000 tables, eps .12–.15 (≈500k rows).

### 10.3 Fitting (`tools/train_trees.py out algo trees depth min_leaf lr data.bin...`, venv with numpy + sklearn)
- Load and stack the files: X = columns[:-2], A = column -2 (int), Y = column -1.
- For each action a with ≥ 4·min_leaf rows: shuffle (`RandomState(0)`); 85/15 split.
  - gbdt: `HistGradientBoostingRegressor(max_iter=trees, max_depth=depth, min_samples_leaf=leaf, learning_rate=lr, l2_regularization=1.0, random_state=0)`;
  - rf: `RandomForestRegressor(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf, max_features=.5, n_jobs=-1)`.
- Export:
  - HGB: `base = _baseline_prediction`; for each `_predictors[i][0].nodes` emit `[feature_idx, num_threshold, left, right, is_leaf, value]`;
  - RF: per tree `[feature or 0, threshold, children_left, children_right, leaf?, value[i][0][0]]` with `base 0` and `"mean": true`.
- Report the validation R² per action and write `{"actions": {a: ens}, "meta": {...}}`.

**Deployed settings: gbdt, 40 trees, depth 4, min_leaf 1500, lr .12** (fits in ~3 s). Expected validation R²: fold ~.5, others .02–.1. That is normal: the rewards are noisy.

### 10.4 Runtime (`TreeBot(HeuristicBot)`)
DEFAULTS = Heuristic + `model "", margin 0, tree_streets "all", tree_budget_us 450`.

- **Compile at load.** `compile_ensemble(ens, name)` emits:
  ```
  def q3(x):
      s = <base>
      if x[f] <= thr:   (nested, 4 spaces per level)
          s += <leaf value>
      else: ...
      [for "mean": s = base + (s - base)/n_trees]
      return s
  ```
  `exec` it once per action and cache the model by path. Equality goes left (`<=`), matching sklearn.
- **`act`** records `_t0 = perf_counter()`.
- **`decide(c)`**:
  - start from `base = HeuristicBot.decide(c)`;
  - skip the trees when `tree_streets == "pre"` and postflop (and vice versa for "post");
  - `x = feature_vector`; `legal` = legal actions that have a model;
  - `ab = _abstract(base)`; evaluate q for ab first, then the others, **stopping when `perf_counter() > _t0 + tree_budget_us`**;
  - `best = argmax`; if `q[best] - q[ab] ≤ margin`, return base; otherwise `concrete(best)`.

**Use `tree_streets="pre"`.** Postflop trees lost 4–13 points, because postflop rewards are too noisy. Preflop-only gained +1.1–1.3.

### 10.5 If sklearn or numpy is unavailable at the event
Implement a minimal exact-split regression-tree booster in pure Python. Algorithm:
- per action, pre-sort each feature once;
- depth 4, min_leaf 1500, 40 rounds, lr .12;
- fit to the residuals;
- emit the same JSON node format.

Subsample to ≤ 150k rows per action to keep fitting under ~2 minutes with multiprocessing over actions. If even that is too slow, skip the trees and ship the heuristic.

### 10.6 Fitted policy iteration (`treelearn.iterate`)
For each round r:
1. collect around the current policy;
2. fit on the last 2 rounds' data;
3. evaluate tree-all, tree-pre, the current policy and the base on 1,500 matched tables;
4. the next policy = the better of tree-all / tree-pre.

Round 2 beat round 0 (27.2 vs 25.0 on training fields). At the event, 1–2 rounds is enough.

---

## 11. Evolution (`evolve.py`) — population training with a top-cycle hall of fame

### 11.1 Concepts and fixed decisions
- **Families** (each has its own CEM distribution in normalized space): `heuristic, equity, montecarlo, linear, allin, hybrid, gbdt, rf`. gbdt and rf are `type: tree` specs that share HeuristicBot params plus `model`, `margin` and `tree_streets: "pre"`. The space kind is `allin` for allin, `linear` for linear, and `core` otherwise.
- **Event build:** all 8 families (§0.3.1). SEEDS come from §12; the hybrid seed has `post: 1` and no table. gbdt and rf start on the model fitted at minutes 13–16. Under a new variant, seeds are only starting means, with initial σ = 0.25.
- **Hall of fame (HoF):** frozen snapshots `<family>_g<gen>`, `{spec, family, gen, fitness}`. Generation 0 = the seed champions (§12).
- **Group** = the top cycle (Smith set) of the HoF win graph, capped at K = 16 by Copeland score. HoF seats are drawn uniformly from the group.
- **Generator:** each seat picks uniformly among the 8 families + 6 baselines and draws uniform-random parameters. Baselines and their ranges:
  - random{raise 0–.4, fold 0–.7};
  - station{raise_eq .6–.95, fold_eq 0–.4};
  - maniac{aggr .2–.9, shove 0–.5, fold 0–.5};
  - nit{play_pct .03–.2};
  - potodds{raise_eq .55–.9, margin -.1–.1};
  - shover{push_pct .05–.6, call_pct .03–.4}.
  - A family seat is `Space.spec(uniform u)`.
- **Mixed tables:** 1 candidate + 7 opponents, alternating 4 HoF + 3 generator (even t) and 3 + 4 (odd t). Shuffle the opponents; candidate seat = t % 8. **Never seat a candidate with its current peers.**
- **Fitness** = the plain mean official points. All candidates of all families play the same tables (CRN).
- **Promotion:** every generation, each family's best candidate joins the HoF.
- **Equilibrium update:** the new members duel the current group and each other only. A duel is a 4 v 4 table, `duel_tables` tables per pair, with the seat pattern alternating `ABABABAB` / `BABABABA`. Seeds = `(seed·1000003 + salt·7919 + pairIndex·104729 + t) & 0x3FFFFFFF`. The edge `"a|b" = [points of a seats, points of b seats]`. `beats(a, b)` = `pa ≥ pb` (ties count both ways).
- **Progress logging** (for monitoring only):
  - the new bests vs the previous generation's bests (fresh seeds);
  - the new bests vs frozen anchors: the gen-0 champions + `station, maniac, nit, potodds, shover20, random, jam_nash, heur_tag`, with fixed seeds.
- **Tree retraining every 8 generations, gated** (§11.4). Decided 2026-11-06: the reference run used 10; the event run uses 8, so ~4 refits fit in 35–40 generations.
- **No holdout.** The final verdict is the HoF against itself (§11.6). **At the event a shortlist verdict is allowed** (decided 2026-11-06); the full-HoF verdict remains the default when time permits.

### 11.2 `top_cycle(members, beats)`
- Adjacency `i → j` if `beats(m_i, m_j)` (i ≠ j).
- Tarjan SCC (recursive is fine for ≤ ~600 nodes; raise the recursion limit or iterate if needed).
- A component has an incoming edge if some edge goes from another component into it. Return the members of all **source** components (no incoming edges), in the original order.

`_capped_cycle(members)`: `cyc = top_cycle`. If `len > cap`, score each m by (wins inside the cycle, with ties = .5; then the sum of margins) and keep the top `cap`, preserving order.

### 11.3 One generation g
```
spaces = {fam: Space(fam.base, kind)}
if g % retrain_every == 0: retrain_trees(g); rebuild spaces
cands = for each fam: [mu] + (pop-1) × clamp(mu + sigma·N(0,1)) → spec "<fam>_g<g>_c<i>"
plan = pool_plan(tables, seed·7777 + g); res = evaluate_candidates(all cands, plan)
for each fam: sort by mean; elites = top max(2, pop//3)
   mu = mean(elites); sigma = clamp(1.1·std(elites), .03, .30)
   promote the best → HoF "<fam>_g<g>" (strip train_* keys; name it)
pairs = [(new, old) for new in new_members for old in group] + all (new_i, new_j) pairs
duels(pairs, salt=g); group = capped_cycle(group + new_members)
progress: fixed_plan(prev_bests, progress_tables, seed·13 + g) and fixed_plan(anchors, progress_tables, seed·17)
   (fixed_plan = 7 opponents drawn uniformly from the list, seat t % 8)
prev_bests = new_members; append a history row; save state.json; append log.jsonl
```
- **Initialization:** seed the HoF with the 8 family champions (name `<fam>_g0`, `mu = Space.center()`, `sigma = .15`), duel all pairs (salt 0), set group = capped cycle, gen = 1.
- **State file** `results/evo_<name>/state.json`: `{gen, families{fam: {base, mu, sigma}}, hof, edges, group, prev_bests, anchors, history, tree_data[], cfg_locked, cpu_s, wall_s}`. Resume if it exists.
- `run(generations, minutes)` stops at the generation count or the wall budget, then writes `summary.md` (per generation: wall, fitness, vs prev, vs anchors, group).

### 11.4 `retrain_trees(g)`
- `champ` = HoF `gbdt_g{g-1}` spec (or the family base).
- Collect `retrain_tables` (3,000) pool-plan tables with `CollectorBot{base: champ, eps .15}`. Write `results/treedata/evo_<name>_g<g>.bin`.
- Fit on the **last 2** data files:
  - gbdt 40×4, leaf 1500, lr .12 → `pokerbot/data/evo_<name>_gbdt_g<g>.json`;
  - rf 30×7, leaf 1500 → `pokerbot/data/evo_<name>_rf_g<g>.json`.
- Point the family bases at the new models. Old HoF members keep their own model paths.
- **Gated refits (required in the event build):** before adopting a refit, play the family's current mu spec with the old model vs the new model on 200 matched `pool_plan` tables (seed `seed·19 + g`). Adopt the new model only if its mean is higher. Otherwise keep the old model and log `refit rejected`. In the reference run, RF regressed after the later ungated refits.

### 11.5 Budget (measured, 7 workers on a shared M-series laptop)
- `--pop 6 --tables 100 --duel-tables 12 --progress-tables 40 --retrain-every 10 --retrain-tables 3000 --group-cap 16`: median 67 s per generation (range 48–230 s, plus one 1,466 s sleep stall), so 60 generations ≈ 1–1.2 h.
- At the event, use `--pop 6 --tables 80 --duel-tables 10 --progress-tables 20 --retrain-every 8` for ≈ 35–40 generations in 36 minutes (§0.3.2). The event run has the same 8 families as the reference run, so the reference timing applies directly.
- macOS: run under `caffeinate -i` and do not `nice` it. A sleep stall made one generation take 25 minutes.
- Without the cap (`--group-cap 0`), the top cycle swallowed 207 of 208 HoF members and generations took 226 s. **Keep the cap.**

### 11.6 Verdict (`evo-verdict --run results/evo_<name> --iterations 300 --tournaments 6 --xtables 1`)
- The whole HoF plays `official(...)` (all-vs-all, seed 99173) and `crossplay(..., xtables, seed 77141)`.
- `verdict.md` contains:
  - the top 60 rows (official weighted, mean per table, cross-play mean, worst, meta);
  - a by-family table (members, mean, cross-play mean, best member);
  - a generation-band table (6 bands).
- With ~500 players it takes ~45–60 min. **At the event**, run the verdict on a shortlist instead: the top 3 per family by in-run fitness plus the deployed bot, ≈ 25 entries, 2,000 iterations × 6 and `xtables 20`.
- **Event mode (`--shortlist K --extra spec.json ...`):**
  - entries = the top K HoF members of each family by in-run `fitness`, plus the extras;
  - the official format at `--iterations`, plus cross-play at `--xtables`;
  - writes the same `verdict.md`.
- Select by the official weighted score. Treat entries within ±1 CI as tied; break ties by cross-play mean, then latency.

### 11.6b Checkpoint helpers (event build)
- `evo-best --run DIR --family F --out spec.json`: among HoF members of family F from the last 5 generations, pick the highest in-run `fitness`. Write its spec.
- `evo-check --run DIR --a A.json --b B.json --tables 200`:
  - build `pool_plan(tables, seed=12345)` from the run's current group and generator;
  - `evaluate_candidates([A, B], plan)`;
  - print both means and the paired difference ± 95% CI (per-table differences on identical tables).
  - "A wins" = diff − CI > 0.
  - Use 2 workers, so evolution keeps the rest.

### 11.7 What the 60-generation run found (provisional event, use as priors)
- **Points per table in the all-vs-all HoF field** (12.5 = average), by band (gen 0–10 → best band):
  - gbdt 9.96 → **16.5** (gens 31–50);
  - rf 10.5 → 15.8 (gens 21–30, then declined after later refits);
  - montecarlo 12.8 → 14.2;
  - equity ~13.4 throughout;
  - heuristic, hybrid, linear ~11 → 12;
  - allin flat ~11.
- **Top 6:** gbdt_g36, g37, g47, g49, g38, g46 (official 360–368, ±19; the top ~20 are tied).
- The generation-0 champions rank low among strong bots: gbdt_g0 (the old deployed bot) is 347th of 488.
- The training-time top cycle was mostly MC and equity bots, which rank 44th–148th in the verdict. **The group is an opponent pool, not a pick list.**
- Gains come mainly from tree **refits**. Parameter families plateau by about generation 20–30.

---

## 12. Tuned values (the starting points; re-tune if payouts or blinds differ)

All of these were tuned under the provisional payouts `[50,30,20,0…]`. If the real payouts differ, use them as CEM seeds or `mu` starts, never as final answers.

### 12.1 Best overall: GBDT preflop-override on a heuristic base (verdict #1, `gbdt_g36`)
```json
{"type":"tree","name":"gbdt_g36","params":{"open_pct":0.2143,"pos_widen":0.1791,"steal_widen":0.2531,"open_size":2.6225,
"threebet_pct":0.0108,"resteal_rate":0.1981,"call_margin":0.0648,"value_eq":0.6331,"raise_eq":0.5554,"value_size":0.9807,
"bluff_size":0.6129,"bluff_rate":0.5189,"semibluff_rate":0.2788,"cbet_rate":0.7451,"shove_bb":13.525,"push_widen":1.1549,
"call_shove_margin":0.1215,"commit_spr":1.7261,"icm_risk":1.2996,"bigstack_pressure":0.0508,"multiway_tighten":0.2188,
"adapt":0.5585,"implied":0.0489,"endgame_push":0.0223,"value_class":4,"call_class":1,"raise_class":11,"big_bet_call_class":5,
"model":"pokerbot/data/<YOUR_FITTED_GBDT>.json","margin":0.3569,"tree_streets":"pre"}}
```
The model must be refit at the event (§10). The tree ranks #1 only with a well-trained model: it was trained around a strong pool, refit at generation 30. With no model loaded, this spec plays as its heuristic base.

### 12.2 First submission, needs no training: `equity_g23` (verdict #45 of 488; the best non-tree, non-MC family member)
```json
{"type":"equity","styles":["tight","aggressive"],"name":"equity_g23","params":{"open_pct":0.1517,"pos_widen":0.3092,
"steal_widen":0.0946,"open_size":3.1378,"threebet_pct":0.0036,"resteal_rate":0.0853,"call_margin":0.1134,"value_eq":0.5396,
"raise_eq":0.7774,"value_size":0.7956,"bluff_size":0.6414,"bluff_rate":0.076,"semibluff_rate":0.2029,"cbet_rate":0.5113,
"shove_bb":17.2353,"push_widen":1.22,"call_shove_margin":0.0887,"commit_spr":1.0963,"icm_risk":0.2999,"bigstack_pressure":0.3271,
"multiway_tighten":0.2317,"adapt":0.5906,"implied":0.0065,"endgame_push":0.0,"fold_prior":0.4239,"fold_slope":0.0563,
"called_penalty":0.0707,"realize":0.7889}}
```
It needs `tables_<v>.json` and `pushfold_<v>.json` (§5); without them it runs on the fallbacks, which is weaker but legal.

### 12.3 Other family champions (verdict rank; mean points per table in the strong field)
- `rf_g21` (#7, 16.87): the same structure as 12.1 with an RF model, margin 0.159. Base params:
  `open_pct .3346 pos_widen .2064 steal_widen .2257 open_size 2.6224 threebet_pct .0269 resteal_rate .016 call_margin .05 value_eq .4754 raise_eq .6169 value_size 1.0214 bluff_size .7108 bluff_rate .2603 semibluff_rate .3869 cbet_rate .9196 shove_bb 9.2547 push_widen .9224 call_shove_margin .0408 commit_spr 1.6558 icm_risk 1.619 bigstack_pressure .1625 multiway_tighten .3569 adapt .4972 implied .0585 endgame_push .0304 value_class 4 call_class 1 raise_class 8 big_bet_call_class 6`.
- `montecarlo_g25` (#53, 14.77), styles [tight, aggressive]:
  `open_pct .0948 pos_widen .0257 steal_widen .1337 open_size 2.9503 threebet_pct .047 resteal_rate .2775 call_margin .0642 value_eq .6215 raise_eq .5788 value_size 1.2983 bluff_size .6612 bluff_rate .1137 semibluff_rate .9478 cbet_rate 1.0 shove_bb 16.7415 push_widen 1.2613 call_shove_margin .0503 commit_spr 1.8109 icm_risk .7518 bigstack_pressure .2461 multiway_tighten .3459 adapt .5341 implied .0774 endgame_push .2221 fold_prior .3387 fold_slope .1312 called_penalty .1567 realize .6035 prior_n 14.1062 range_tighten .3843`.
- `heuristic_g58` (#147, 13.23), styles [tight, aggressive]:
  `open_pct .2869 pos_widen .2216 steal_widen .2728 open_size 2.6025 threebet_pct .0752 resteal_rate 0 call_margin .086 value_eq .6214 raise_eq .5975 value_size .7646 bluff_size .8507 bluff_rate .1956 semibluff_rate .3566 cbet_rate .6753 shove_bb 13.0155 push_widen .8786 call_shove_margin .0306 commit_spr 1.4252 icm_risk .7177 bigstack_pressure .1792 multiway_tighten .0312 adapt .6166 implied .0267 endgame_push .6746 value_class 4 call_class 1 raise_class 12 big_bet_call_class 6`.
- `allin_g12` (#230, 12.11): `push_pct .3099 call_pct .3569 ref_bb 7.4196 stack_exp 1.0208 post_class 2`.
- `linear_g18` (#142, 13.88), 48 weights:
  `[-0.8735,0.9367,7.2296,1.6574,1.5598,0.4694,0.838,-0.1991,-5.709,-0.3591,-0.1872,-1.0266,-5.9718,3.7152,7.2579,-0.6274,-1.2411,-0.1178,0.0821,-0.7707,0.2802,0.1367,-0.5053,-0.6119,-4.6144,9.3611,6.2788,0.3956,-0.4357,1.1793,-1.354,0.2934,-0.4034,2.2912,-0.1238,-1.1352,-9.0087,8.1417,4.9234,0.7939,1.2314,0.7438,-2.0351,0.1093,-2.0644,0.5892,0.5599,-2.2213]`.
- Weak-field heuristic (the old hold-out winner, 25.0 vs weak fields; use it if the field turns out weak), styles [tight, aggressive]:
  `open_pct .3517 pos_widen .1352 steal_widen .1803 open_size 2.3768 threebet_pct .021 resteal_rate .0858 call_margin .0946 value_eq .586 raise_eq .5991 value_size .7046 bluff_size .6598 bluff_rate .3283 semibluff_rate .5473 cbet_rate .7394 shove_bb 9.1229 push_widen 1.0237 call_shove_margin .0334 commit_spr 1.4768 icm_risk 1.0103 bigstack_pressure .1432 multiway_tighten .3545 adapt .4735 implied .0597 endgame_push .1829 value_class 5 call_class 1 raise_class 10 big_bet_call_class 6`.

The evolution seeds (gen 0) = these champions, one per family. For hybrid, use `{"type":"hybrid","params":{...weak-field heuristic params..., "post":1}}`.

### 12.4 Style the tuned bots converged to (strong field)
- Preflop:
  - opens of ~21–29% first in, ~40–50% on the button, and ~65–78% when folded to the button or SB (steal_widen);
  - opens are small (2.6 BB);
  - value 3-bets are very rare (≈1%), but resteal is ~20% vs late opens.
- Postflop:
  - c-bets are frequent (~75%) and bluffing is high (~0.5 heads-up in position);
  - it raises bets only with sets or better (raise_class 10 = SET, 11 = STRAIGHT; sets always qualify);
  - it calls bets of ≤ 0.6 pot with overcards or better (call_class 1 = OVERS) and bigger bets with top-pair-good or better (big_bet_call_class 5);
  - it value-bets top pair weak or better (value_class 4).
- Push/fold below 9–13.5 BB with ICM risk ≈ 1.3.
- The GBDT override (margin ≈ .35) mostly corrects preflop fold/call/jam decisions at short and medium stacks.

---

## 13. CLI (`cli.py`, argparse; global `--event PATH --workers N`)
```
precompute --samples 5                    pushfold --iters 300
simulate --lineup a,b,... --tables 200 --seed 1 [--rotate]
tournament --entries a,b,@file.json --iterations 1000 --tournaments 6 --seed 424242 --out f.json
optimize --base SPEC --method cem|random --space core|allin|linear|preflop|postflop --generations 8 --popsize 16
         --elite 4 --screen-tables 240 --final-tables 1200 --finalists 4 --sigma .18 --archive-frac .15 --seed 1 --out DIR
train-trees --base SPEC --name tree --algo gbdt|rf --rounds 3 --tables 3000 --eps .12 --trees 120 --depth 5 --leaf 400
            --lr .08 --margin 0 --eval-tables 1500 --seed 31
evaluate --candidates a,b --tables 2000 --fields F --seed 777000111 --out results/leaderboard.json [--no-memory]
crossplay --strategies a,b --tables 60 --seed 5150 --out results/crossplay.json
evolve --name main --generations 60 --minutes M --pop 8 --tables 150 --duel-tables 24 --progress-tables 80
       --retrain-every 10 --retrain-tables 6000 --group-cap 16 --seed 2026
evo-verdict --run results/evo_main --iterations 300 --tournaments 6 --xtables 1 [--shortlist K --extra a.json,b.json]
evo-best --run DIR --family gbdt --out cand.json      evo-check --run DIR --a A.json --b B.json --tables 200
export --spec SPEC --out submission       bench --dir submission --tables 40       test
```
- SPEC = a preset name or a path to a spec JSON.
- `load_spec("…/best.json")` names the spec `T-<dirname without opt_>`.

---

## 14. Pitfalls already paid for (do not rediscover)
1. **Parallel timing jitter.** Never enforce 1 ms inside a parallel sim (use time_scale 3). Always `gc.freeze()` after loading tables, in workers and in `bot.py`.
2. **Opponent model.** Keep postflop fold-to-bet (`fold_post`) separate from preflop folds. Mixing them inflated fold equity and caused over-bluffing.
3. **Monte Carlo under 1 ms.** CPython fits only ~30 samples in 250 µs. Check the clock every 4 tries, blend with the table prior (`prior_n`), and test the max latency, not just p99.
4. **Full-size GBDT is too slow.** Use 40 trees × depth 4. Keep the `tree_budget_us` guard and evaluate the base action first.
5. **Trees only preflop.** Postflop Q-regression on single-hand rewards does not work (R² ≤ .16).
6. **The lookup table does not transfer across postflop policies.** Preflop action values depend on how you play later.
7. **Mixtures never beat their best component** against non-adaptive fields.
8. **Overfitting with matched seeds.** Pick finalists on fresh seeds, and note the winner's curse: the best of N is inflated by about one standard error.
9. **The uncapped top cycle explodes** through noisy duels. Cap at 16 by Copeland.
10. **Absolute scores depend on the field** (fixed 100 points per table). Compare only on identical tables.
11. **Fixed-limit and pot-limit** use different raise sizing (§3.3). Short deck needs its own straight table (A-6-7-8-9) and category order. Rerun `precompute` and `pushfold` for any new variant.
12. **zsh does not word-split** unquoted loop variables. Write loops in Python.
13. **Edge cases in obs consumers:**
    - Treat `to_call` as already capped at my stack.
    - `can_raise` is False after an incomplete all-in for seats that already acted.
    - `max_raise_to` may equal `min_raise_to` (an all-in only).
