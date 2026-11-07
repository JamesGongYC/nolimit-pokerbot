"""Table engine: one sit-and-go table played to one survivor or the hand cap.

Rules implemented (see docs/ASSUMPTIONS.md for what is provisional):
  * moving button over live seats; heads-up button posts SB and acts first preflop
  * short blinds/antes post all-in; preflop bet to match is the full big blind
  * NL min-raise = last full raise (>= BB); all-in under-raises do not reopen
    raising for players who already acted; PL and FL supported via Variant
  * burn cards, uncalled bets returned through side-pot layering, odd chips
    to the first winner left of the button
  * busts on the same hand ranked by start-of-hand stack (configurable)
  * at the hand cap, survivors split remaining places by ICM
"""
import random
import time

from .agent_api import FOLD, CHECK, CALL, RAISE, legalize, fallback_action
from .scoring import icm, bust_points

LAT_BUCKET_US = 10
LAT_BUCKETS = 400  # 0..4ms in 10us buckets (last bucket = overflow)


class SeatStats(object):
    __slots__ = ("decisions", "time_ns", "max_ns", "timeouts", "errors", "illegal", "hist",
                 "hands", "vpip", "pfr", "pf_open_opps", "pf_opens", "bets", "raises", "calls",
                 "checks", "folds", "face_bet", "fold_to_bet", "saw_flop", "showdowns",
                 "won_showdown", "allins", "pf_3bet_opps", "pf_3bets", "steal_opps", "steals",
                 "chips_won")

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, 0)
        self.hist = [0] * LAT_BUCKETS

    def merge(self, o):
        for k in self.__slots__:
            if k == "hist":
                for i, v in enumerate(o.hist):
                    self.hist[i] += v
            elif k == "max_ns":
                self.max_ns = max(self.max_ns, o.max_ns)
            else:
                setattr(self, k, getattr(self, k) + getattr(o, k))

    def as_dict(self):
        d = {k: getattr(self, k) for k in self.__slots__ if k != "hist"}
        d["hist"] = list(self.hist)
        return d

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for k, v in d.items():
            setattr(s, k, list(v) if k == "hist" else v)
        return s


class TableResult(object):
    def __init__(self, n):
        self.points = [0.0] * n
        self.places = [0] * n
        self.final_stacks = [0] * n
        self.hands_played = 0
        self.capped = False
        self.stats = [SeatStats() for _ in range(n)]
        self.hand_logs = []
        self.chip_check_failures = 0


class Table(object):
    def __init__(self, event, agents, seed, record=False, time_scale=1.0, deck_fn=None, stacks=None,
                 button=None):
        self.event = event
        self.variant = event.variant
        self.agents = agents
        self.n = len(agents)
        self.seed = seed
        self.rng = random.Random(seed)
        self.record = record
        self.limit_ns = event.time_limit_ms * 1e6
        # enforcement threshold; parallel sims use time_scale > 1 to ignore
        # scheduler jitter (timeouts are still counted against limit_ns)
        self.enforce_ns = self.limit_ns * time_scale
        self.stacks = list(stacks) if stacks else [event.starting_stack] * self.n
        self.button = self.rng.randrange(self.n) if button is None else button
        self.deck_fn = deck_fn  # tests: deck_fn(hand_no) -> deck list (cards popped from the end)
        self.res = TableResult(self.n)

    # ------------------------------------------------------------------ table
    def run(self):
        ev, n, res = self.event, self.n, self.res
        total_chips = sum(self.stacks)
        hand_no = 0
        while hand_no < ev.max_hands and sum(1 for s in self.stacks if s > 0) > 1:
            hand_no += 1
            start = list(self.stacks)
            self.play_hand(hand_no)
            if sum(self.stacks) != total_chips:
                res.chip_check_failures += 1
            busted = [i for i in range(n) if start[i] > 0 and self.stacks[i] == 0]
            if busted:
                survivors = sum(1 for s in self.stacks if s > 0)
                for seat, (place, pts) in bust_points(busted, start, survivors, ev.payouts,
                                                      ev.bust_tie_rule).items():
                    res.places[seat] = place
                    res.points[seat] = pts
        res.hands_played = hand_no
        alive = [i for i in range(n) if self.stacks[i] > 0]
        if len(alive) == 1:
            res.places[alive[0]] = 1
            res.points[alive[0]] = ev.payouts[0]
        else:
            res.capped = True
            eq = icm(self.stacks, ev.payouts[:len(alive)])
            order = sorted(alive, key=lambda i: -self.stacks[i])
            for rank, i in enumerate(order):
                res.places[i] = rank + 1  # chip rank (informational)
                res.points[i] = eq[i]
        res.final_stacks = list(self.stacks)
        return res

    # ------------------------------------------------------------------- hand
    def _next_alive(self, i, alive):
        n = self.n
        for k in range(1, n + 1):
            j = (i + k) % n
            if alive[j]:
                return j
        return i

    def play_hand(self, hand_no):
        ev, var, n = self.event, self.variant, self.n
        stacks = self.stacks
        sb_amt, bb_amt, ante = ev.blinds(hand_no)
        alive = [s > 0 for s in stacks]
        n_alive = sum(alive)
        self.button = self._next_alive(self.button, alive) if hand_no > 1 else (
            self.button if alive[self.button] else self._next_alive(self.button, alive))
        btn = self.button
        if n_alive == 2:
            sb_seat = btn
            bb_seat = self._next_alive(btn, alive)
        else:
            sb_seat = self._next_alive(btn, alive)
            bb_seat = self._next_alive(sb_seat, alive)
        start = list(stacks)
        folded = [not a for a in alive]
        bets = [0] * n
        contrib = [0] * n
        history = []
        stats = self.res.stats

        def pay(i, amt):
            amt = min(amt, stacks[i])
            stacks[i] -= amt
            bets[i] += amt
            contrib[i] += amt
            return amt

        if ante:
            for i in range(n):
                if alive[i]:
                    pay(i, ante)
            bets = [0] * n  # antes are dead money, not part of the street bet
        pay(sb_seat, sb_amt)
        pay(bb_seat, bb_amt)
        history.append((0, sb_seat, "sb", bets[sb_seat]))
        history.append((0, bb_seat, "bb", bets[bb_seat]))

        hrng = random.Random((self.seed * 1000003 + hand_no * 7919) & 0xFFFFFFFFFFFF)
        if self.deck_fn is not None:
            deck = list(self.deck_fn(hand_no))
        else:
            deck = list(var.deck)
            hrng.shuffle(deck)
        holes = [[] for _ in range(n)]
        order = []
        j = btn
        for _ in range(n_alive):
            j = self._next_alive(j, alive)
            order.append(j)
        for _ in range(var.hole_cards):
            for j in order:
                holes[j].append(deck.pop())
        for j in order:
            stats[j].hands += 1
        board = []
        hand_ctx = {"hand_no": hand_no, "sb": sb_amt, "bb": bb_amt, "ante": ante, "button": btn,
                    "sb_seat": sb_seat, "bb_seat": bb_seat, "level": ev.level(hand_no),
                    "alive": alive}
        pf_flags = {"raised": 0, "vpip": set(), "first_raiser": None}

        for si, (street, ncards) in enumerate(var.streets):
            if si > 0:
                if sum(1 for i in range(n) if not folded[i]) < 2:
                    break
                if var.burn and ncards:
                    deck.pop()
                for _ in range(ncards):
                    board.append(deck.pop())
                bets = [0] * n
                if si == 1:
                    for i in range(n):
                        if not folded[i]:
                            stats[i].saw_flop += 1
            if sum(1 for i in range(n) if not folded[i]) < 2:
                break
            first = self._next_alive(bb_seat, alive) if si == 0 else self._next_alive(btn, alive)
            bets = self._betting_round(si, street, first, bets, contrib, folded, holes, board,
                                       history, hand_ctx, pf_flags, bb_amt)

        self._settle(contrib, folded, holes, board, btn, history, hand_no, start)

    def _betting_round(self, si, street, first, bets, contrib, folded, holes, board, history,
                       ctx, pf, bb_amt):
        n, stacks, var, stats = self.n, self.stacks, self.variant, self.res.stats
        current = max(bets) if si == 0 else 0
        if si == 0:
            current = max(current, bb_amt)
        last_raise = bb_amt
        fixed_unit = bb_amt if si <= 1 else 2 * bb_amt
        n_bets = 1 if si == 0 else 0
        full_version = 0
        acted_version = [-1] * n
        pending = set(i for i in range(n) if not folded[i] and stacks[i] > 0)
        ptr = first
        while pending:
            live = [i for i in range(n) if not folded[i]]
            if len(live) < 2:
                break
            # next pending seat from ptr
            i = None
            for k in range(n):
                j = (ptr + k) % n
                if j in pending:
                    i = j
                    break
            if i is None:
                break
            pending.discard(i)
            ptr = (i + 1) % n
            others_can_act = any(j != i and not folded[j] and stacks[j] > 0 for j in range(n))
            to_call = min(max(current - bets[i], 0), stacks[i])
            if to_call == 0 and not others_can_act:
                continue
            can_check = to_call == 0
            raise_right = acted_version[i] != full_version
            can_raise = (stacks[i] > to_call and others_can_act and raise_right
                         and (var.raise_cap is None or n_bets < var.raise_cap))
            max_to = bets[i] + stacks[i]
            if var.betting == "fixed_limit":
                min_to = max_to_cap = current + fixed_unit
            else:
                min_to = current + max(last_raise, bb_amt)
                max_to_cap = max_to
                if var.betting == "pot_limit":
                    max_to_cap = current + sum(contrib) + (current - bets[i])
            min_raise_to = min(min_to, max_to)
            max_raise_to = min(max(max_to_cap, min_raise_to), max_to)
            obs = {
                "seat": i, "num_seats": n, "hand_no": ctx["hand_no"],
                "max_hands": self.event.max_hands, "hands_per_level": self.event.hands_per_level,
                "level": ctx["level"], "sb": ctx["sb"], "bb": ctx["bb"], "ante": ctx["ante"],
                "button": ctx["button"], "sb_seat": ctx["sb_seat"], "bb_seat": ctx["bb_seat"],
                "street": street, "street_index": si, "hole": list(holes[i]), "board": list(board),
                "stacks": list(stacks), "bets": list(bets), "contrib": list(contrib),
                "pot": sum(contrib), "in_hand": [not f for f in folded], "alive": list(ctx["alive"]),
                "to_call": to_call, "min_raise_to": min_raise_to, "max_raise_to": max_raise_to,
                "can_check": can_check, "can_raise": can_raise, "history": list(history),
                "payouts": self.event.payouts, "current_bet": current,
            }
            action = self._ask(i, obs)
            kind, amt = action
            st = stats[i]
            facing = current > bets[i]
            if facing:
                st.face_bet += 1
            if si == 0 and not pf["raised"] and i != ctx["bb_seat"]:
                st.pf_open_opps += 1
                if i == ctx["button"] or i == ctx["sb_seat"]:
                    st.steal_opps += 1
            if si == 0 and pf["raised"] == 1 and pf["first_raiser"] != i:
                st.pf_3bet_opps += 1
            if kind == FOLD:
                folded[i] = True
                st.folds += 1
                if facing:
                    st.fold_to_bet += 1
                pending.discard(i)
                history.append((si, i, FOLD, bets[i]))
            elif kind == CHECK:
                st.checks += 1
                history.append((si, i, CHECK, bets[i]))
            elif kind == CALL:
                self._pay(i, to_call, bets, contrib)
                st.calls += 1
                if si == 0:
                    pf["vpip"].add(i)
                if stacks[i] == 0:
                    st.allins += 1
                history.append((si, i, CALL, bets[i]))
            else:  # RAISE
                self._pay(i, amt - bets[i], bets, contrib)
                if stacks[i] == 0:
                    st.allins += 1
                if current == 0:
                    st.bets += 1
                else:
                    st.raises += 1
                if si == 0:
                    pf["vpip"].add(i)
                    if not pf["raised"]:
                        st.pf_opens += 1
                        pf["first_raiser"] = i
                        if i == ctx["button"] or i == ctx["sb_seat"]:
                            st.steals += 1
                    elif pf["raised"] == 1:
                        st.pf_3bets += 1
                    pf["raised"] += 1
                inc = bets[i] - current
                if inc > 0:
                    if inc >= last_raise or var.betting == "fixed_limit":
                        last_raise = max(inc, last_raise) if var.betting != "fixed_limit" else last_raise
                        full_version += 1
                        n_bets += 1
                    current = bets[i]
                    pending = set(j for j in range(n) if j != i and not folded[j] and stacks[j] > 0)
                history.append((si, i, RAISE, bets[i]))
            acted_version[i] = full_version
        if si == 0:
            for i in pf["vpip"]:
                stats[i].vpip += 1
            raisers = set(h[1] for h in history if h[0] == 0 and h[2] == RAISE)
            for i in raisers:
                stats[i].pfr += 1
        return bets

    def _pay(self, i, amt, bets, contrib):
        amt = min(amt, self.stacks[i])
        self.stacks[i] -= amt
        bets[i] += amt
        contrib[i] += amt

    def _ask(self, i, obs):
        st = self.res.stats[i]
        agent = self.agents[i]
        t0 = time.perf_counter_ns()
        try:
            raw = agent.act(obs)
            err = False
        except Exception:
            raw = None
            err = True
        dt = time.perf_counter_ns() - t0
        st.decisions += 1
        st.time_ns += dt
        if dt > st.max_ns:
            st.max_ns = dt
        b = int(dt / (LAT_BUCKET_US * 1000))
        st.hist[b if b < LAT_BUCKETS else LAT_BUCKETS - 1] += 1
        if err:
            st.errors += 1
            return fallback_action(obs)
        if dt > self.limit_ns:
            st.timeouts += 1
            if self.event.enforce_time_limit and dt > self.enforce_ns:
                return fallback_action(obs)
        action, illegal = legalize(obs, raw, self.event.invalid_raise_policy)
        if illegal:
            st.illegal += 1
        return action

    def _settle(self, contrib, folded, holes, board, btn, history, hand_no, start):
        n, stacks, var = self.n, self.stacks, self.variant
        contenders = [i for i in range(n) if not folded[i]]
        won = [0] * n
        showdown = {}
        if len(contenders) == 1:
            won[contenders[0]] = sum(contrib)
        else:
            scores = {i: var.hand_value(holes[i], board) for i in contenders}
            showdown = {i: list(holes[i]) for i in contenders}
            levels = sorted(set(c for c in contrib if c > 0))
            prev = 0
            for lv in levels:
                slice_ = sum(min(c, lv) - min(c, prev) for c in contrib)
                elig = [i for i in contenders if contrib[i] >= lv]
                if not elig:
                    elig = [i for i in contenders if contrib[i] >= prev] or contenders
                best = max(scores[i] for i in elig)
                winners = [i for i in elig if scores[i] == best]
                winners.sort(key=lambda s: (s - btn - 1) % n)
                share, odd = divmod(slice_, len(winners))
                for k, w in enumerate(winners):
                    won[w] += share + (1 if k < odd else 0)
                prev = lv
            for i in contenders:
                self.res.stats[i].showdowns += 1
                if won[i] > contrib[i]:
                    self.res.stats[i].won_showdown += 1
        for i in range(n):
            stacks[i] += won[i]
            self.res.stats[i].chips_won += stacks[i] - start[i]
        info = {"hand_no": hand_no, "stacks": list(stacks), "showdown": showdown,
                "board": list(board), "winners": [i for i in range(n) if won[i] > contrib[i]],
                "won": won, "history": history}
        for i in range(n):
            if start[i] > 0:
                try:
                    self.agents[i].on_hand_end(info)
                except Exception:
                    self.res.stats[i].errors += 1
        if self.record:
            self.res.hand_logs.append({"hand_no": hand_no, "start": start, "holes": holes,
                                       "board": list(board), "history": history, "won": won,
                                       "end": list(stacks)})


def play_table(event, agents, seed, record=False, time_scale=1.0):
    return Table(event, agents, seed, record=record, time_scale=time_scale).run()
