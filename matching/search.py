"""matching/search.py -- best-first search with anytime budgets.

Phase 5 search layer. The existing extension search is a greEDY "first improving
trial wins" walk under a hard MCS budget. This module replaces that with a
BEST-FIRST search: a priority queue ordered by an optimistic bound expands the
most promising partial solution first, keeps the best exact solution seen, and
returns it when the budget is exhausted (anytime). Because it stores the best
found so far, a truncated search degrades gracefully instead of returning the
last greedy step.

Design decisions (mine, stated so they can be audited):

* **Budgets are explicit and hard.** ``Budget`` carries expansion, operation and
  (optional) wall-clock caps. Every expansion ticks the budget; the search stops
  the instant a cap is hit and reports ``budget_exhausted``.

* **Anytime.** ``best_exact`` is tracked as soon as it is discovered, so the
  result is always usable. A caller that only needs "good enough now" can set a
  tiny budget.

* **Determinism.** The frontier is ordered by ``(-bound, -exact, key(state))``,
  where ``key`` must be a deterministic, hashable, sortable tuple derived from
  the state. No insertion counters, so run-to-run order is identical.

* **Admissible bound by contract.** The caller supplies ``bound``; when it is a
  true upper bound the search is A* and the first popped completion is optimal.
  When it is not, the search is still well-defined best-first and the result is
  the best exact solution found.

Pure and stateless apart from the optional wall-clock budget.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
)


@dataclass
class Budget:
    """Hard search budget. `max_seconds=None` disables the wall-clock cap so a
    run is time-independent when only expansion/op caps are set."""
    max_expansions: int = 4096
    max_ops: Optional[int] = None
    max_seconds: Optional[float] = None
    expansions: int = 0
    ops: int = 0
    _t0: float = field(default=0.0, repr=False)
    _timed: bool = field(default=False, repr=False)

    def start(self) -> None:
        self.expansions = 0
        self.ops = 0
        self._timed = self.max_seconds is not None
        self._t0 = time.perf_counter() if self._timed else 0.0

    def tick(self, n: int = 1) -> None:
        self.ops += n

    def can_expand(self) -> bool:
        if self.expansions >= self.max_expansions:
            return False
        if self.max_ops is not None and self.ops >= self.max_ops:
            return False
        if self._timed and (time.perf_counter() - self._t0) >= self.max_seconds:
            return False
        return True

    def note_expansion(self) -> None:
        self.expansions += 1


@dataclass(frozen=True)
class SearchResult:
    best_state: Any
    best_score: float
    expanded: int
    frontier_peak: int
    budget_exhausted: bool
    complete: bool   # frontier drained before the budget was hit
    #: [C2] the search stopped because a GOAL was reached, not because the
    #: budget ran out and not because the frontier was drained.
    goal_reached: bool = False


def best_first_search(
    initial: Iterable[Any],
    expand: Callable[[Any], Iterable[Any]],
    score: Callable[[Any], float],
    *,
    bound: Optional[Callable[[Any], float]] = None,
    key: Optional[Callable[[Any], Hashable]] = None,
    is_goal: Optional[Callable[[Any], bool]] = None,
    budget: Optional[Budget] = None,
) -> SearchResult:
    """Best-first search returning the best exact-scored state found.

    expand(state) -> children; score(state) -> exact objective (maximised);
    bound(state)  -> optimistic value (defaults to score); key(state) ->
    deterministic sortable tie-break (defaults to repr(state)).
    """
    bound_is_score = bound is None
    if bound is None:
        bound = score
    if key is None:
        key = repr
    if budget is None:
        budget = Budget()
    budget.start()

    # Heap entries: (-bound, -exact, key, serial, state). `serial` only breaks
    # exact ties among identical keys (identical states), so it never affects
    # which distinct state expands first.
    serial = 0
    heap: List[Tuple[float, float, Hashable, int, Any]] = []
    seen: Dict[Hashable, float] = {}

    def _push(state: Any, s_exact: Optional[float] = None) -> None:
        nonlocal serial
        k = key(state)
        if s_exact is None:
            s_exact = score(state)
        prev = seen.get(k)
        if prev is not None and prev >= s_exact:
            return
        seen[k] = s_exact
        serial += 1
        # [C4] the state is scored ONCE: with the default bound (the score
        # itself) the state was scored twice -- once for the exact value and
        # once for the bound -- which doubled the cost of every push.
        b = s_exact if bound_is_score else bound(state)
        heapq.heappush(heap, (-float(b), -float(s_exact), k, serial, state))

    for st in initial:
        _push(st)

    best_state: Any = None
    best_score = float("-inf")
    frontier_peak = len(heap)
    complete = True
    goal_reached = False
    stopped_by_budget = False

    while heap:
        if not budget.can_expand():
            complete = False
            stopped_by_budget = True
            break
        _neg_bound, neg_exact, _k, _serial, state = heapq.heappop(heap)
        exact = -neg_exact
        if exact > best_score:
            best_score = exact
            best_state = state
        if is_goal is not None and is_goal(state):
            # [C2] the goal is NOT a drain of the frontier and NOT a budget
            # exhaustion: `complete` says the frontier was drained, and the
            # budget did not stop the search.
            complete = False
            goal_reached = True
            break
        budget.note_expansion()
        for child in expand(state):
            if not budget.can_expand():
                # [C1] THE OPERATION CAP IS HARD.  It used to be tested only
                # BETWEEN the expansions, so one expansion could push its
                # whole subtree past the cap: the budget said 1 operation
                # and the search did 5.
                complete = False
                stopped_by_budget = True
                break
            budget.tick()
            _push(child)
        if stopped_by_budget:
            break
        if len(heap) > frontier_peak:
            frontier_peak = len(heap)

    if best_state is None and heap:
        # Degenerate: budget too small to even pop once.  [C3] The best
        # EXACT score in the frontier is returned -- the fallback used the
        # BOUND (the top of the heap), which with the default bound is the
        # bound and not the score, so the search returned the WORST-scoring
        # state in the frontier.
        best_state = max(heap, key=lambda e: (-e[1], e[2]))[4]
        best_score = -max(heap, key=lambda e: (-e[1], e[2]))[1]

    return SearchResult(
        best_state=best_state,
        best_score=best_score,
        expanded=budget.expansions,
        frontier_peak=frontier_peak,
        budget_exhausted=stopped_by_budget,
        complete=complete,
        goal_reached=goal_reached,
    )


def best_first_extension(
    seed: FrozenSet,
    candidates: FrozenSet,
    objective: Callable[[FrozenSet], float],
    *,
    bound: Optional[Callable[[FrozenSet], float]] = None,
    budget: Optional[Budget] = None,
) -> SearchResult:
    """Best-first growth of a seed set by adding one candidate at a time.

    The state is a frozenset of elements (nodes and/or edges). Expanding a
    state adds each not-yet-present candidate once, so the lattice is the
    subset lattice of `candidates` above `seed`. `objective` is maximised; the
    default `bound` is the objective (exhaustive best-first, no pruning).

    Determinism: children are pushed in sorted candidate order; the frontier
    tie-break key is the sorted element list.
    """
    ordered = tuple(sorted(candidates, key=repr))

    def _expand(state: FrozenSet) -> Iterable[FrozenSet]:
        for el in ordered:
            if el not in state:
                yield state | {el}

    def _key(state: FrozenSet) -> Hashable:
        return tuple(sorted(state, key=repr))

    return best_first_search(
        [seed],
        _expand,
        objective,
        bound=bound,
        key=_key,
        budget=budget,
    )
