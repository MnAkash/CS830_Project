"""
Physical-adversary heuristics for Akash's PACT experiments.

Adversary agents can chase social agents, walk randomly, block goals, or mix
those behaviors.

The default strategy is A* shortest-path pursuit with a Manhattan-distance
heuristic. PACT imports shared PPO/evaluation utilities, but keeps these
physical threat definitions in this package so Akash's implementation is
separate from the shared baseline.
"""

import heapq
from collections import deque
import numpy as np

# ─── POGEMA action deltas ────────────────────────────────────────
#   0 = stay, 1 = up (row−1), 2 = down (row+1),
#   3 = left (col−1), 4 = right (col+1)
ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

BASE_STRATEGIES = ("astar_pursuit", "random_walk", "goal_blocking")
ADVERSARY_STRATEGIES = BASE_STRATEGIES + ("mixed",)


def normalize_strategy(strategy):
    """Return a canonical physical-adversary strategy name."""
    aliases = {
        "astar": "astar_pursuit",
        "a_star": "astar_pursuit",
        "a*_pursuit": "astar_pursuit",
        "pursuit": "astar_pursuit",
        "greedy": "astar_pursuit",
        "random": "random_walk",
        "random-walk": "random_walk",
        "goal": "goal_blocking",
        "block": "goal_blocking",
        "goal-blocking": "goal_blocking",
        "mix": "mixed",
        "mixed_strategy": "mixed",
        "mixed-strategy": "mixed",
        "mixed_adversary": "mixed",
        "mixed-adversary": "mixed",
    }
    key = str(strategy).strip().lower().replace(" ", "_")
    key = aliases.get(key, key)
    if key not in ADVERSARY_STRATEGIES:
        valid = ", ".join(ADVERSARY_STRATEGIES)
        raise ValueError(f"Unknown adversary strategy '{strategy}'. Valid: {valid}")
    return key


def valid_move_actions(start, obstacles, include_stay=True):
    """Return actions that keep an agent inside the map and off obstacles."""
    H, W = obstacles.shape
    sr, sc = start
    actions = [0] if include_stay else []
    for action in [1, 2, 3, 4]:
        dr, dc = ACTION_DELTAS[action]
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] < 0.5:
            actions.append(action)
    return actions


def balanced_closest_assignment(adv_positions, target_positions, active_indices):
    """Assign adversaries to active targets with closest-first load balancing."""
    n_adv = len(adv_positions)
    if not active_indices:
        return {}

    max_per = max(1, -(-n_adv // len(active_indices)))
    load = {i: 0 for i in active_indices}
    assignments = {}

    pairs = []
    for adv_idx, (ar, ac) in enumerate(adv_positions):
        for target_idx in active_indices:
            tr, tc = target_positions[target_idx]
            pairs.append((abs(ar - tr) + abs(ac - tc), adv_idx, target_idx))
    pairs.sort()

    for _, adv_idx, target_idx in pairs:
        if adv_idx in assignments:
            continue
        if load[target_idx] >= max_per:
            continue
        assignments[adv_idx] = target_idx
        load[target_idx] += 1

    for adv_idx in range(n_adv):
        if adv_idx not in assignments:
            best = min(active_indices, key=lambda target_idx: load[target_idx])
            assignments[adv_idx] = best
            load[best] += 1
    return assignments


def astar_next_step(start, goal, obstacles):
    """A* shortest path with Manhattan distance heuristic.

    Returns the first-step action (1-4) to move from *start* toward *goal*,
    or 0 if the goal is unreachable.  A* expands fewer nodes than BFS on
    average because the heuristic guides the search toward the goal.
    """
    H, W = obstacles.shape
    sr, sc = start
    gr, gc_ = goal
    if (sr, sc) == (gr, gc_):
        return 0

    def h(r, c):
        return abs(r - gr) + abs(c - gc_)

    # Priority queue entries: (f, g, row, col, first_action)
    # f = g + h;  g = steps so far
    open_heap = []
    for a in [1, 2, 3, 4]:
        dr, dc = ACTION_DELTAS[a]
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] < 0.5:
            if (nr, nc) == (gr, gc_):
                return a
            heapq.heappush(open_heap, (1 + h(nr, nc), 1, nr, nc, a))

    closed = set()
    closed.add((sr, sc))

    while open_heap:
        f, g, r, c, first_a = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        closed.add((r, c))

        for a in [1, 2, 3, 4]:
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] < 0.5:
                if (nr, nc) == (gr, gc_):
                    return first_a
                if (nr, nc) not in closed:
                    heapq.heappush(open_heap,
                                   (g + 1 + h(nr, nc), g + 1, nr, nc, first_a))
    return 0


# Keep BFS as a fallback / comparison baseline
def bfs_next_step(start, goal, obstacles):
    """BFS shortest path; return the first-step action (1-4) or 0 if stuck."""
    H, W = obstacles.shape
    sr, sc = start
    gr, gc_ = goal
    if (sr, sc) == (gr, gc_):
        return 0
    visited = set()
    visited.add((sr, sc))
    q = deque()  # (row, col, first_action)
    for a in [1, 2, 3, 4]:
        dr, dc = ACTION_DELTAS[a]
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] < 0.5:
            if (nr, nc) == (gr, gc_):
                return a
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, a))
    while q:
        r, c, first_a = q.popleft()
        for a in [1, 2, 3, 4]:
            dr, dc = ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] < 0.5:
                if (nr, nc) == (gr, gc_):
                    return first_a
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, first_a))
    return 0


def random_walk_actions(adv_positions, obstacles, rng=None):
    """Move each adversary randomly among valid neighboring cells."""
    if rng is None:
        rng = np.random.default_rng()
    actions = []
    for pos in adv_positions:
        valid = valid_move_actions(pos, obstacles, include_stay=True)
        if len(valid) == 1:
            actions.append(0)
        else:
            actions.append(int(rng.choice(valid)))
    return actions


def pursuit_actions(adv_positions, social_positions, social_done, obstacles):
    """A* pure pursuit toward assigned social agents' current positions."""
    n_adv = len(adv_positions)
    active = [i for i in range(len(social_positions)) if not social_done[i]]
    if not active:
        return [0] * n_adv

    assignments = balanced_closest_assignment(
        adv_positions, social_positions, active,
    )
    actions = []
    for adv_idx, (ar, ac) in enumerate(adv_positions):
        target_idx = assignments[adv_idx]
        sr, sc = social_positions[target_idx]
        if (ar, ac) == (sr, sc):
            actions.append(0)
        else:
            actions.append(astar_next_step((ar, ac), (sr, sc), obstacles))
    return actions


def goal_blocking_actions(adv_positions, social_goals, social_done, obstacles):
    """A* movement toward assigned active social-agent goals."""
    n_adv = len(adv_positions)
    active = [i for i in range(len(social_goals)) if not social_done[i]]
    if not active:
        return [0] * n_adv

    assignments = balanced_closest_assignment(adv_positions, social_goals, active)
    actions = []
    for adv_idx, (ar, ac) in enumerate(adv_positions):
        target_idx = assignments[adv_idx]
        gr, gc_ = social_goals[target_idx]
        if (ar, ac) == (gr, gc_):
            actions.append(0)
        else:
            actions.append(astar_next_step((ar, ac), (gr, gc_), obstacles))
    return actions


def adversary_actions(adv_positions, social_positions, social_goals,
                      social_done, obstacles, strategy="astar_pursuit",
                      rng=None):
    """Return physical-adversary actions for the selected strategy.

    Strategies:
      - ``astar_pursuit``: chase assigned social agents' current positions.
      - ``random_walk``: move randomly among valid non-obstacle cells.
      - ``goal_blocking``: move to assigned social-agent goals and camp there.
            - ``mixed``: sample one of the above behaviors at each decision step.

    ``astar_pursuit`` is the historical shared-baseline behavior.

    A* pursuit assignment: balanced closest-first.
    Assignment:  balanced closest-first.
      1. Compute Manhattan distance from every adversary to every *active*
         (not yet finished) social agent.
      2. Greedily assign each adversary to its closest available target.
      3. Load-balance: no agent receives more than ceil(n_adv / n_active)
         pursuers unless there are no other options.

    Behaviour: A* toward the social agent's **current position** — pure
    pursuit.  Adversaries never camp on goal cells; they always converge
    on the agent itself, creating dynamic pressure that forces evasive
    manoeuvring.
    """
    strategy = normalize_strategy(strategy)
    if rng is None:
        rng = np.random.default_rng()
    if strategy == "mixed":
        strategy = str(rng.choice(BASE_STRATEGIES))
    if strategy == "random_walk":
        return random_walk_actions(adv_positions, obstacles, rng=rng)
    if strategy == "goal_blocking":
        return goal_blocking_actions(adv_positions, social_goals, social_done, obstacles)
    return pursuit_actions(adv_positions, social_positions, social_done, obstacles)
