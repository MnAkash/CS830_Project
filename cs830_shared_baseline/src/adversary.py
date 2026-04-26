"""
Shared adversary heuristic for physical attacker agents.

Adversary agents chase social agents using A* shortest-path pursuit
with Manhattan distance heuristic. Assignment is balanced closest-first
so pressure is distributed evenly.

Used by:
  - ppo_mapf.py       (training with physical adversaries)
  - adv_train.py      (robust training)
  - visualize.py      (animation)
  - evaluate_fragility.py / scalability.py  (evaluation)
"""

import heapq
from collections import deque
import numpy as np

# ─── POGEMA action deltas ────────────────────────────────────────
#   0 = stay, 1 = up (row−1), 2 = down (row+1),
#   3 = left (col−1), 4 = right (col+1)
ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


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


def adversary_actions(adv_positions, social_positions, social_goals,
                      social_done, obstacles):
    """Pure-pursuit adversary strategy — chase the nearest social agent.

    Assignment:  balanced closest-first.
      1. Compute Manhattan distance from every adversary to every *active*
         (not yet finished) social agent.
      2. Greedily assign each adversary to its closest available target.
      3. Load-balance: no agent receives more than ceil(n_adv / n_active)
         pursuers unless there are no other options.

    Behaviour:  BFS toward the social agent's **current position** — pure
    pursuit.  Adversaries never camp on goal cells; they always converge
    on the agent itself, creating dynamic pressure that forces evasive
    manoeuvring.
    """
    n_adv = len(adv_positions)
    n_social = len(social_positions)

    active = [i for i in range(n_social) if not social_done[i]]
    if not active:
        return [0] * n_adv

    # ── balanced closest-first assignment ─────────────────────────
    max_per = max(1, -(-n_adv // len(active)))  # ceil division
    load = {i: 0 for i in active}
    assignments = {}  # adv_idx → social_idx

    pairs = []
    for j in range(n_adv):
        ar, ac = adv_positions[j]
        for si in active:
            sr, sc = social_positions[si]
            pairs.append((abs(ar - sr) + abs(ac - sc), j, si))
    pairs.sort()

    for _, j, si in pairs:
        if j in assignments:
            continue
        if load[si] >= max_per:
            continue
        assignments[j] = si
        load[si] += 1

    # fallback: any unassigned adversary → least-loaded active agent
    for j in range(n_adv):
        if j not in assignments:
            best = min(active, key=lambda si: load[si])
            assignments[j] = best
            load[best] += 1

    # ── A* toward each assigned social agent ─────────────────────
    actions = []
    for j, (ar, ac) in enumerate(adv_positions):
        ti = assignments[j]
        sr, sc = social_positions[ti]
        if (ar, ac) == (sr, sc):
            actions.append(0)  # on top of agent — stay
        else:
            actions.append(astar_next_step((ar, ac), (sr, sc), obstacles))
    return actions
