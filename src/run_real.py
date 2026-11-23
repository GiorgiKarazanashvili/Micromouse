import os

import numpy as np
import matplotlib.pyplot as plt

from maze_io import load_maze, flood_fill, N, E, S, W
from methods import (report, my_next_state_table, my_transition_matrices, my_rewards, my_value_iteration,
                     my_policy_matrix, my_gauss_solve, my_q_learning, my_steps_to_goal)

DATA_DIR = os.path.join("data", "mazes_classic")
os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(0)
slip, gamma = 0.1, 0.95
synthetic = ["empty.txt", "long.txt", "wiggly1.txt", "zigzag.txt"]
unseen = "AAMC24Maze.txt"


def build(walls, goals):
    P = my_transition_matrices(my_next_state_table(walls, goals), slip)
    R = my_rewards(walls.size, [r * walls.shape[1] + c for r, c in goals])
    return P, R


print("Cleaning")
files = sorted(os.listdir(DATA_DIR))
mazes = {f: load_maze(os.path.join(DATA_DIR, f)) for f in files}
print("  files:", len(files))
unreachable = [f for f in files if flood_fill(mazes[f][0], mazes[f][2])[mazes[f][1]] < 0]
for f in unreachable + synthetic:
    del mazes[f]
print("  removed, goal unreachable:", unreachable)
print("  removed, synthetic test layouts:", synthetic)
seen = {}
for f in sorted(mazes):
    key = mazes[f][0].tobytes()
    if key in seen:
        print(f"  removed, duplicate of {seen[key]}: {f}")
        del mazes[f]
    else:
        seen[key] = f
train = [f for f in sorted(mazes) if f != unseen]
print("  clean mazes:", len(mazes), " training:", len(train), " unseen test maze:", unseen)

walls, start, goals = mazes[unseen]
fig, ax = plt.subplots(figsize=(5, 5))
for r in range(16):
    for c in range(16):
        if walls[r, c] & N:
            ax.plot([c - .5, c + .5], [r - .5, r - .5], "k", lw=2)
        if walls[r, c] & S:
            ax.plot([c - .5, c + .5], [r + .5, r + .5], "k", lw=2)
        if walls[r, c] & W:
            ax.plot([c - .5, c - .5], [r - .5, r + .5], "k", lw=2)
        if walls[r, c] & E:
            ax.plot([c + .5, c + .5], [r - .5, r + .5], "k", lw=2)
ax.plot(start[1], start[0], "gs", ms=10)
for r, c in goals:
    ax.plot(c, r, "r*", ms=12)
ax.set_title(f"Parsed maze {unseen}")
ax.set_xticks([])
ax.set_yticks([])
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join("figures", "parsed_maze.png"), dpi=150)

print("\nLibrary comparison on every clean maze")
d_vi, d_np, row_err = 0.0, 0.0, 0.0
for f in sorted(mazes):
    P, R = build(mazes[f][0], mazes[f][2])
    V, pol, errs = my_value_iteration(P, R, gamma, tol=1e-10)
    A = np.eye(len(R)) - gamma * my_policy_matrix(P, pol)
    V_ge = my_gauss_solve(A, R)
    row_err = max(row_err, np.abs(P.sum(axis=2) - 1).max())
    d_vi = max(d_vi, np.abs(V - V_ge).max())
    d_np = max(d_np, np.abs(V_ge - np.linalg.solve(A, R)).max())
print("  max |row sum - 1|:", row_err)
print("  max |V value iteration - V Gauss|:", d_vi)
print("  max |V Gauss - V numpy.solve|:", d_np)
P, R = build(walls, goals)
V, pol, errs = my_value_iteration(P, R, gamma, tol=1e-10)
P_pi = my_policy_matrix(P, pol)
V_ge = my_gauss_solve(np.eye(256) - gamma * P_pi, R)
report(f"  {unseen}: VI vs Gauss", V, V_ge)
report(f"  {unseen}: Gauss vs numpy.solve", V_ge, np.linalg.solve(np.eye(256) - gamma * P_pi, R))
e = np.array(errs)
e = e[e > 1e-6]
print("  max error ratio:", (e[1:] / e[:-1]).max(), " gamma:", gamma)

print("\nQ5")
episodes = 30000
print("  training mazes, Q-learning with", episodes, "episodes, steps from the start (mean over 200 runs):")
for i in rng.choice(len(train), 10, replace=False):
    w, st, g = mazes[train[i]]
    P_t, R_t = build(w, g)
    _, pol_t, _ = my_value_iteration(P_t, R_t, gamma)
    Q = my_q_learning(P_t, R_t, gamma, episodes=episodes, seed=0)
    steps = my_steps_to_goal(P_t, Q.argmax(axis=0), st[0] * 16 + st[1])
    print(f"   {train[i]:32s} shortest path {flood_fill(w, g)[st]:3d}   value iteration "
          f"{my_steps_to_goal(P_t, pol_t, st[0] * 16 + st[1]).mean():6.1f}   Q-learning {steps.mean():6.1f}"
          f"   reached goal in {np.mean(steps < 1000):.0%} of runs")

s0 = start[0] * 16 + start[1]
Q, visits = my_q_learning(P, R, gamma, episodes=episodes, seed=0, return_visits=True)
steps_q = my_steps_to_goal(P, Q.argmax(axis=0), s0)
steps_vi = my_steps_to_goal(P, pol, s0)
print(f"  {unseen}: shortest path {flood_fill(walls, goals)[start]} cells")
print(f"  {unseen}: Q-learning {steps_q.mean():.1f} steps, value iteration {steps_vi.mean():.1f} steps"
      f" (mean over {len(steps_q)} runs)")
print(f"  {unseen}: max |max_a Q - V| = {np.abs(Q.max(axis=0) - V).max():.3f}")

print("\nAssumptions")
eig = np.linalg.eigvals(P_pi)
print("  A3 max |row sum - 1| over all mazes:", row_err)
print("  A4 min |eigenvalue of I - gamma P_pi|:", np.abs(1 - gamma * eig).min().round(6),
      " cond(I - gamma P_pi):", np.linalg.cond(np.eye(256) - gamma * P_pi).round(2))
print("  A5 max |R|:", np.abs(R).max())
print("  A6 states, actions:", len(R), 4)
print("  A7 mazes removed with unreachable goal:", len(unreachable))
reach = (flood_fill(walls, goals) > 0).ravel()
print("  A9 min N(s, a) over reachable cells:", visits[:, reach].min())
