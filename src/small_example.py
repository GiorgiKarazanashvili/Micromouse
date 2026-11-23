from fractions import Fraction

import numpy as np

from maze_io import parse_maze
from methods import (report, my_next_state_table, my_transition_matrices, my_rewards, my_policy_matrix,
                     my_gauss_solve, my_value_iteration, my_q_learning)

maze = """\
o---o---o---o
|           |
o   o---o   o
|   | G     |
o   o---o   o
| S         |
o---o---o---o
"""
walls, start, goals = parse_maze(maze)
n = walls.size
goal_states = [r * 3 + c for r, c in goals]
slip, gamma = Fraction(1, 10), Fraction(9, 10)

print("Maze")
print(maze)
print("  wall matrix M:\n", walls)
print("  start:", start, " goal:", goals)

nxt = my_next_state_table(walls, goals)
print("\n  next(s, b) for b = up, right, down, left:")
for s in range(n):
    print(f"   s = {s}:", nxt[s].tolist())

P = my_transition_matrices(nxt, slip)
R = my_rewards(n, goal_states, step=Fraction(-1), goal=Fraction(10))
for a in range(4):
    print(f"\n  P_{a}:")
    print("\n".join("   " + " ".join(f"{str(x):>5s}" for x in row) for row in P[a]))
print("\n  R:", [str(x) for x in R])

print("\nQ3")
print("  row sums of every P_a:", sorted({str(x) for x in P.sum(axis=2).ravel()}))

print("\nQ1")
pol = np.array([1, 1, 2, 0, 0, 3, 1, 1, 0])
print("  fixed policy (0 up, 1 right, 2 down, 3 left):\n", pol.reshape(3, 3))
P_pi = my_policy_matrix(P, pol)
I = np.array([[Fraction(int(i == j)) for j in range(n)] for i in range(n)], dtype=object)
A = I - gamma * P_pi
print("\n  A = I - gamma P_pi:")
print("\n".join("   " + " ".join(f"{str(x):>7s}" for x in row) for row in A))
print("\n  Gaussian elimination on [A | R]:")
V_pi = my_gauss_solve(A, R, verbose=True)
print("\n  V_pi:", [str(x) for x in V_pi])
print("  V_pi:", V_pi.astype(float).round(4))
print("  (I - gamma P_pi) V_pi - R:", [str(x) for x in A @ V_pi - R])
report("  Gauss vs numpy.solve", V_pi, np.linalg.solve(A.astype(float), R.astype(float)))

print("\nQ2")
Pf, Rf, gf = P.astype(float), R.astype(float), float(gamma)
V, pol_vi, errs = my_value_iteration(Pf, Rf, gf, tol=1e-10, verbose=True)
e = np.array(errs)
e = e[e > 1e-6]
print("  steps:", len(errs))
print("  max error ratio:", (e[1:] / e[:-1]).max(), " gamma:", gf)
report("  value iteration vs V_pi", V, V_pi)

print("\nQ4")
for episodes in [1000, 10000, 100000]:
    Q = my_q_learning(Pf, Rf, gf, episodes=episodes, seed=0)
    print(f"  episodes {episodes:6d}: max |max_a Q - V| = {np.abs(Q.max(axis=0) - V).max():.3f}")
