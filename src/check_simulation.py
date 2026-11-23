import os

import numpy as np
import matplotlib.pyplot as plt

from methods import report, my_value_iteration, my_gauss_solve, my_q_learning, my_steps_to_goal
from simulate import build_maze_mdp

os.makedirs("figures", exist_ok=True)

print("Simulation test")
P, R, gamma = build_maze_mdp(size=5, slip=0.1, gamma=0.95)   # P[a, s, s']
print(np.abs(P.sum(axis=2) - 1).max())              # Q3
V, pol, errs = my_value_iteration(P, R, gamma, tol=1e-10)    # errs[k] = ||V_k - V_k-1||_inf
P_pi = P[pol, np.arange(P.shape[1])]
V_ge = my_gauss_solve(np.eye(len(R)) - gamma * P_pi, R)       # Q1, Q2
report("value iteration vs Gauss", V, V_ge)
report("Gauss vs numpy.solve", V_ge, np.linalg.solve(np.eye(len(R)) - gamma * P_pi, R))
e = np.array(errs); e = e[e > 1e-6]                 # ignore rounding-level steps
print((e[1:] / e[:-1]).max(), gamma)                # contraction: ratio <= gamma
Q = my_q_learning(P, R, gamma, episodes=20000, seed=0)       # Q4
print(np.abs(Q.max(axis=0) - V).max(), np.mean(Q.argmax(axis=0) == pol))

print("\nQ-learning against value iteration")
Q_star = R[None, :] + gamma * P @ V
cells = [s for s in range(len(R)) if s != 24]
episodes = [500, 5000, 50000]
q_err = []
for ep in episodes:
    Q, N = my_q_learning(P, R, gamma, episodes=ep, seed=0, return_visits=True)
    q_err.append(np.abs(Q.max(axis=0) - V).max())
    learned = Q.argmax(axis=0)
    optimal = np.mean([Q_star[learned[s], s] >= Q_star[:, s].max() - 1e-6 for s in cells])
    print(f"  episodes {ep:6d}: max |max_a Q - V| = {q_err[-1]:.3f}   greedy action optimal in {optimal:.0%} of cells"
          f"   min N(s, a) = {N[:, cells].min():.0f}")
print("  log-log slope:", np.polyfit(np.log(episodes), np.log(q_err), 1)[0].round(2))
print("  steps from (0, 0): Q-learning", my_steps_to_goal(P, learned, 0).mean(),
      " value iteration", my_steps_to_goal(P, pol, 0).mean())

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.loglog(episodes, q_err, "o-")
ax.set_xlabel("Q-learning episodes")
ax.set_ylabel(r"$\max_s |\max_a Q(a,s) - V^*(s)|$")
ax.set_title("Q-learning error against the value-iteration values")
fig.tight_layout()
fig.savefig(os.path.join("figures", "qlearning_error.png"), dpi=150)

print("\nStress test")
P9, R9, gamma9 = build_maze_mdp(size=5, slip=0.1, gamma=0.999)
_, _, errs9 = my_value_iteration(P9, R9, gamma9, tol=1e-10)
print("  value iteration steps: gamma = 0.95:", len(errs), "  gamma = 0.999:", len(errs9))
print("  predicted log(tol / ||V1 - V0||) / log(gamma):", round(np.log(1e-10 / errs[0]) / np.log(gamma)),
      round(np.log(1e-10 / errs9[0]) / np.log(gamma9)))

print("\nAssumptions")
eig = np.linalg.eigvals(P_pi)
print("  A3 max |row sum - 1|:", np.abs(P.sum(axis=2) - 1).max())
print("  A4 min |eigenvalue of I - gamma P_pi|:", np.abs(1 - gamma * eig).min().round(6),
      " cond(I - gamma P_pi):", np.linalg.cond(np.eye(len(R)) - gamma * P_pi).round(2))
print("  A5 max |R|:", np.abs(R).max())
print("  A6 states, actions:", len(R), 4)
print("  A7 steps from (0, 0) to the goal under pi, max over 200 runs:", my_steps_to_goal(P, pol, 0).max())
print("  A9 min N(s, a), 20000 episodes:", my_q_learning(P, R, gamma, 20000, seed=0, return_visits=True)[1][:, cells].min())
