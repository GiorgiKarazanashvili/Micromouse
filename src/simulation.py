import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(0)

P3 = np.array([[0.2, 0.8, 0.0],
               [0.0, 0.2, 0.8],
               [0.0, 0.0, 1.0]])
R3 = np.array([-1.0, -1.0, 0.0])
g3 = 0.9
V = np.zeros(3)
print("Hand example")
for k in range(1, 3):
    V = R3 + g3 * P3 @ V
    print(f"  value iteration V{k}:", V.round(4))
print("  exact V:", np.linalg.solve(np.eye(3) - g3 * P3, R3).round(4))

H = W = 5
nS = H * W
gamma, slip = 0.95, 0.1
walls = {(1, 1), (1, 2), (1, 3), (3, 1), (3, 3)}
goal = (4, 4)
g = goal[0] * W + goal[1]
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def step(s, a):
    r, c = divmod(s, W)
    if (r, c) == goal:
        return s
    nr, nc = r + moves[a][0], c + moves[a][1]
    if not (0 <= nr < H and 0 <= nc < W) or (nr, nc) in walls:
        return s
    return nr * W + nc


P = np.zeros((4, nS, nS))
for s in range(nS):
    for a in range(4):
        for b, pr in [(a, 1 - 2 * slip), ((a + 1) % 4, slip), ((a - 1) % 4, slip)]:
            P[a, s, step(s, b)] += pr
R = np.full(nS, -1.0)
R[g] = 10.0
free = [s for s in range(nS) if divmod(s, W) not in walls]
start_cells = [s for s in free if s != g]

print("\nQ1")
print("  max |row sum - 1|:", np.abs(P.sum(axis=2) - 1).max())

V = np.zeros(nS)
hist = []
for it in range(2000):
    V_new = R + gamma * (P @ V).max(axis=0)
    hist.append(V_new)
    if np.abs(V_new - V).max() < 1e-12:
        break
    V = V_new
V_star = V_new
pol = (P @ V_star).argmax(axis=0)
err = np.array([np.abs(h - V_star).max() for h in hist])
print("\nQ4")
print("  iterations:", it + 1)
print("  max error ratio:", (err[1:40] / err[:39]).max().round(4), " gamma:", gamma)

k = np.arange(len(err))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(k, err + 1e-16, label=r"$\|V_k - V^*\|_\infty$")
ax.semilogy(k, err[0] * gamma ** k, "--", label=r"$\gamma^k \|V_0 - V^*\|_\infty$")
ax.set_xlabel("iteration k")
ax.set_ylabel("max-norm error")
ax.set_title("Value iteration error shrinks at least by a factor gamma")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join("figures", "vi_convergence.png"), dpi=150)

P_pi = P[pol, np.arange(nS)]
V_exact = np.linalg.solve(np.eye(nS) - gamma * P_pi, R)
eig = np.linalg.eigvals(P_pi)
print("\nQ2 / Q3")
print("  max |V_VI - V_exact|:", np.abs(V_star - V_exact).max())
print("  max |eigenvalue of P_pi|:", np.abs(eig).max().round(6))
print("  min |eigenvalue of I - gamma P_pi|:", np.abs(1 - gamma * eig).min().round(6))
print("  V(goal):", V_exact[g].round(4), " R(goal)/(1-gamma):", round(R[g] / (1 - gamma), 4))

Phi = np.zeros(nS)
for s in free:
    r, c = divmod(s, W)
    Phi[s] = -(abs(r - goal[0]) + abs(c - goal[1]))
R_shaped = R[None, :] + gamma * P @ Phi - Phi[None, :]
V_shaped = np.zeros(nS)
for _ in range(2000):
    V_shaped = (R_shaped + gamma * P @ V_shaped).max(axis=0)
Q_star = R[None, :] + gamma * P @ V_star
Q_shaped = R_shaped + gamma * P @ V_shaped
print("\nReward shaping")
print("  max |Q_shaped - (Q* - Phi)|:", np.abs(Q_shaped - (Q_star - Phi[None, :]))[:, free].max())
print("  same optimal policy:", bool(np.all(Q_shaped.argmax(axis=0)[free] == pol[free])))

samples = rng.normal(3.0, 2.0, size=1000)
q = 0.0
for n, x in enumerate(samples, start=1):
    q = q + (x - q) / n
print("\nQ5")
print("  running estimate:", round(q, 6), " sample mean:", round(samples.mean(), 6))

V_goal = R[g] / (1 - gamma)
Q = np.zeros((nS, 4))
visits = np.zeros((nS, 4))
eps, n_episodes, max_len = 0.2, 20000, 200
for ep in range(n_episodes):
    s = rng.choice(start_cells)
    for t in range(max_len):
        a = rng.integers(4) if rng.random() < eps else int(Q[s].argmax())
        b = rng.choice([a, (a + 1) % 4, (a - 1) % 4], p=[1 - 2 * slip, slip, slip])
        s2 = step(s, b)
        target = R[s] + gamma * (V_goal if s2 == g else Q[s2].max())
        visits[s, a] += 1
        Q[s, a] += visits[s, a] ** -0.6 * (target - Q[s, a])
        s = s2
        if s == g:
            break
learned = Q.argmax(axis=1)
optimal = [abs(Q_star[learned[s], s] - Q_star[:, s].max()) < 1e-6 for s in start_cells]
print("  greedy action optimal in", sum(optimal), "/", len(start_cells), "cells")
print("  max |Q - Q*|:", np.abs(Q[start_cells] - Q_star[:, start_cells].T).max().round(3))
