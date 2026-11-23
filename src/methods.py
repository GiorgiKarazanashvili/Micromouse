from fractions import Fraction

import numpy as np

moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
wall_bits = [1, 2, 4, 8]


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")


# next(s, b) = neighbour of s in direction b if no wall or border is in the way, else s;  next(g, b) = g for g in G
def my_next_state_table(walls, goals):
    H, W = walls.shape
    nxt = np.zeros((H * W, 4), dtype=int)
    for r in range(H):
        for c in range(W):
            s = r * W + c
            for b, (dr, dc) in enumerate(moves):
                nr, nc = r + dr, c + dc
                if (r, c) in goals or walls[r, c] & wall_bits[b] or not (0 <= nr < H and 0 <= nc < W):
                    nxt[s, b] = s
                else:
                    nxt[s, b] = nr * W + nc
    return nxt


# (P_a)_{s,s'} = (1 - 2p) [next(s, a) = s'] + p [next(s, a+) = s'] + p [next(s, a-) = s']
def my_transition_matrices(nxt, slip):
    n = len(nxt)
    P = np.zeros((4, n, n), dtype=object if isinstance(slip, Fraction) else float)
    for a in range(4):
        for b, pr in [(a, 1 - 2 * slip), ((a + 1) % 4, slip), ((a - 1) % 4, slip)]:
            for s in range(n):
                P[a, s, nxt[s, b]] += pr
    return P


# R(s) = -1 for s not in G,  R(g) = 10 for g in G
def my_rewards(n, goal_states, step=-1, goal=10):
    R = np.array([step] * n, dtype=object if isinstance(step, Fraction) else float)
    R[goal_states] = goal
    return R


# Q_k(a, s) = R(s) + gamma (P_a V_k)(s)
def my_bellman_Q(P, R, gamma, V):
    return R[None, :] + gamma * (P @ V)


# V_{k+1}(s) = max_a Q_k(a, s),  stop when ||V_{k+1} - V_k||_inf < tol,  pi(s) = argmax_a Q(a, s)
def my_value_iteration(P, R, gamma, tol=1e-10, verbose=False):
    V = np.zeros(P.shape[1])
    errs = []
    while True:
        V_new = my_bellman_Q(P, R, gamma, V).max(axis=0)
        errs.append(np.abs(V_new - V).max())
        V = V_new
        if verbose:
            print(f"  V{len(errs):<4d}", V.round(3))
        if errs[-1] < tol:
            break
    pol = my_bellman_Q(P, R, gamma, V).argmax(axis=0)
    return V, pol, errs


# (P_pi)_{s,:} = (P_{pi(s)})_{s,:}
def my_policy_matrix(P, pol):
    return P[pol, np.arange(P.shape[1])]


# [A | b] -> row-echelon form with row_i <- row_i - (A_ik / A_kk) row_k,  then x_i = (b_i - sum_{j>i} A_ij x_j) / A_ii
def my_gauss_solve(A, b, verbose=False):
    M = np.column_stack([A, b])
    n = len(b)
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if M[p, k] == 0:
            raise ValueError("singular matrix")
        if p != k:
            M[[k, p]] = M[[p, k]]
            if verbose:
                print(f"  swap R{k + 1} and R{p + 1}")
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            if m != 0:
                M[i] = M[i] - m * M[k]
                if verbose:
                    print(f"  R{i + 1} <- R{i + 1} - ({m}) R{k + 1}")
        if verbose:
            print("\n".join("   " + " ".join(f"{str(x):>12s}" for x in row) for row in M))
    x = np.zeros(n, dtype=M.dtype)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


# x = R(s) + gamma max_a' Q(a', s'),  with max_a' Q(a', g) = R(g) / (1 - gamma) for g in G
# N(s, a) += 1,  Q(a, s) <- Q(a, s) + N(s, a)^(-0.6) (x - Q(a, s)),  a eps-greedy,  s' ~ (P_a)_{s,:}
def my_q_learning(P, R, gamma, episodes, seed=0, eps=0.2, max_len=200, return_visits=False):
    rng = np.random.default_rng(seed)
    n = P.shape[1]
    goal = np.all(P[:, np.arange(n), np.arange(n)] == 1, axis=0)
    Q = np.zeros((4, n))
    Q[:, goal] = R[goal] / (1 - gamma)
    N = np.zeros((4, n))
    cdf = P.astype(float).cumsum(axis=2)
    cdf[:, :, -1] = 1
    starts = np.flatnonzero(~goal)
    for ep in range(episodes):
        s = int(rng.choice(starts))
        for t in range(max_len):
            a = rng.integers(4) if rng.random() < eps else int(Q[:, s].argmax())
            s2 = int(np.searchsorted(cdf[a, s], rng.random(), side="right"))
            x = R[s] + gamma * Q[:, s2].max()
            N[a, s] += 1
            Q[a, s] += N[a, s] ** -0.6 * (x - Q[a, s])
            s = s2
            if goal[s]:
                break
    return (Q, N) if return_visits else Q


# steps to reach G from the start when following pi, S_{t+1} ~ (P_pi(S_t))_{S_t,:}
def my_steps_to_goal(P, pol, start, runs=200, seed=0, max_steps=1000):
    rng = np.random.default_rng(seed)
    n = P.shape[1]
    goal = np.all(P[:, np.arange(n), np.arange(n)] == 1, axis=0)
    cdf = P.astype(float).cumsum(axis=2)
    cdf[:, :, -1] = 1
    steps = []
    for _ in range(runs):
        s, t = start, 0
        while not goal[s] and t < max_steps:
            s = int(np.searchsorted(cdf[pol[s], s], rng.random(), side="right"))
            t += 1
        steps.append(t)
    return np.array(steps)
