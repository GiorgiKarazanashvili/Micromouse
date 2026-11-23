import numpy as np

from methods import moves, my_transition_matrices, my_rewards

wall_cells = {(1, 1), (1, 2), (1, 3), (3, 1), (3, 3)}


def build_maze_mdp(size=5, slip=0.1, gamma=0.95):
    goal = (size - 1, size - 1)
    nxt = np.zeros((size * size, 4), dtype=int)
    for s in range(size * size):
        r, c = divmod(s, size)
        for b, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if (r, c) == goal or not (0 <= nr < size and 0 <= nc < size) or (nr, nc) in wall_cells:
                nxt[s, b] = s
            else:
                nxt[s, b] = nr * size + nc
    P = my_transition_matrices(nxt, slip)
    R = my_rewards(size * size, [goal[0] * size + goal[1]])
    return P, R, gamma
