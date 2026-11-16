import numpy as np

N, E, S, W = 1, 2, 4, 8
DIRS = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}


def parse_maze(text):
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    n_rows = (len(lines) - 1) // 2
    n_cols = (len(lines[0]) - 1) // 4
    walls = np.zeros((n_rows, n_cols), dtype=int)
    start, goals = None, []
    for r in range(n_rows):
        top, mid, bot = lines[2 * r], lines[2 * r + 1], lines[2 * r + 2]
        for c in range(n_cols):
            x = 4 * c
            if top[x + 1:x + 4] == "---":
                walls[r, c] |= N
            if bot[x + 1:x + 4] == "---":
                walls[r, c] |= S
            if len(mid) > x and mid[x] == "|":
                walls[r, c] |= W
            if len(mid) > x + 4 and mid[x + 4] == "|":
                walls[r, c] |= E
            label = mid[x + 1:x + 4].strip()
            if label == "S":
                start = (r, c)
            elif label == "G":
                goals.append((r, c))
    return walls, start, goals


def load_maze(path):
    with open(path) as f:
        return parse_maze(f.read())


def neighbours(walls, r, c):
    out = []
    for bit, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if not walls[r, c] & bit and 0 <= nr < walls.shape[0] and 0 <= nc < walls.shape[1]:
            out.append((nr, nc))
    return out


def flood_fill(walls, goals):
    dist = -np.ones(walls.shape, dtype=int)
    for g in goals:
        dist[g] = 0
    frontier = list(goals)
    while frontier:
        nxt = []
        for r, c in frontier:
            for nr, nc in neighbours(walls, r, c):
                if dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + 1
                    nxt.append((nr, nc))
        frontier = nxt
    return dist
