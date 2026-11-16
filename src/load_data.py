import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maze_io import load_maze, flood_fill, neighbours, N, E, S, W

DATA_DIR = os.path.join("data", "mazes_classic")
os.makedirs("figures", exist_ok=True)

rows = []
for name in sorted(os.listdir(DATA_DIR)):
    walls, start, goals = load_maze(os.path.join(DATA_DIR, name))
    dist = flood_fill(walls, goals)
    rows.append(dict(
        maze=name,
        rows=walls.shape[0],
        cols=walls.shape[1],
        goal_cells=len(goals),
        start_row=start[0],
        start_col=start[1],
        open_sides=4 * walls.size - sum(int(np.count_nonzero(walls & b)) for b in (N, E, S, W)),
        dead_ends=int(np.sum([bin(w).count("1") == 3 for w in walls.ravel()])),
        unreachable_cells=int(np.sum(dist < 0)),
        shortest_path=dist[start] if dist[start] >= 0 else np.nan,
    ))

df = pd.DataFrame(rows)
df.to_csv(os.path.join("data", "maze_summary.csv"), index=False)

print("number of mazes:", len(df))
print("number of columns:", df.shape[1])
print("\ncolumn types:\n", df.dtypes.to_string())
print("\nmissing values:\n", df.isna().sum().to_string())
print("\nmazes with unreachable goal:", df.maze[df.shortest_path.isna()].tolist())
print("\nsummary:\n", df.describe().round(2).to_string())

walls, start, goals = load_maze(os.path.join(DATA_DIR, "AAMC23Maze.txt"))
dist = flood_fill(walls, goals)
path = [start]
while dist[path[-1]] > 0:
    path.append(min(neighbours(walls, *path[-1]), key=lambda p: dist[p]))

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(dist, cmap="viridis_r")
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
pr, pc = zip(*path)
ax.plot(pc, pr, "r-", lw=2.5)
ax.set_title(f"AAMC 2023 maze: flood-fill distance, shortest path = {dist[start]} cells")
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join("figures", "maze_example.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(df.shortest_path.dropna(), bins=30, edgecolor="white")
ax.set_xlabel("shortest path from start to goal (cells)")
ax.set_ylabel("number of mazes")
ax.set_title("Shortest path length across past competition mazes")
fig.tight_layout()
fig.savefig(os.path.join("figures", "path_length_hist.png"), dpi=150)
