# Micromouse Maze as a Markov Decision Process

DS5020 Final Project (Fall 2026, Dr. Fatema Nafa)
Team: Giorgi Karazanashvili, Dhruv Dharmaraj Shetty

We model a Micromouse maze as a Markov decision process and train a reinforcement
learning agent, implemented from scratch in NumPy, to reach the goal of new, unseen
competition mazes as fast as possible.

## Repository layout

```
src/        maze parser, data loading, methods, simulation and real-data scripts
data/       maze summary (raw maze files are downloaded, not in Git)
figures/    plots produced by the scripts
```

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Getting the data

Download the community archive of past competition mazes and copy the `classic`
folder to `data/mazes_classic`:

```
Invoke-WebRequest https://github.com/micromouseonline/mazefiles/archive/refs/heads/master.zip -OutFile mazefiles.zip
Expand-Archive mazefiles.zip -DestinationPath mazefiles-tmp
Copy-Item -Recurse mazefiles-tmp\mazefiles-master\classic data\mazes_classic
Remove-Item -Recurse -Force mazefiles.zip, mazefiles-tmp
```

## Running

From the repository root:

```
python src/load_data.py
python src/simulation.py
python src/small_example.py
python src/check_simulation.py
python src/run_real.py
```

`load_data.py` describes the real mazes and writes `data/maze_summary.csv` and two figures.
`simulation.py` runs the hand-worked example and the simulation check of the Iteration 2 derivations.
`small_example.py` solves a 3 x 3 maze step by step: exact transition matrices, policy evaluation with our
Gaussian elimination, value iteration after every step, and Q-learning (Iteration 3, Q1-Q4).
`check_simulation.py` runs the Iteration 2 5 x 5 simulation against the true values and against NumPy,
the Q-learning error for 500, 5000 and 50000 episodes, the gamma = 0.999 stress test and the assumption checks.
`run_real.py` cleans the competition mazes, draws a parsed maze, compares our solvers with NumPy on every maze,
trains Q-learning on competition mazes and on the unseen maze `AAMC24Maze.txt` (Q5), and checks the assumptions
(about 3 minutes).
The methods are in `src/methods.py` (NumPy only); `src/simulate.py` builds the simulation model.
All randomness uses `np.random.default_rng(0)`.
