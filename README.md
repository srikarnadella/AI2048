# AI2048 Project

Small 2048 environment plus baseline and expectimax agents.

## How to run

Human, terminal:
```
python play.py human
```

Human, GUI:
```
python play.py human --gui
```

Single agent game (no GUI):
```
python play.py expectimax --depth 4 --time-limit 0.08
```

Evaluate an agent over N games (prints score stats, win rate, tile distribution, and average move time):
```
python play.py eval-expectimax --games 20 --depth 4 --time-limit 0.08 --seed 123
python play.py eval-greedy --games 20 --seed 123
python play.py eval-random --games 20 --seed 123
```

Flags:
- `--games`: number of evaluation games (eval-* modes only)
- `--depth`: search depth for expectimax (default 4)
- `--time-limit`: per-move time budget in seconds for expectimax (default 0.08)
- `--seed`: base seed for deterministic runs; increments per game in eval mode
- `--gui`: enable the pygame GUI (human or single-agent modes)
