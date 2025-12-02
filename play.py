# play.py

import argparse
import sys
import time
import statistics
from collections import Counter

from src.game_2048 import Game2048
from src.gui_2048 import Game2048GUI

from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.expectimax_agent import ExpectimaxAgent


# ----------------------------------------------------
# Load agent by name
# ----------------------------------------------------
def load_agent(name, depth=4, time_limit=0.08, debug=False):
    if name == "random":
        return RandomAgent()
    elif name == "greedy":
        return GreedyAgent()
    elif name == "expectimax":
        return ExpectimaxAgent(depth=depth, time_limit_seconds=time_limit, debug=debug)
    else:
        raise ValueError(f"Unknown agent: {name}")


# ----------------------------------------------------
# Human mode
# ----------------------------------------------------
def play_human(gui=False):
    if gui:
        Game2048GUI(agent=None).run()
        return

    env = Game2048()
    env.reset()
    env.render()

    key_map = {"w": "up", "s": "down", "a": "left", "d": "right"}

    while not env.done:
        move = input("Move (W/A/S/D): ").lower()
        if move in key_map:
            env.step(key_map[move])
            env.render()

    print("Game over! Score:", env.score)


# ----------------------------------------------------
# Agent single game mode
# ----------------------------------------------------
def play_agent(agent_name, gui=False, depth=4, time_limit=0.08, seed=None, debug=False):
    agent = load_agent(agent_name, depth=depth, time_limit=time_limit, debug=debug)

    if gui:
        Game2048GUI(agent=agent).run()
        return

    env = Game2048(seed=seed)
    env.reset()

    while not env.done:
        action = agent.select_action(env)
        env.step(action)

    print(f"[{agent_name}] Score:", env.score)


# ----------------------------------------------------
# Agent batch evaluation mode (10 games)
# ----------------------------------------------------
def eval_agent(agent_name, games=10, depth=4, time_limit=0.08, seed=None, debug=False):
    agent = load_agent(agent_name, depth=depth, time_limit=time_limit, debug=debug)

    scores = []
    max_tiles = []
    move_times = []
    wins = 0

    for i in range(games):
        game_seed = None if seed is None else seed + i
        env = Game2048(seed=game_seed)
        env.reset()

        while not env.done:
            t0 = time.perf_counter()
            action = agent.select_action(env)
            dt = time.perf_counter() - t0
            move_times.append(dt)
            env.step(action)

        # record results
        scores.append(env.score)
        max_tile = max(max(row) for row in env.board)
        max_tiles.append(max_tile)
        if max_tile >= 2048:
            wins += 1

        print(f"Game {i+1}: Score = {env.score}, Max tile = {max_tiles[-1]}")

    max_tile_counts = Counter(max_tiles)
    avg_score = sum(scores) / games
    avg_max_tile = sum(max_tiles) / games
    median_score = statistics.median(scores)
    stdev_score = statistics.pstdev(scores)
    avg_move_time_ms = (sum(move_times) / len(move_times) * 1000) if move_times else 0.0

    # Summary
    print("\n====================================")
    print(f"Agent: {agent_name}")
    print(f"Games played: {games}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Median score: {median_score:.2f} (stdev {stdev_score:.2f})")
    print(f"Average max tile: {avg_max_tile:.2f}")
    print(f"Win rate (>=2048): {wins/games*100:.1f}%")
    print(f"Best tile reached: {max(max_tiles)}")
    print(f"Average move time: {avg_move_time_ms:.1f} ms")
    print("Max tile distribution:")
    for tile, count in sorted(max_tile_counts.items()):
        print(f"  {tile}: {count} game(s)")
    print("====================================\n")


# ----------------------------------------------------
# CLI ENTRYPOINT
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        help="'human', 'random', 'greedy', 'expectimax', or 'eval-[agent]'"
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("--games", type=int, default=10, help="Number of games for eval-* modes")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for expectimax")
    parser.add_argument("--time-limit", type=float, default=0.08,
                        help="Per-move time budget (seconds) for expectimax")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for deterministic evals (increments per game)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug prints (expectimax decisions and per-move time)")

    args = parser.parse_args()

    # Batch evaluation:
    if args.mode.startswith("eval-"):
        agent_name = args.mode.split("eval-")[1]
        eval_agent(agent_name, games=args.games, depth=args.depth,
                   time_limit=args.time_limit, seed=args.seed, debug=args.debug)
        sys.exit()

    # Human mode
    if args.mode == "human":
        play_human(gui=args.gui)
        sys.exit()

    # Single agent game
    play_agent(args.mode, gui=args.gui, depth=args.depth,
               time_limit=args.time_limit, seed=args.seed, debug=args.debug)
