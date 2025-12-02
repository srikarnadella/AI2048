import argparse
from src.game_2048 import Game2048
from src.gui_2048 import Game2048GUI
from src.agents.random_agent import RandomAgent


def play_human(gui=False):
    if gui:
        Game2048GUI().run()
        return

    # fallback console mode
    env = Game2048()
    env.reset()
    env.render()

    while not env.done:
        move = input("Move (W/A/S/D): ").lower()
        key_map = {"w": "up", "s": "down", "a": "left", "d": "right"}
        if move not in key_map:
            continue
        env.step(key_map[move])
        env.render()


def play_agent(name, gui=False):
    if name == "random":
        agent = RandomAgent()
    else:
        raise ValueError("Unknown agent:", name)

    if gui:
        Game2048GUI(agent).run()
        return

    env = Game2048()
    env.reset()
    env.render()

    while not env.done:
        action = agent.select_action(env)
        env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="'human' or agent name ('random')")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
    args = parser.parse_args()

    if args.mode == "human":
        play_human(gui=args.gui)
    else:
        play_agent(args.mode, gui=args.gui)
