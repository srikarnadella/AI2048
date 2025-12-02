# src/agents/greedy_agent.py

from src.game_2048 import ACTIONS, Game2048

class GreedyAgent:
    """
    Picks the move that gives the highest immediate merge reward.
    """

    def select_action(self, env):
        best_action = None
        best_reward = -1

        for action in ACTIONS:
            moved, reward, _ = env.simulate_action(action)

            if moved and reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action
