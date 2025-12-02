# src/agents/random_agent.py

import random
from src.game_2048 import ACTIONS

class RandomAgent:
    def select_action(self, env):
        actions = env.get_available_actions()
        if not actions:
            return None
        return random.choice(actions)
