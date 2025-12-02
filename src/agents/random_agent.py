# src/agents/random_agent.py

import random

class RandomAgent:
    """
    The simplest agent. Chooses a random legal move.
    """
    def select_action(self, env):
        actions = env.get_available_actions()
        if not actions:
            return None
        return random.choice(actions)
