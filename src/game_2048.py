# src/game_2048.py

import random
from typing import List, Tuple, Dict, Optional


# Type alias for the board state: 4x4 grid of ints
Board = List[List[int]]

# Legal actions for the agent/human
ACTIONS = ["up", "down", "left", "right"]


class Game2048:
    """
    Core 2048 environment.

    This class is designed so BOTH a human player and an AI agent can interact
    with it using a simple API:

        env = Game2048()
        state = env.reset()
        done = False
        while not done:
            action = some_policy(state)  # "up", "down", "left", "right"
            next_state, reward, done, info = env.step(action)
            state = next_state

    Later, you can plug in:
      - random agent
      - greedy agent
      - minimax / expectimax agent
    using the exact same interface.
    """

    def __init__(self, size: int = 4, seed: Optional[int] = None) -> None:
        """
        Initialize the environment.

        :param size: board size (default 4 for the standard 2048 game).
        :param seed: optional random seed for reproducibility.
        """
        self.size = size

        # Internal PRNG so you can control randomness for experiments
        self._rng = random.Random(seed)

        # Board state: created in reset()
        self.board: Board = [[0] * size for _ in range(size)]

        # Total score accumulated over the game
        self.score: int = 0

        # Flag indicating if the game has ended
        self.done: bool = False

    # -------------------------------------------------------------------------
    # Public API (Gym-like)
    # -------------------------------------------------------------------------

    def reset(self) -> Board:
        """
        Reset the game to the initial state.

        - Clears the board
        - Spawns two initial tiles
        - Resets score and done flag

        :return: the initial board state.
        """
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.done = False

        # Spawn two starting tiles
        self._spawn_tile()
        self._spawn_tile()

        # Return a deep copy of the board so callers can't mutate internal state
        return self._copy_board(self.board)

    def step(self, action: str) -> Tuple[Board, int, bool, Dict]:
        """
        Apply one action to the environment.

        :param action: one of "up", "down", "left", "right"
        :return: (next_state, reward, done, info)
                 - next_state: board AFTER the move and tile spawn
                 - reward: score gained from this move (sum of merged tiles)
                 - done: True if no more moves are available
                 - info: extra debug info (currently empty, but you can add stuff)
        """
        if self.done:
            # If the game is already over, no-op but keep contract consistent.
            return self._copy_board(self.board), 0, True, {}

        if action not in ACTIONS:
            # In a larger project you might raise an exception.
            # For now, we just ignore invalid actions.
            raise ValueError(f"Invalid action '{action}'. Must be one of {ACTIONS}.")

        # Perform the move in the requested direction
        moved, reward = self._move(action)

        # If the move actually changed the board, spawn a new tile
        if moved:
            self._spawn_tile()

        # Update the total score
        self.score += reward

        # Check if the game is now over (no valid moves remain)
        self.done = self._is_game_over()

        # Return a deep copy of the board (to protect internal state)
        return self._copy_board(self.board), reward, self.done, {}

    def get_available_actions(self) -> List[str]:
        """
        Return a list of all actions that would actually change the board.

        This is useful for:
        - Agents that want to avoid exploring illegal / no-op moves.
        - Detecting terminal states (no available_actions).

        :return: list of actions from ACTIONS that cause a change.
        """
        available = []
        for action in ACTIONS:
            moved, _, _ = self.simulate_action(action)
            if moved:
                available.append(action)
        return available

    def simulate_action(self, action: str) -> Tuple[bool, int, Board]:
        """
        Simulate an action on a copy of the current board without mutating the
        environment.

        :param action: one of ACTIONS
        :return: (moved, reward, next_board_copy)
        """
        board_copy = self._copy_board(self.board)
        moved, reward = self._move_board(board_copy, action)
        return moved, reward, board_copy

    def get_state(self) -> Board:
        """
        Get a copy of the current board state.

        :return: a deep copy of the board.
        """
        return self._copy_board(self.board)

    def render(self) -> None:
        """
        Pretty-print the current board and score to the console.

        This is handy for debugging or for a simple human-playable interface.
        """
        print("-" * (self.size * 6))
        print(f"Score: {self.score}")
        print("-" * (self.size * 6))
        for row in self.board:
            # Format each cell right-aligned in 5 spaces; use '.' for empty (0)
            row_str = " ".join(f"{val:4d}" if val != 0 else "   ."
                               for val in row)
            print(row_str)
        print("-" * (self.size * 6))

    # -------------------------------------------------------------------------
    # Internal board utilities
    # -------------------------------------------------------------------------

    def _copy_board(self, board: Board) -> Board:
        """
        Create a deep copy of the board (list of lists).

        We manually copy instead of using copy.deepcopy for clarity and speed.
        """
        return [row[:] for row in board]

    def _spawn_tile(self) -> None:
        """
        Spawn a new tile (2 or 4) at a random empty position on the board.

        - 2 appears with probability 0.9
        - 4 appears with probability 0.1
        """
        empty_positions = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board[r][c] == 0
        ]

        # If no empty positions, nothing to spawn (game might be over)
        if not empty_positions:
            return

        # Choose a random empty cell
        r, c = self._rng.choice(empty_positions)

        # Choose tile value (2 with prob 0.9, else 4)
        value = 2 if self._rng.random() < 0.9 else 4
        self.board[r][c] = value

    def _is_game_over(self) -> bool:
        """
        Check if the game is over (no valid moves remain).

        A game is over if:
        - no empty cells AND
        - no horizontal or vertical adjacent tiles that can be merged.
        """
        # If any empty cell exists, the game is not over
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    return False

        # Check horizontal merges
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r][c] == self.board[r][c + 1]:
                    return False

        # Check vertical merges
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r][c] == self.board[r + 1][c]:
                    return False

        # No empty cells and no possible merges: game over
        return True

    # -------------------------------------------------------------------------
    # Move logic (slide + merge) for each direction
    # -------------------------------------------------------------------------

    def _move(self, action: str) -> Tuple[bool, int]:
        """
        Public-facing move function that uses the agent's action to mutate
        the *real* board.

        :param action: "up", "down", "left", or "right"
        :return: (moved, reward) where:
                 - moved is True if the board changed
                 - reward is the sum of merged tiles from this move
        """
        moved, reward = self._move_board(self.board, action)
        return moved, reward

    def _move_board(self, board: Board, action: str) -> Tuple[bool, int]:
        """
        Core move logic that operates on ANY board (real or copy).

        This is useful because:
        - For get_available_actions(), we want to simulate moves on a copy.
        - For search/planning, you may want to simulate moves on custom boards.

        :param board: the board to modify IN PLACE.
        :param action: direction of movement.
        :return: (moved, reward)
        """
        moved = False
        total_reward = 0

        # For each direction, we transform the board into a set of "lines"
        # (rows or columns), apply the "slide and merge" logic to each line,
        # then write the result back to the board.

        if action == "left":
            for r in range(self.size):
                original_line = board[r]
                new_line, reward = self._compress_and_merge_line(original_line)
                board[r] = new_line
                if new_line != original_line:
                    moved = True
                total_reward += reward

        elif action == "right":
            for r in range(self.size):
                original_line = list(reversed(board[r]))
                new_line, reward = self._compress_and_merge_line(original_line)
                new_line = list(reversed(new_line))
                if new_line != board[r]:
                    moved = True
                board[r] = new_line
                total_reward += reward

        elif action == "up":
            for c in range(self.size):
                original_line = [board[r][c] for r in range(self.size)]
                new_line, reward = self._compress_and_merge_line(original_line)
                for r in range(self.size):
                    if board[r][c] != new_line[r]:
                        moved = True
                    board[r][c] = new_line[r]
                total_reward += reward

        elif action == "down":
            for c in range(self.size):
                original_line = [board[r][c] for r in range(self.size - 1, -1, -1)]
                new_line, reward = self._compress_and_merge_line(original_line)
                new_line = list(reversed(new_line))
                for r in range(self.size):
                    if board[r][c] != new_line[r]:
                        moved = True
                    board[r][c] = new_line[r]
                total_reward += reward

        else:
            raise ValueError(f"Unknown action: {action}")

        return moved, total_reward

    def _compress_and_merge_line(self, line: List[int]) -> Tuple[List[int], int]:
        """
        Given a single row/column (as a list of ints), perform the standard
        2048 move operation *in one direction*:

        Steps:
          1. "Compress" the line by sliding all non-zero tiles toward the start.
          2. Merge adjacent equal tiles from the start (left) side:
             - When two equal tiles merge, they become a tile with double value.
             - The merged tile contributes its value to the reward.
             - The next tile is skipped to avoid merging it twice.
          3. Compress again to fill gaps created by merges.
          4. Pad with zeros at the end so length stays constant.

        Example:
          Input line: [2, 0, 2, 4]
          After compression: [2, 2, 4, 0]
          After merge: [4, 4, 0, 0]   (reward += 4)
          After final compression: [4, 4, 0, 0] (already compressed)
        """
        size = len(line)

        # Step 1: Compress (remove zeros while preserving order of non-zero tiles)
        compressed = [v for v in line if v != 0]

        # Step 2: Merge adjacent equal tiles
        merged: List[int] = []
        reward = 0
        skip_next = False

        for i in range(len(compressed)):
            if skip_next:
                # We've already merged compressed[i - 1] and compressed[i]
                skip_next = False
                continue

            if i + 1 < len(compressed) and compressed[i] == compressed[i + 1]:
                # Merge compressed[i] and compressed[i + 1]
                new_value = compressed[i] * 2
                merged.append(new_value)
                reward += new_value
                skip_next = True  # Skip the next tile since it's now merged
            else:
                # No merge; just carry the value over
                merged.append(compressed[i])

        # Step 3: Compress again (usually already compressed, but safe to keep)
        merged = [v for v in merged if v != 0]

        # Step 4: Pad with zeros to maintain the original length
        while len(merged) < size:
            merged.append(0)

        return merged, reward
