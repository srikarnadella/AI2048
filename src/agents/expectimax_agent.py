# src/agents/expectimax_agent.py

import math
import time
from src.game_2048 import ACTIONS


class ExpectimaxAgent:
    def __init__(self, depth=4, time_limit_seconds: float = 0.12, debug: bool = False):
        self.depth = depth
        # Per-move soft time budget; stops search early with heuristic value
        self.time_limit_seconds = time_limit_seconds
        self._deadline = None
        self._cache = {}
        self.debug = debug

    # ---------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------
    def select_action(self, env):
        self._deadline = time.time() + self.time_limit_seconds if self.time_limit_seconds else None
        self._cache = {}

        best_score = -float("inf")
        best_action = None

        for action in ACTIONS:
            moved, reward, board_copy = env.simulate_action(action)
            if not moved:
                continue
            score = reward + self.expectimax(board_copy, self.depth - 1, is_chance=True)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ---------------------------------------------------------
    # EXPECTIMAX CORE
    # ---------------------------------------------------------
    def expectimax(self, board, depth, is_chance):
        if self._deadline and time.time() >= self._deadline:
            return self.evaluate(board)

        key = (is_chance, depth, self.board_key(board))
        if key in self._cache:
            return self._cache[key]

        if depth == 0 or self.is_terminal(board):
            return self.evaluate(board)

        if is_chance:
            val = self.chance_node(board, depth)
        else:
            val = self.max_node(board, depth)

        self._cache[key] = val
        return val

    def max_node(self, board, depth):
        max_score = -float("inf")

        for action in ACTIONS:
            board_copy = self.copy_board(board)
            moved, reward = self.move_board(board_copy, action)

            if not moved:
                continue

            score = reward + self.expectimax(board_copy, depth - 1, is_chance=True)
            max_score = max(max_score, score)

        return max_score if max_score != -float("inf") else self.evaluate(board)

    def chance_node(self, board, depth):
        size = len(board)
        empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

        if not empty:
            return self.evaluate(board)

        total = 0
        for (r, c) in empty:
            # spawn 2
            b2 = self.copy_board(board)
            b2[r][c] = 2
            total += 0.9 * self.expectimax(b2, depth - 1, is_chance=False)

            # spawn 4
            b4 = self.copy_board(board)
            b4[r][c] = 4
            total += 0.1 * self.expectimax(b4, depth - 1, is_chance=False)

        return total / len(empty)

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------
    def board_key(self, board):
        return tuple(tuple(row) for row in board)

    def copy_board(self, board):
        return [row[:] for row in board]

    def move_board(self, board, action):
        size = len(board)
        moved = False
        total_reward = 0

        if action == "left":
            for r in range(size):
                original_line = board[r]
                new_line, reward = self.compress_and_merge_line(original_line)
                board[r] = new_line
                if new_line != original_line:
                    moved = True
                total_reward += reward

        elif action == "right":
            for r in range(size):
                original_line = list(reversed(board[r]))
                new_line, reward = self.compress_and_merge_line(original_line)
                new_line = list(reversed(new_line))
                if new_line != board[r]:
                    moved = True
                board[r] = new_line
                total_reward += reward

        elif action == "up":
            for c in range(size):
                original_line = [board[r][c] for r in range(size)]
                new_line, reward = self.compress_and_merge_line(original_line)
                for r in range(size):
                    if board[r][c] != new_line[r]:
                        moved = True
                    board[r][c] = new_line[r]
                total_reward += reward

        elif action == "down":
            for c in range(size):
                original_line = [board[r][c] for r in range(size - 1, -1, -1)]
                new_line, reward = self.compress_and_merge_line(original_line)
                new_line = list(reversed(new_line))
                for r in range(size):
                    if board[r][c] != new_line[r]:
                        moved = True
                    board[r][c] = new_line[r]
                total_reward += reward

        else:
            raise ValueError(f"Unknown action: {action}")

        return moved, total_reward

    def is_terminal(self, board):
        size = len(board)
        # If any empty cell exists, not terminal
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    return False

        # Horizontal or vertical merge available?
        for r in range(size):
            for c in range(size - 1):
                if board[r][c] == board[r][c + 1]:
                    return False
        for c in range(size):
            for r in range(size - 1):
                if board[r][c] == board[r + 1][c]:
                    return False

        return True

    # ---------------------------------------------------------
    # EVALUATION — STRONG HEURISTICS
    # ---------------------------------------------------------
    def evaluate(self, board):
        return (
            self.smoothness(board) * -0.1 +
            self.monotonicity(board) * 1.0 +
            self.empty_cells(board) * 2.7 +
            self.corner_max(board) * 1.0 +
            self.weighted_grid(board) * 1.0
        )

    def empty_cells(self, board):
        size = len(board)
        return sum(board[r][c] == 0 for r in range(size) for c in range(size))

    def smoothness(self, board):
        size = len(board)
        score = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] != 0:
                    value = math.log(board[r][c], 2)
                    for dr, dc in [(1,0),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] != 0:
                            score += abs(value - math.log(board[nr][nc], 2))
        return score

    def monotonicity(self, board):
        size = len(board)
        totals = [0, 0, 0, 0]

        # rows
        for r in range(size):
            for c in range(size - 1):
                if board[r][c] and board[r][c+1]:
                    x, y = math.log(board[r][c], 2), math.log(board[r][c+1], 2)
                    if x > y: totals[0] += x - y
                    else: totals[1] += y - x

        # columns
        for c in range(size):
            for r in range(size - 1):
                if board[r][c] and board[r+1][c]:
                    x, y = math.log(board[r][c], 2), math.log(board[r+1][c], 2)
                    if x > y: totals[2] += x - y
                    else: totals[3] += y - x

        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def corner_max(self, board):
        max_tile = max(max(row) for row in board)
        size = len(board)
        corners = [
            board[0][0],
            board[0][size - 1],
            board[size - 1][0],
            board[size - 1][size - 1],
        ]
        return max_tile * (2 if max_tile in corners else 0.5)

    def weighted_grid(self, board):
        size = len(board)
        # Default snake-like gradient for 4x4; fall back to a generic gradient otherwise.
        default_weights = [
            [65536, 32768, 16384, 8192],
            [512, 256, 128, 64],
            [16, 8, 4, 2],
            [1, 1, 1, 1],
        ]

        if size == 4:
            weights = default_weights
        else:
            weights = []
            base = 2 ** (size * size)
            for r in range(size):
                row = []
                for c in range(size):
                    idx = r * size + c if r % 2 == 0 else r * size + (size - 1 - c)
                    row.append(base // (2 ** idx))
                weights.append(row)

        score = 0
        for r in range(size):
            for c in range(size):
                score += weights[r][c] * board[r][c]
        return score

    # ---------------------------------------------------------
    # LOW-LEVEL MOVE (shared with env logic)
    # ---------------------------------------------------------
    def compress_and_merge_line(self, line):
        size = len(line)
        compressed = [v for v in line if v != 0]

        merged = []
        reward = 0
        skip_next = False

        for i in range(len(compressed)):
            if skip_next:
                skip_next = False
                continue

            if i + 1 < len(compressed) and compressed[i] == compressed[i + 1]:
                new_value = compressed[i] * 2
                merged.append(new_value)
                reward += new_value
                skip_next = True
            else:
                merged.append(compressed[i])

        merged = [v for v in merged if v != 0]

        while len(merged) < size:
            merged.append(0)

        return merged, reward
