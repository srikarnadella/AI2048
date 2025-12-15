import math
import time
import zlib
from array import array
from src.game_2048 import ACTIONS


class MinimaxAgent:
    """
    Deterministic minimax agent for 2048.

    This agent assumes the environment is adversarial:
    - MAX nodes: player choosing moves
    - MIN nodes: worst-case tile placement

    This is NOT a correct model for 2048, but it is useful
    as a baseline for comparison with Expectimax.
    """

    def __init__(self, depth=4, time_limit_seconds=0.12, debug=False):
        self.depth = depth
        self.time_limit_seconds = time_limit_seconds
        self.debug = debug
        self._deadline = None
        self._cache = {}

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

            score = reward + self.minimax(board_copy, self.depth - 1, is_min=True)
            if score > best_score:
                best_score = score
                best_action = action

            if self._deadline and time.time() >= self._deadline:
                break

        return best_action

    # ---------------------------------------------------------
    # MINIMAX CORE
    # ---------------------------------------------------------
    def minimax(self, board, depth, is_min):
        if self._deadline and time.time() >= self._deadline:
            return self.evaluate(board)

        key = (is_min, depth, self.board_key(board))
        if key in self._cache:
            return self._cache[key]

        if depth == 0 or self.is_terminal(board):
            val = self.evaluate(board)
            self._cache[key] = val
            return val

        if is_min:
            val = self.min_node(board, depth)
        else:
            val = self.max_node(board, depth)

        self._cache[key] = val
        return val

    def max_node(self, board, depth):
        best = -float("inf")

        for action in ACTIONS:
            if not self.can_move(board, action):
                continue

            board_copy = self.copy_board(board)
            moved, reward = self.move_board(board_copy, action)
            if not moved:
                continue

            score = reward + self.minimax(board_copy, depth - 1, is_min=True)
            best = max(best, score)

            if self._deadline and time.time() >= self._deadline:
                break

        return best if best != -float("inf") else self.evaluate(board)

    def min_node(self, board, depth):
        """
        Adversarial environment:
        chooses the WORST possible tile placement.
        """
        size = len(board)
        empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

        if not empty:
            return self.evaluate(board)

        worst = float("inf")

        for (r, c) in empty:
            for tile in (2, 4):
                board[r][c] = tile
                val = self.minimax(board, depth - 1, is_min=False)
                worst = min(worst, val)
                board[r][c] = 0

                if self._deadline and time.time() >= self._deadline:
                    return self.evaluate(board)

        return worst

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------
    def board_key(self, board):
        flat = array("H", (cell for row in board for cell in row))
        return zlib.crc32(flat.tobytes())

    def copy_board(self, board):
        return [row[:] for row in board]

    def is_terminal(self, board):
        size = len(board)
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    return False
        for r in range(size):
            for c in range(size - 1):
                if board[r][c] == board[r][c + 1]:
                    return False
        for c in range(size):
            for r in range(size - 1):
                if board[r][c] == board[r + 1][c]:
                    return False
        return True

    def can_move(self, board, action):
        size = len(board)
        if action == "left":
            for r in range(size):
                for c in range(1, size):
                    if board[r][c] and (board[r][c - 1] == 0 or board[r][c - 1] == board[r][c]):
                        return True
        elif action == "right":
            for r in range(size):
                for c in range(size - 2, -1, -1):
                    if board[r][c] and (board[r][c + 1] == 0 or board[r][c + 1] == board[r][c]):
                        return True
        elif action == "up":
            for c in range(size):
                for r in range(1, size):
                    if board[r][c] and (board[r - 1][c] == 0 or board[r - 1][c] == board[r][c]):
                        return True
        elif action == "down":
            for c in range(size):
                for r in range(size - 2, -1, -1):
                    if board[r][c] and (board[r + 1][c] == 0 or board[r + 1][c] == board[r][c]):
                        return True
        return False

    def move_board(self, board, action):
        size = len(board)
        moved = False
        total_reward = 0

        def compress_and_merge(line):
            nonlocal total_reward
            compressed = [v for v in line if v != 0]
            merged = []
            skip = False

            for i in range(len(compressed)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(compressed) and compressed[i] == compressed[i + 1]:
                    val = compressed[i] * 2
                    merged.append(val)
                    total_reward += val
                    skip = True
                else:
                    merged.append(compressed[i])

            while len(merged) < size:
                merged.append(0)
            return merged

        if action == "left":
            for r in range(size):
                new = compress_and_merge(board[r])
                if new != board[r]:
                    moved = True
                board[r] = new

        elif action == "right":
            for r in range(size):
                rev = list(reversed(board[r]))
                new = list(reversed(compress_and_merge(rev)))
                if new != board[r]:
                    moved = True
                board[r] = new

        elif action == "up":
            for c in range(size):
                col = [board[r][c] for r in range(size)]
                new = compress_and_merge(col)
                for r in range(size):
                    if board[r][c] != new[r]:
                        moved = True
                    board[r][c] = new[r]

        elif action == "down":
            for c in range(size):
                col = [board[r][c] for r in range(size - 1, -1, -1)]
                new = list(reversed(compress_and_merge(col)))
                for r in range(size):
                    if board[r][c] != new[r]:
                        moved = True
                    board[r][c] = new[r]

        return moved, total_reward

    # ---------------------------------------------------------
    # HEURISTIC (same as Expectimax for fair comparison)
    # ---------------------------------------------------------
    def evaluate(self, board):
        log_board = [[0 if v == 0 else int(math.log2(v)) for v in row] for row in board]

        empty = sum(cell == 0 for row in board for cell in row)
        smooth = 0
        mono = 0

        size = len(board)
        for r in range(size):
            for c in range(size - 1):
                if log_board[r][c] and log_board[r][c + 1]:
                    mono -= abs(log_board[r][c] - log_board[r][c + 1])
        for c in range(size):
            for r in range(size - 1):
                if log_board[r][c] and log_board[r + 1][c]:
                    mono -= abs(log_board[r][c] - log_board[r + 1][c])

        max_tile = max(max(row) for row in board)
        corner = 0
        if max_tile in (board[0][0], board[0][-1], board[-1][0], board[-1][-1]):
            corner = int(math.log2(max_tile))

        return 3.0 * empty + 1.5 * mono + 1.2 * corner - 0.25 * smooth
