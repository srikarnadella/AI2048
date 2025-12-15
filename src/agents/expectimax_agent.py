import math
import time
import zlib
from array import array
from src.game_2048 import ACTIONS

# Fallback order (used if dynamic ordering ties or fails)
FALLBACK_ACTIONS = ["left", "up", "right", "down"]


class ExpectimaxAgent:
    """
    Expectimax agent with:
      - iterative deepening up to self.depth
      - strict per-move time budget
      - depth-aware transposition caching (Tier 1.2)
      - strong heuristic evaluation
      - dynamic move ordering
      - NEW: corner stability penalty
      - NEW: late-game adaptive depth
    """

    def __init__(self, depth=4, time_limit_seconds: float = 0.12, debug: bool = False):
        self.depth = depth
        self.time_limit_seconds = time_limit_seconds
        self._deadline = None
        self._cache = {}
        self.debug = debug

        # Fixed target corner (top-left)
        self.target_corner = (0, 0)

    # ---------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------
    def select_action(self, env):
        self._deadline = time.time() + self.time_limit_seconds if self.time_limit_seconds else None
        self._cache = {}

        best_score = -float("inf")
        best_action = None

        # -------- NEW: adaptive depth for late game --------
        empty = self.empty_cells(env.board)
        max_depth = self.depth + 1 if empty <= 4 else self.depth

        # Iterative deepening
        for depth_limit in range(1, max_depth + 1):
            depth_best_score = best_score
            depth_best_action = best_action

            ordered_actions = self.order_actions(env)

            for action in ordered_actions:
                moved, reward, board_copy = env.simulate_action(action)
                if not moved:
                    continue

                score = reward + self.expectimax(board_copy, depth_limit - 1, is_chance=True)

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_action = action

                if self._deadline and time.time() >= self._deadline:
                    break

            if depth_best_score > best_score:
                best_score = depth_best_score
                best_action = depth_best_action

            if self._deadline and time.time() >= self._deadline:
                break

        return best_action

    # ---------------------------------------------------------
    # Dynamic move ordering
    # ---------------------------------------------------------
    def order_actions(self, env):
        scored = []
        for action in ACTIONS:
            moved, reward, board_copy = env.simulate_action(action)
            if not moved:
                continue
            scored.append((reward + self.evaluate(board_copy), action))

        if not scored:
            return FALLBACK_ACTIONS

        scored.sort(reverse=True, key=lambda x: x[0])
        return [a for _, a in scored]

    # ---------------------------------------------------------
    # EXPECTIMAX CORE
    # ---------------------------------------------------------
    def expectimax(self, board, depth, is_chance):
        if self._deadline and time.time() >= self._deadline:
            return self.evaluate(board)

        key = (is_chance, self.board_key(board))
        if key in self._cache:
            cached_depth, cached_value = self._cache[key]
            if cached_depth >= depth:
                return cached_value

        if depth == 0 or self.is_terminal(board):
            val = self.evaluate(board)
            self._cache[key] = (depth, val)
            return val

        if is_chance:
            val = self.chance_node(board, depth)
        else:
            val = self.max_node(board, depth)

        self._cache[key] = (depth, val)
        return val

    def max_node(self, board, depth):
        max_score = -float("inf")

        for action in FALLBACK_ACTIONS:
            if not self.can_move(board, action):
                continue

            board_copy = self.copy_board(board)
            moved, reward = self.move_board(board_copy, action)
            if not moved:
                continue

            score = reward + self.expectimax(board_copy, depth - 1, is_chance=True)
            max_score = max(max_score, score)

            if self._deadline and time.time() >= self._deadline:
                break

        return max_score if max_score != -float("inf") else self.evaluate(board)

    def chance_node(self, board, depth):
        size = len(board)
        empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

        if not empty:
            return self.evaluate(board)

        total = 0.0

        for (r, c) in empty:
            board[r][c] = 2
            total += 0.9 * self.expectimax(board, depth - 1, is_chance=False)

            board[r][c] = 4
            total += 0.1 * self.expectimax(board, depth - 1, is_chance=False)

            board[r][c] = 0

            if self._deadline and time.time() >= self._deadline:
                return self.evaluate(board)

        return total / len(empty)

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------
    def board_key(self, board):
        flat = array("H", (cell for row in board for cell in row))
        return zlib.crc32(flat.tobytes())

    def copy_board(self, board):
        return [row[:] for row in board]

    def move_board(self, board, action):
        size = len(board)
        moved = False
        total_reward = 0

        if action == "left":
            for r in range(size):
                new_line, reward = self.compress_and_merge_line(board[r])
                if new_line != board[r]:
                    moved = True
                board[r] = new_line
                total_reward += reward

        elif action == "right":
            for r in range(size):
                rev = list(reversed(board[r]))
                new_line, reward = self.compress_and_merge_line(rev)
                new_line = list(reversed(new_line))
                if new_line != board[r]:
                    moved = True
                board[r] = new_line
                total_reward += reward

        elif action == "up":
            for c in range(size):
                col = [board[r][c] for r in range(size)]
                new_line, reward = self.compress_and_merge_line(col)
                for r in range(size):
                    if board[r][c] != new_line[r]:
                        moved = True
                    board[r][c] = new_line[r]
                total_reward += reward

        elif action == "down":
            for c in range(size):
                col = [board[r][c] for r in range(size - 1, -1, -1)]
                new_line, reward = self.compress_and_merge_line(col)
                new_line = list(reversed(new_line))
                for r in range(size):
                    if board[r][c] != new_line[r]:
                        moved = True
                    board[r][c] = new_line[r]
                total_reward += reward

        return moved, total_reward

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

    # ---------------------------------------------------------
    # EVALUATION — STABLE + CORNER STABILITY
    # ---------------------------------------------------------
    def evaluate(self, board):
        log_board = self.log_board(board)
        smooth_penalty = self.smoothness(log_board)
        monotonic_score = self.monotonicity(log_board)
        empty_bonus = self.empty_cells(board)
        corner_bonus = self.corner_max(board)
        gradient_bonus = self.weighted_grid(log_board)

        score = (
            3.0 * empty_bonus
            + 1.4 * corner_bonus
            + 0.65 * gradient_bonus
            + 1.5 * monotonic_score
            - 0.25 * smooth_penalty
        )

        # -------- NEW: corner stability penalty --------
        max_tile = max(max(row) for row in board)
        if max_tile > 0:
            r, c = self.target_corner
            if board[r][c] != max_tile:
                score -= 4.0 * math.log2(max_tile)

        return score

    def log_board(self, board):
        return [[0 if v == 0 else int(math.log2(v)) for v in row] for row in board]

    def empty_cells(self, board):
        return sum(cell == 0 for row in board for cell in row)

    def smoothness(self, log_board):
        size = len(log_board)
        score = 0.0
        for r in range(size):
            for c in range(size):
                if log_board[r][c] != 0:
                    value = log_board[r][c]
                    for dr, dc in ((1, 0), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and log_board[nr][nc] != 0:
                            score += abs(value - log_board[nr][nc])
        return score

    def monotonicity(self, log_board):
        size = len(log_board)
        totals = [0.0, 0.0, 0.0, 0.0]

        for r in range(size):
            for c in range(size - 1):
                if log_board[r][c] and log_board[r][c + 1]:
                    diff = log_board[r][c] - log_board[r][c + 1]
                    if diff > 0:
                        totals[0] -= diff
                    else:
                        totals[1] += diff

        for c in range(size):
            for r in range(size - 1):
                if log_board[r][c] and log_board[r + 1][c]:
                    diff = log_board[r][c] - log_board[r + 1][c]
                    if diff > 0:
                        totals[2] -= diff
                    else:
                        totals[3] += diff

        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def corner_max(self, board):
        max_tile = max(max(row) for row in board)
        if max_tile == 0:
            return 0
        max_log = int(math.log2(max_tile))
        size = len(board)
        corners = [
            board[0][0], board[0][size - 1],
            board[size - 1][0], board[size - 1][size - 1],
        ]
        return max_log * (2.0 if max_tile in corners else -1.0)

    def weighted_grid(self, log_board):
        size = len(log_board)
        weights = []
        w = size * size
        for r in range(size):
            row = []
            for _ in range(size):
                row.append(w)
                w -= 1
            weights.append(row)

        score = 0.0
        for r in range(size):
            for c in range(size):
                score += weights[r][c] * log_board[r][c]
        return score

    # ---------------------------------------------------------
    # LOW-LEVEL MOVE
    # ---------------------------------------------------------
    def compress_and_merge_line(self, line):
        size = len(line)
        compressed = [v for v in line if v != 0]

        merged = []
        reward = 0
        skip = False

        for i in range(len(compressed)):
            if skip:
                skip = False
                continue
            if i + 1 < len(compressed) and compressed[i] == compressed[i + 1]:
                new_val = compressed[i] * 2
                merged.append(new_val)
                reward += new_val
                skip = True
            else:
                merged.append(compressed[i])

        while len(merged) < size:
            merged.append(0)

        return merged, reward
