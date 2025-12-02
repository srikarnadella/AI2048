# src/gui_2048.py

import pygame
import sys
import time
from src.game_2048 import Game2048

# ---------------------------------------
# GUI Constants
# ---------------------------------------
BOARD_SIZE = 4
TILE_SIZE = 110
TILE_PADDING = 12

WINDOW_SIZE = TILE_SIZE * BOARD_SIZE + TILE_PADDING * (BOARD_SIZE + 1)
BOTTOM_BAR = 90

FPS = 60
ANIMATION_SPEED = 0.18  # lower = slower animations

# Colors
BACKGROUND = (187, 173, 160)
EMPTY_TILE = (205, 193, 180)

TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 200, 80),
    512: (237, 197, 63),
    1024: (237, 194, 46),
    2048: (255, 186, 0),
}

TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)


# ---------------------------------------
# Helper Functions
# ---------------------------------------
def tile_rect(r, c):
    x = TILE_PADDING + c * (TILE_SIZE + TILE_PADDING)
    y = TILE_PADDING + r * (TILE_SIZE + TILE_PADDING)
    return pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)


# ---------------------------------------
# GUI Class
# ---------------------------------------
class Game2048GUI:
    def __init__(self, agent=None):
        pygame.init()
        pygame.display.set_caption("2048")

        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + BOTTOM_BAR))
        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        self.env = Game2048()
        self.agent = agent
        self.env.reset()

        # Track animations
        self.previous_board = self.env.get_state()
        self.animations = []

    # ------------------------------------------------
    # Animation System
    # ------------------------------------------------
    def start_animation(self, prev, new):
        """Compute slide animations for changed tiles."""
        self.animations = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if prev[r][c] != 0 and prev[r][c] != new[r][c]:
                    # Find where it moved
                    for nr in range(BOARD_SIZE):
                        for nc in range(BOARD_SIZE):
                            if new[nr][nc] == prev[r][c]:
                                self.animations.append({
                                    "value": prev[r][c],
                                    "start": (r, c),
                                    "end": (nr, nc),
                                    "time": 0.0
                                })

    def animate(self):
        """Smooth slide animation between states."""
        if not self.animations:
            return False

        dt = self.clock.get_time() / 1000.0
        finished = True

        for anim in self.animations:
            anim['time'] += dt
            if anim['time'] < ANIMATION_SPEED:
                finished = False

        self.draw_board(animate=True)
        pygame.display.flip()

        return finished

    # ------------------------------------------------
    # Drawing
    # ------------------------------------------------
    def draw_board(self, animate=False):
        self.screen.fill(BACKGROUND)

        board = self.env.get_state()

        # If animating, draw previous + overlay animation tiles
        if animate:
            base_board = self.previous_board
        else:
            base_board = board

        # Draw tiles
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = base_board[r][c]
                rect = tile_rect(r, c)

                # Tile background
                pygame.draw.rect(self.screen,
                                 TILE_COLORS.get(val, EMPTY_TILE),
                                 rect, border_radius=10)

                # Value text
                if val != 0:
                    color = TEXT_DARK if val <= 4 else TEXT_LIGHT
                    text_surface = self.font_big.render(str(val), True, color)
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)

        # Overlay moving tiles
        if animate:
            for anim in self.animations:
                progress = min(1, anim['time'] / ANIMATION_SPEED)
                sr, sc = anim['start']
                er, ec = anim['end']

                # Interpolate position
                x = (1 - progress) * sc + progress * ec
                y = (1 - progress) * sr + progress * er

                rect = tile_rect(y, x)
                val = anim['value']

                pygame.draw.rect(self.screen,
                                 TILE_COLORS.get(val, EMPTY_TILE),
                                 rect, border_radius=10)

                text_surface = self.font_big.render(str(val), True, TEXT_DARK)
                text_rect = text_surface.get_rect(center=rect.center)
                self.screen.blit(text_surface, text_rect)

        # Draw score
        score_text = self.font_small.render(f"Score: {self.env.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (15, WINDOW_SIZE + 20))

    # ------------------------------------------------
    # Input
    # ------------------------------------------------
    def handle_human_input(self, event):
        if event.key == pygame.K_UP:
            return "up"
        if event.key == pygame.K_DOWN:
            return "down"
        if event.key == pygame.K_LEFT:
            return "left"
        if event.key == pygame.K_RIGHT:
            return "right"
        return None

    # ------------------------------------------------
    # Main Loop
    # ------------------------------------------------
    def run(self):
        running = True

        while running:
            self.clock.tick(FPS)

            # AGENT OR HUMAN
            action = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if self.agent is None and event.type == pygame.KEYDOWN:
                    action = self.handle_human_input(event)

            if self.agent is not None:
                pygame.time.delay(100)
                action = self.agent.select_action(self.env)

            if action:
                old = self.previous_board
                _, _, _, _ = self.env.step(action)
                new = self.env.get_state()

                self.start_animation(old, new)

                # Run animation loop
                while not self.animate():
                    pass

                self.previous_board = new

            self.draw_board()
            pygame.display.flip()

            if self.env.done:
                pygame.time.wait(1200)
                running = False
