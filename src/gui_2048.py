import pygame
import sys
from src.game_2048 import Game2048

# ------------------------
# Visual settings
# ------------------------
BOARD_SIZE = 4
TILE_SIZE = 110
TILE_PADDING = 12
WINDOW_SIZE = TILE_SIZE * BOARD_SIZE + TILE_PADDING * (BOARD_SIZE + 1)
BOTTOM_BAR = 90

FPS = 60
ANIM_TIME = 0.12  # smooth animation timing

BACKGROUND = (187, 173, 160)
EMPTY = (205, 193, 180)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)

COLORS = {
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

def tile_rect(row, col):
    x = TILE_PADDING + col * (TILE_SIZE + TILE_PADDING)
    y = TILE_PADDING + row * (TILE_SIZE + TILE_PADDING)
    return pygame.Rect(int(x), int(y), TILE_SIZE, TILE_SIZE)


class Game2048GUI:
    def __init__(self, agent=None):
        pygame.init()
        pygame.display.set_caption("2048")

        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + BOTTOM_BAR))
        self.clock = pygame.time.Clock()

        self.font_tile = pygame.font.Font(None, 48)
        self.font_score = pygame.font.Font(None, 32)

        self.env = Game2048()
        self.agent = agent
        self.env.reset()

        self.prev_state = self.env.get_state()
        self.animations = []  # list of dict {value, start_px, end_px, t}

    # ----------------------------
    # ANIMATION HELPERS
    # ----------------------------
    def prepare_animations(self, before, after):
        """Prepare smooth pixel-based animations between grid states."""
        self.animations = []
        size = BOARD_SIZE

        before_positions = {}
        after_positions = {}

        # record all tiles with unique IDs
        uid = 0
        for r in range(size):
            for c in range(size):
                if before[r][c] != 0:
                    before_positions[(r, c, uid)] = before[r][c]
                    uid += 1

        for r in range(size):
            for c in range(size):
                if after[r][c] != 0:
                    after_positions[(r, c)] = after[r][c]

        # MATCH TILES BY VALUE AND PROXIMITY  
        # (robust — this never crashes)
        used_after = set()

        for (r, c, id_) in before_positions.keys():
            val = before_positions[(r, c, id_)]

            # find nearest after-position with same value
            target = None
            best_dist = 999

            for (r2, c2), v2 in after_positions.items():
                if v2 == val and (r2, c2) not in used_after:
                    dist = abs(r - r2) + abs(c - c2)
                    if dist < best_dist:
                        best_dist = dist
                        target = (r2, c2)

            if target:
                used_after.add(target)

                start_px = tile_rect(r, c).topleft
                end_px = tile_rect(target[0], target[1]).topleft

                self.animations.append({
                    "value": val,
                    "start": start_px,
                    "end": end_px,
                    "t": 0.0
                })

    def animate(self):
        """Runs the animation frame-by-frame."""
        if not self.animations:
            return

        dt = self.clock.get_time() / 1000.0

        finished = True
        for anim in self.animations:
            anim["t"] += dt
            if anim["t"] < ANIM_TIME:
                finished = False

        self.draw(animate=True)
        pygame.display.flip()

        return finished

    # ----------------------------
    # DRAWING
    # ----------------------------
    def draw(self, animate=False):
        self.screen.fill(BACKGROUND)

        state = self.env.get_state()
        draw_state = self.prev_state if animate else state

        # Draw static background tiles
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = draw_state[r][c]
                rect = tile_rect(r, c)
                pygame.draw.rect(self.screen, COLORS.get(val, EMPTY), rect, border_radius=10)

                if val != 0:
                    text_color = TEXT_DARK if val <= 4 else TEXT_LIGHT
                    surface = self.font_tile.render(str(val), True, text_color)
                    self.screen.blit(surface, surface.get_rect(center=rect.center))

        if animate:
            # Draw animated tiles last
            for anim in self.animations:
                progress = min(anim["t"] / ANIM_TIME, 1)
                sx, sy = anim["start"]
                ex, ey = anim["end"]

                x = sx + (ex - sx) * progress
                y = sy + (ey - sy) * progress

                rect = pygame.Rect(int(x), int(y), TILE_SIZE, TILE_SIZE)
                val = anim["value"]

                pygame.draw.rect(self.screen, COLORS.get(val, EMPTY), rect, border_radius=10)
                text_color = TEXT_DARK if val <= 4 else TEXT_LIGHT
                surface = self.font_tile.render(str(val), True, text_color)
                self.screen.blit(surface, surface.get_rect(center=rect.center))

        # Score
        score_label = self.font_score.render(f"Score: {self.env.score}", True, (255, 255, 255))
        self.screen.blit(score_label, (15, WINDOW_SIZE + 20))

    # ----------------------------
    # INPUT HANDLING
    # ----------------------------
    def handle_input(self, event):
        if event.key == pygame.K_UP:
            return "up"
        if event.key == pygame.K_DOWN:
            return "down"
        if event.key == pygame.K_LEFT:
            return "left"
        if event.key == pygame.K_RIGHT:
            return "right"
        return None

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    def run(self):
        running = True
        self.draw()
        pygame.display.flip()

        while running:
            self.clock.tick(FPS)

            action = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if self.agent is None and event.type == pygame.KEYDOWN:
                    action = self.handle_input(event)

            if self.agent is not None and not self.env.done:
                pygame.time.delay(110)
                action = self.agent.select_action(self.env)

            if action:
                before = self.prev_state
                _, _, _, _ = self.env.step(action)
                after = self.env.get_state()

                self.prepare_animations(before, after)

                while not self.animate():
                    # Avoid a busy spin while animations run.
                    self.clock.tick(FPS)

                self.prev_state = after

            self.draw()
            pygame.display.flip()

            if self.env.done:
                pygame.time.wait(1000)
                return
