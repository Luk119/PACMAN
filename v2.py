import pygame
import random
import sys
import math
import json
import os

# Inicjalizacja
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Stałe
WIDTH, HEIGHT = 600, 750
CELL_SIZE = 30
ROWS, COLS = 19, 19
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pacman - 3 Poziomy!")

# Kolory
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
POWER_COLOR = (255, 215, 0)
FONT = pygame.font.SysFont("Arial", 20)
BIG_FONT = pygame.font.SysFont("Arial", 48)

# Dźwięki (opcjonalne)
try:
    CHOMP_SOUND = pygame.mixer.Sound("chomp.wav")
    POWER_SOUND = pygame.mixer.Sound("power.wav")
    EAT_GHOST_SOUND = pygame.mixer.Sound("eatghost.wav")
    START_SOUND = pygame.mixer.Sound("start.wav")
except:
    CHOMP_SOUND = POWER_SOUND = EAT_GHOST_SOUND = START_SOUND = None

# Labirynty dla 3 poziomów
MAZES = {
    1: [  # Poziom 1 - Klasyczny
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [3, 3, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 3],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 3, 3, 3, 1, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [3, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 3, 3],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    2: [  # Poziom 2 - Trudniejszy
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 3, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 3, 3, 3, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    3: [  # Poziom 3 - Najtrudniejszy
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
}

DIRS = [(1, 0), (-1, 0), (0, -1), (0, 1)]
HIGHSCORE_FILE = "pacman_scores.json"


# --- ZARZĄDZANIE WYNIKAMI ---
def load_highscores():
    if os.path.exists(HIGHSCORE_FILE):
        try:
            with open(HIGHSCORE_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_highscore(score, level):
    scores = load_highscores()
    scores.append({"score": score, "level": level})
    scores.sort(key=lambda x: x["score"], reverse=True)
    scores = scores[:10]  # Top 10
    with open(HIGHSCORE_FILE, 'w') as f:
        json.dump(scores, f)


def show_highscores():
    scores = load_highscores()
    SCREEN.fill(BLACK)
    title = BIG_FONT.render("TOP 10", True, YELLOW)
    SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

    for i, entry in enumerate(scores[:10]):
        text = FONT.render(f"{i + 1}. {entry['score']} pkt (Poziom {entry['level']})", True, WHITE)
        SCREEN.blit(text, (WIDTH // 2 - text.get_width() // 2, 150 + i * 35))

    back_text = FONT.render("Naciśnij SPACJĘ aby wrócić", True, CYAN)
    SCREEN.blit(back_text, (WIDTH // 2 - back_text.get_width() // 2, HEIGHT - 80))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False


# --- PACMAN ---
class Pacman:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 9 * CELL_SIZE
        self.y = 14 * CELL_SIZE
        self.size = CELL_SIZE // 2
        self.speed = CELL_SIZE
        self.direction = 0
        self.buffered_direction = -1
        self.mouth_open = True
        self.last_mouth_toggle = 0

    def try_change_direction(self, d):
        self.buffered_direction = d

    def move(self, maze):
        if self.buffered_direction != -1 and self.can_move(self.buffered_direction, maze):
            self.direction = self.buffered_direction
            self.buffered_direction = -1

        nx, ny = self.x, self.y
        dx, dy = DIRS[self.direction]
        nx += dx * self.speed
        ny += dy * self.speed

        if self.can_move_to(nx, ny, maze):
            self.x, self.y = nx, ny

        # Tunel
        if self.x <= -CELL_SIZE:
            self.x = WIDTH - CELL_SIZE
        if self.x >= WIDTH:
            self.x = 0

        if pygame.time.get_ticks() - self.last_mouth_toggle > 150:
            self.mouth_open = not self.mouth_open
            self.last_mouth_toggle = pygame.time.get_ticks()

    def can_move(self, d, maze):
        nx, ny = self.x, self.y
        dx, dy = DIRS[d]
        nx += dx * self.speed
        ny += dy * self.speed
        return self.can_move_to(nx, ny, maze)

    def can_move_to(self, x, y, maze):
        col = x // CELL_SIZE
        row = y // CELL_SIZE

        if col < 0:
            col = COLS - 1
        if col >= COLS:
            col = 0

        return 0 <= row < ROWS and maze[row][col] != 1

    def eat_dot(self, maze):
        col = self.x // CELL_SIZE
        row = self.y // CELL_SIZE

        if col < 0:
            col = COLS - 1
        if col >= COLS:
            col = 0

        if 0 <= row < ROWS and maze[row][col] in (0, 2):
            value = 10 if maze[row][col] == 0 else 50
            maze[row][col] = 3
            if value == 10 and CHOMP_SOUND:
                CHOMP_SOUND.play()
            elif value == 50 and POWER_SOUND:
                POWER_SOUND.play()
            return value
        return 0

    def draw(self, screen):
        center_x = self.x + CELL_SIZE // 2
        center_y = self.y + CELL_SIZE // 2
        mouth_angle = 45 if self.mouth_open else 10
        start_angle = [0, 180, 90, 270][self.direction]

        pygame.draw.circle(screen, YELLOW, (center_x, center_y), self.size)
        pygame.draw.arc(
            screen, BLACK,
            (self.x, self.y, CELL_SIZE, CELL_SIZE),
            math.radians(start_angle + mouth_angle),
            math.radians(start_angle + 360 - mouth_angle),
            width=3
        )


# --- DUCH (NAPRAWIONY) ---
class Ghost:
    def __init__(self, color=RED, start_x=9, start_y=8):
        self.start_x = start_x
        self.start_y = start_y
        self.color = color
        self.reset()

    def reset(self):
        self.x = self.start_x * CELL_SIZE
        self.y = self.start_y * CELL_SIZE
        self.size = CELL_SIZE // 2
        self.speed = CELL_SIZE
        self.direction = random.randint(0, 3)
        self.eaten = False
        self.stuck_counter = 0
        self.last_pos = (self.x, self.y)

    def move(self, pacman, power_mode, maze):
        # Wykrywanie zablokowania
        if (self.x, self.y) == self.last_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_pos = (self.x, self.y)

        # Jeśli duch jest zablokowany, losuj nowy kierunek
        if self.stuck_counter > 3:
            valid_dirs = [i for i in range(4) if self.can_move(i, maze)]
            if valid_dirs:
                self.direction = random.choice(valid_dirs)
            self.stuck_counter = 0

        if self.eaten:
            target = (9, 9)
        elif power_mode:
            # Ucieka od Pacmana
            px, py = pacman.x // CELL_SIZE, pacman.y // CELL_SIZE
            target = (18 - px, 18 - py)
        else:
            target = (pacman.x // CELL_SIZE, pacman.y // CELL_SIZE)

        # Znajdź najlepszy ruch
        valid_moves = []
        for i, (dx, dy) in enumerate(DIRS):
            nx = self.x + dx * self.speed
            ny = self.y + dy * self.speed
            if not self.can_move_to(nx, ny, maze):
                continue

            # Oblicz dystans do celu
            nx_col = nx // CELL_SIZE
            if nx_col < 0:
                nx_col = COLS - 1
            elif nx_col >= COLS:
                nx_col = 0

            dist = abs(nx_col - target[0]) + abs(ny // CELL_SIZE - target[1])

            # Preferuj nie zawracanie, ale pozwól jeśli to jedyna opcja
            opposite = (i + 2) % 4
            is_backtrack = (i == opposite)

            valid_moves.append((dist, is_backtrack, i))

        if valid_moves:
            # Sortuj: najpierw po dystansie, potem preferuj ruch bez zawracania
            valid_moves.sort(key=lambda x: (x[0], x[1]))
            self.direction = valid_moves[0][2]
        else:
            # Jeśli nie ma ruchu, zostań w miejscu
            return

        dx, dy = DIRS[self.direction]
        nx = self.x + dx * self.speed
        ny = self.y + dy * self.speed

        if self.can_move_to(nx, ny, maze):
            self.x, self.y = nx, ny

        # Tunel
        if self.x <= -CELL_SIZE:
            self.x = WIDTH - CELL_SIZE
        if self.x >= WIDTH:
            self.x = 0

    def can_move(self, d, maze):
        nx, ny = self.x, self.y
        dx, dy = DIRS[d]
        nx += dx * self.speed
        ny += dy * self.speed
        return self.can_move_to(nx, ny, maze)

    def can_move_to(self, x, y, maze):
        col = x // CELL_SIZE
        row = y // CELL_SIZE

        if col < 0:
            col = COLS - 1
        if col >= COLS:
            col = 0

        return 0 <= row < ROWS and maze[row][col] != 1

    def draw(self, screen, power_mode):
        if self.eaten:
            return
        color = WHITE if power_mode else self.color
        center = (self.x + CELL_SIZE // 2, self.y + CELL_SIZE // 2)
        pygame.draw.circle(screen, color, center, self.size)
        eye_offset = self.size // 3
        pygame.draw.circle(screen, BLACK, (center[0] - eye_offset, center[1] - eye_offset), 4)
        pygame.draw.circle(screen, BLACK, (center[0] + eye_offset, center[1] - eye_offset), 4)


# --- RYSOWANIE LABIRYNTU ---
def draw_maze(maze):
    for row in range(ROWS):
        for col in range(COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            if maze[row][col] == 1:
                pygame.draw.rect(SCREEN, BLUE, (x, y, CELL_SIZE, CELL_SIZE))
            elif maze[row][col] == 0:
                pygame.draw.circle(SCREEN, WHITE, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), 4)
            elif maze[row][col] == 2:
                pygame.draw.circle(SCREEN, POWER_COLOR, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), 8)


# --- EKRAN STARTOWY ---
def show_start_screen():
    SCREEN.fill(BLACK)
    title = BIG_FONT.render("PACMAN", True, YELLOW)
    start = FONT.render("Naciśnij SPACJĘ aby grać", True, WHITE)
    scores = FONT.render("Naciśnij H aby zobaczyć wyniki", True, CYAN)

    SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2, 200))
    SCREEN.blit(start, (WIDTH // 2 - start.get_width() // 2, 320))
    SCREEN.blit(scores, (WIDTH // 2 - scores.get_width() // 2, 360))
    pygame.display.flip()

    if START_SOUND:
        START_SOUND.play()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_h:
                    show_highscores()
                    show_start_screen()


# --- RESET GRY ---
def reset_game(pacman, ghosts, level):
    maze = [row[:] for row in MAZES[level]]
    pacman.reset()
    for ghost in ghosts:
        ghost.reset()
    return 0, False, 0, maze


# --- MAIN ---
def main():
    clock = pygame.time.Clock()
    pacman = Pacman()

    # Tworzenie wielu duchów dla większej trudności
    ghosts = [
        Ghost(RED, 9, 8),
        Ghost(PINK, 8, 9),
        Ghost(CYAN, 10, 9)
    ]

    show_start_screen()

    current_level = 1
    total_score = 0
    score, power_mode, power_timer, maze = reset_game(pacman, ghosts, current_level)
    game_over = False
    level_complete = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: pacman.try_change_direction(1)
                if event.key == pygame.K_RIGHT: pacman.try_change_direction(0)
                if event.key == pygame.K_UP: pacman.try_change_direction(2)
                if event.key == pygame.K_DOWN: pacman.try_change_direction(3)
                if event.key == pygame.K_r and game_over:
                    current_level = 1
                    total_score = 0
                    score, power_mode, power_timer, maze = reset_game(pacman, ghosts, current_level)
                    game_over = False
                    level_complete = False
                    show_start_screen()

        if not game_over and not level_complete:
            pacman.move(maze)
            points = pacman.eat_dot(maze)
            score += points
            total_score += points

            if points == 50:
                power_mode = True
                power_timer = 300

            if power_mode:
                power_timer -= 1
                if power_timer <= 0:
                    power_mode = False
                    for ghost in ghosts:
                        ghost.eaten = False

            for ghost in ghosts:
                if not ghost.eaten:
                    ghost.move(pacman, power_mode, maze)

            # Kolizje
            px = pacman.x + CELL_SIZE // 2
            py = pacman.y + CELL_SIZE // 2

            for ghost in ghosts:
                gx = ghost.x + CELL_SIZE // 2
                gy = ghost.y + CELL_SIZE // 2
                if abs(px - gx) < CELL_SIZE and abs(py - gy) < CELL_SIZE:
                    if power_mode and not ghost.eaten:
                        ghost.eaten = True
                        score += 200
                        total_score += 200
                        if EAT_GHOST_SOUND:
                            EAT_GHOST_SOUND.play()
                    elif not ghost.eaten:
                        game_over = True
                        save_highscore(total_score, current_level)

            # Sprawdź koniec poziomu
            if all(cell not in (0, 2) for row in maze for cell in row):
                if current_level < 3:
                    level_complete = True
                else:
                    game_over = True
                    save_highscore(total_score, current_level)

        # Ekran zakończenia poziomu
        if level_complete:
            SCREEN.fill(BLACK)
            draw_maze(maze)
            msg = BIG_FONT.render(f"POZIOM {current_level} UKOŃCZONY!", True, YELLOW)
            next_text = FONT.render("Naciśnij SPACJĘ dla następnego poziomu", True, WHITE)
            score_text = FONT.render(f"Wynik: {total_score}", True, CYAN)

            SCREEN.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 80))
            SCREEN.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 20))
            SCREEN.blit(next_text, (WIDTH // 2 - next_text.get_width() // 2, HEIGHT // 2 + 40))
            pygame.display.flip()

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        current_level += 1
                        score, power_mode, power_timer, maze = reset_game(pacman, ghosts, current_level)
                        level_complete = False
                        waiting = False

        # Rysowanie
        SCREEN.fill(BLACK)
        draw_maze(maze)
        pacman.draw(SCREEN)
        for ghost in ghosts:
            ghost.draw(SCREEN, power_mode)

        # HUD
        score_text = FONT.render(f"WYNIK: {total_score}", True, WHITE)
        level_text = FONT.render(f"POZIOM: {current_level}/3", True, CYAN)
        SCREEN.blit(score_text, (10, HEIGHT - 50))
        SCREEN.blit(level_text, (WIDTH - 150, HEIGHT - 50))

        if game_over:
            win = current_level == 3 and all(cell not in (0, 2) for row in maze for cell in row)
            msg = "GRATULACJE! UKOŃCZYŁEŚ GRĘ!" if win else "GAME OVER"
            msg_text = BIG_FONT.render(msg, True, YELLOW if win else RED)
            restart_text = FONT.render("R = restart", True, WHITE)
            final_score = FONT.render(f"Końcowy wynik: {total_score}", True, CYAN)

            SCREEN.blit(msg_text, (WIDTH // 2 - msg_text.get_width() // 2, HEIGHT // 2 - 80))
            SCREEN.blit(final_score, (WIDTH // 2 - final_score.get_width() // 2, HEIGHT // 2 - 20))
            SCREEN.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 40))

        pygame.display.flip()
        clock.tick(12)


if __name__ == "__main__":
    main()