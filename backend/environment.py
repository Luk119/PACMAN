"""
=============================================================================
ŚRODOWISKO GRY PAC-MAN DLA DEEP Q-LEARNING
=============================================================================

Moduł zawiera klasę PacmanEnvironment – czyste środowisko gry Pac-Man
niezależne od jakiejkolwiek biblioteki wizualnej (bez pygame).
Pełni rolę analogiczną do środowisk OpenAI Gym i jest używane zarówno
przez agenta DQN podczas treningu, jak i przez serwer Flask w trybie
interaktywnej gry.

LEGENDA LABIRYNTU:
    0 – wolne pole z kropką (dot)
    1 – ściana
    2 – power pellet (duża energetyczna kropka)
    3 – wolne pole bez kropki (korytarz, dom duszka, tunel)
    4 – speed pellet (przyspiesza Pac-Mana na 3 sekundy)

PRZESTRZEŃ AKCJI (4 ruchy):
    0 – prawo  (col + 1)
    1 – lewo   (col - 1)
    2 – góra   (row - 1)
    3 – dół    (row + 1)

PRZESTRZEŃ STANÓW (wektor 15 cech):
    [0]  ghost_row / (ROWS-1)                          – znorm. wiersz duszka
    [1]  ghost_col / (COLS-1)                          – znorm. kolumna duszka
    [2]  pacman_row / (ROWS-1)                         – znorm. wiersz Pac-Mana
    [3]  pacman_col / (COLS-1)                         – znorm. kolumna Pac-Mana
    [4]  (pacman_row - ghost_row) / (ROWS-1)           – kierunek dr (podpisany)
    [5]  (pacman_col - ghost_col) / (COLS-1)           – kierunek dc (podpisany)
    [6]  manhattan_dist / (ROWS + COLS - 2)            – znorm. dystans Manhattana
    [7]  wall_up_ghost                                 – ściana nad duszkiem
    [8]  wall_down_ghost                               – ściana pod duszkiem
    [9]  wall_left_ghost                               – ściana po lewej duszka
    [10] wall_right_ghost                              – ściana po prawej duszka
    [11] wall_up_pacman                                – ściana nad Pac-Manem
    [12] wall_down_pacman                              – ściana pod Pac-Manem
    [13] wall_left_pacman                              – ściana po lewej Pac-Mana
    [14] wall_right_pacman                             – ściana po prawej Pac-Mana

FUNKCJA NAGRODY (dla duszka):
    +100.0  – złapanie Pac-Mana (koniec epizodu)
    +1.0    – zmniejszenie dystansu Manhattana do Pac-Mana
    -1.0    – zwiększenie dystansu Manhattana do Pac-Mana
    -5.0    – próba ruchu w ścianę (nieważna akcja)
    -0.1    – kara za każdy krok (zachęca do szybkiego łapania)
=============================================================================
"""

import random
import copy
import numpy as np

# ---------------------------------------------------------------------------
# STAŁE
# ---------------------------------------------------------------------------
ROWS, COLS = 19, 19
STATE_SIZE = 15    # długość wektora stanu
ACTION_SIZE = 4    # liczba możliwych akcji

# Kierunki: (delta_row, delta_col)
# Indeksy: 0=prawo, 1=lewo, 2=góra, 3=dół
ACTION_DELTAS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
ACTION_NAMES  = ["right", "left", "up", "down"]

# Odwrotna akcja (używana do zapobiegania zawracaniu)
OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}

# ---------------------------------------------------------------------------
# LABIRYNTY (trzy poziomy z v2.py)
# ---------------------------------------------------------------------------
MAZES = {
    1: [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 3, 3, 3, 1, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    2: [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 3, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 3, 3, 3, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    3: [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}

# ---------------------------------------------------------------------------
# PROFILE NAGRÓD (3 różne systemy kar i nagród)
# ---------------------------------------------------------------------------
REWARD_PROFILES = {
    1: {
        # Agresywny – wysoka nagroda za złapanie, duża kara za każdy krok
        # Efekt: duszek goni Pac-Mana bez wahania, ryzykuje ściany
        "catch_reward":       500.0,
        "direction_mult":      8.0,
        "step_penalty":       -1.0,
        "ghost_eaten":       -20.0,
        "timeout_penalty":  -100.0,
        "pacman_won":       -100.0,
    },
    2: {
        # Standardowy – zbalansowane wartości (domyślny)
        "catch_reward":       500.0,
        "direction_mult":      5.0,
        "step_penalty":       -0.5,
        "ghost_eaten":       -20.0,
        "timeout_penalty":   -50.0,
        "pacman_won":        -50.0,
    },
    3: {
        # Cierpliwy – mała kara za krok, duża kara za bycie zjedzonym
        # Efekt: duszek jest ostrożny, unika power pelletów
        "catch_reward":       300.0,
        "direction_mult":      3.0,
        "step_penalty":       -0.1,
        "ghost_eaten":      -100.0,
        "timeout_penalty":   -30.0,
        "pacman_won":        -30.0,
    },
}

# ---------------------------------------------------------------------------
# POZYCJE STARTOWE
# ---------------------------------------------------------------------------
PACMAN_START = (14, 9)   # (row, col) – środek dołu labiryntu
GHOST_START  = (8, 9)    # (row, col) – środek labiryntu (dom duszka)

# Limity epizodu (zapobiegają nieskończonym epizodom)
MAX_STEPS_PER_EPISODE = 400


# ===========================================================================
# KLASA ŚRODOWISKA
# ===========================================================================
class PacmanEnvironment:
    """
    Środowisko gry Pac-Man zgodne z interfejsem OpenAI Gym.

    Odpowiada za:
    - przechowywanie stanu gry (labirynt, pozycje graczy),
    - wykonywanie kroków symulacji (step),
    - obliczanie nagród dla agenta DQN (duszka),
    - generowanie wektorów stanu dla sieci neuronowej,
    - generowanie akcji autopilota dla Pac-Mana w trybie treningu.
    """

    def __init__(self, level: int = 1, training_mode: bool = False, reward_profile: int = 1):
        """
        Args:
            level:         Numer poziomu labiryntu (1, 2 lub 3).
            training_mode: Jeśli True, wyłącza power mode podczas treningu
                           (ghost uczy się gonować bez strachu przed zjedzeniem).
        """
        self.level = level
        self.training_mode = training_mode
        self.maze_template = MAZES[level]
        self._rewards = REWARD_PROFILES.get(reward_profile, REWARD_PROFILES[1])


        # Stan gry – inicjalizowany przez reset()
        self.maze: list[list[int]] = []
        self.ghost_row: int = 0
        self.ghost_col: int = 0
        self.pacman_row: int = 0
        self.pacman_col: int = 0
        self.prev_manhattan: float = 0.0
        self.ghost_last_action: int = 3   # dół jako akcja domyślna
        self.pacman_last_action: int = 0  # prawo jako akcja domyślna
        self.step_count: int = 0
        self.score: int = 0
        self.lives: int = 3
        self.done: bool = False
        self.game_won: bool = False
        self.power_mode: bool = False
        self.power_timer: int = 0

        self.reset()

    # -----------------------------------------------------------------------
    # RESET ŚRODOWISKA
    # -----------------------------------------------------------------------
    def reset(self) -> list[float]:
        """
        Resetuje środowisko do stanu początkowego.

        Returns:
            Wektor stanu (lista 15 liczb zmiennoprzecinkowych).
        """
        # Głęboka kopia labiryntu, by nie niszczyć szablonu
        self.maze = copy.deepcopy(self.maze_template)

        # Pozycje startowe
        if self.training_mode:
            # Losowe pozycje na wolnych polach — ghost uczy się ogólnej zasady "gonuj"
            free = [
                (r, c)
                for r in range(ROWS)
                for c in range(COLS)
                if self.maze[r][c] != 1
            ]
            pos = random.sample(free, 2)
            # Upewnij się że są wystarczająco daleko od siebie (min 5 kroków)
            for _ in range(200):
                pos = random.sample(free, 2)
                if abs(pos[0][0] - pos[1][0]) + abs(pos[0][1] - pos[1][1]) >= 5:
                    break
            self.ghost_row,  self.ghost_col  = pos[0]
            self.pacman_row, self.pacman_col = pos[1]
        else:
            self.ghost_row, self.ghost_col   = GHOST_START
            self.pacman_row, self.pacman_col = PACMAN_START

        # Historia do obliczania nagrody za zbliżanie/oddalanie
        self.prev_manhattan = self._manhattan()

        # Metadane
        self.ghost_last_action  = 3   # dół
        self.pacman_last_action = 0   # prawo
        self.step_count = 0
        self.score      = 0
        self.lives      = 3
        self.done       = False
        self.game_won   = False
        self.power_mode = False
        self.power_timer = 0

        return self.get_state()

    # -----------------------------------------------------------------------
    # KROK SYMULACJI
    # -----------------------------------------------------------------------
    def step(self, ghost_action: int, pacman_action: int):
        """
        Wykonuje jeden krok symulacji gry.

        Kolejność operacji:
        1. Przesuń duszka (akcja DQN).
        2. Przesuń Pac-Mana (akcja gracza lub autopilota).
        3. Sprawdź zbieranie kropek przez Pac-Mana.
        4. Sprawdź kolizję duszka z Pac-Manem.
        5. Oblicz nagrodę dla duszka.
        6. Sprawdź warunek końca epizodu.

        Args:
            ghost_action:  Akcja duszka (0–3) wybrana przez DQN.
            pacman_action: Akcja Pac-Mana (0–3) wybrana przez gracza lub autopilota.

        Returns:
            Krotka (state, reward, done, info):
                state  – nowy wektor stanu (lista 15 floatów),
                reward – nagroda dla duszka (float),
                done   – czy epizod się zakończył (bool),
                info   – słownik z dodatkowymi informacjami.
        """
        if self.done:
            # Epizod już skończony – zwróć stan końcowy bez zmian
            return self.get_state(), 0.0, True, {"reason": "episode_already_done"}

        reward = 0.0
        info   = {}

        # --- 1. RUCH DUSZKA ------------------------------------------------
        ghost_prev_r, ghost_prev_c = self.ghost_row, self.ghost_col
        pacman_prev_r, pacman_prev_c = self.pacman_row, self.pacman_col

        ghost_moved = self._try_move_ghost(ghost_action)
        if not ghost_moved:
            info["ghost_wall_hit"] = True
        else:
            self.ghost_last_action = ghost_action


        # --- 2. RUCH PAC-MANA ----------------------------------------------
        self._try_move_pacman(pacman_action)

        # --- 3. ZBIERANIE KROPEK PRZEZ PAC-MANA ----------------------------
        cell = self.maze[self.pacman_row][self.pacman_col]
        if cell == 0:
            self.maze[self.pacman_row][self.pacman_col] = 3
            self.score += 10
        elif cell == 2:
            self.maze[self.pacman_row][self.pacman_col] = 3
            self.score += 50
            if not self.training_mode:
                self.power_mode = True
                self.power_timer = 50   # ~50 kroków trybu mocy
        # Odliczanie trybu mocy
        if self.power_mode:
            self.power_timer -= 1
            if self.power_timer <= 0:
                self.power_mode = False

        # --- 4. KOLIZJA DUSZKA Z PAC-MANEM ---------------------------------
        if self._check_collision(ghost_prev_r, ghost_prev_c, pacman_prev_r, pacman_prev_c):
            if self.power_mode:
                # Pac-Man zjada duszka → kara, ale nie na tyle duża żeby uciekać od gracza
                reward += self._rewards["ghost_eaten"]
                self.score += 200
                info["ghost_eaten"] = True
                # Zresetuj pozycję duszka do domu
                self.ghost_row, self.ghost_col = GHOST_START
                self.ghost_last_action = 3
                self.power_mode = False
            else:
                # Duszek złapał Pac-Mana → dominująca nagroda
                reward += self._rewards["catch_reward"]
                info["caught_pacman"] = True
                self.done = True
                info["reason"] = "ghost_caught_pacman"
                self.step_count += 1
                return self.get_state(), reward, self.done, info

        # --- 5. NAGRODA: ruch ducha w stronę Pac-Mana -------------------------
        dist_now = self._manhattan()

        # Mierzymy czy DUSZEK przybliżył się do aktualnej pozycji Pac-Mana.
        # ghost_prev_r/c = pozycja ducha PRZED jego ruchem.
        # self.pacman_row/col = pozycja Pac-Mana PO jego ruchu.
        # Oba dystanse muszą uwzględniać tunel (zawijanie poziome) tak jak _manhattan().
        dr_b  = abs(ghost_prev_r - self.pacman_row)
        dc_b  = abs(ghost_prev_c - self.pacman_col)
        dist_ghost_before = float(dr_b + min(dc_b, COLS - dc_b))

        ghost_delta = dist_ghost_before - dist_now   # >0 = duszek zbliżył się

        reward += ghost_delta * self._rewards["direction_mult"]   # jedyny sygnał kierunkowy

        self.prev_manhattan = dist_now

        # Stała kara za krok — bez ruchu = zawsze ujemny wynik
        reward += self._rewards["step_penalty"]

        # --- 6. WARUNEK KOŃCA EPIZODU --------------------------------------
        self.step_count += 1

        # Przekroczenie limitu kroków — kara za niezłapanie
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            self.done = True
            reward += self._rewards["timeout_penalty"]
            info["reason"] = "max_steps_reached"

        # Wszystkie kropki zjedzone (Pac-Man wygrał poziom)
        if self._all_dots_eaten():
            self.done = True
            self.game_won = True
            reward += self._rewards["pacman_won"]   # Pac-Man wygrał → dodatkowa kara dla duszka
            info["reason"] = "pacman_won"

        return self.get_state(), reward, self.done, info

    # -----------------------------------------------------------------------
    # WEKTOR STANU DLA SIECI NEURONOWEJ
    # -----------------------------------------------------------------------
    def get_state(self) -> list[float]:
        """
        Zwraca wektor stanu jako listę 16 liczb zmiennoprzecinkowych
        gotową do podania na wejście sieci neuronowej DQN.

        Wszystkie wartości są znormalizowane do przedziału [-1, 1] lub [0, 1].
        """
        gr, gc = self.ghost_row, self.ghost_col
        pr, pc = self.pacman_row, self.pacman_col

        # Znormalizowane pozycje
        norm_gr = gr / (ROWS - 1)
        norm_gc = gc / (COLS - 1)
        norm_pr = pr / (ROWS - 1)
        norm_pc = pc / (COLS - 1)

        # Podpisane różnice (kierunek od duszka do Pac-Mana)
        delta_r = (pr - gr) / (ROWS - 1)
        delta_c = (pc - gc) / (COLS - 1)

        # Znormalizowany dystans Manhattana
        man_dist = self._manhattan() / (ROWS + COLS - 2)

        # Ściany wokół duszka (1 = ściana, 0 = wolne)
        wg_up    = 1.0 if self._is_wall(gr - 1, gc) else 0.0
        wg_down  = 1.0 if self._is_wall(gr + 1, gc) else 0.0
        wg_left  = 1.0 if self._is_wall(gr, gc - 1) else 0.0
        wg_right = 1.0 if self._is_wall(gr, gc + 1) else 0.0

        # Ściany wokół Pac-Mana
        wp_up    = 1.0 if self._is_wall(pr - 1, pc) else 0.0
        wp_down  = 1.0 if self._is_wall(pr + 1, pc) else 0.0
        wp_left  = 1.0 if self._is_wall(pr, pc - 1) else 0.0
        wp_right = 1.0 if self._is_wall(pr, pc + 1) else 0.0

        # Znormalizowany czas power-up (0.0 = brak / wygasł, 1.0 = pełny)
        power_timer_norm = self.power_timer / 50.0 if self.power_mode else 0.0

        return [
            norm_gr, norm_gc,
            norm_pr, norm_pc,
            delta_r, delta_c,
            man_dist,
            wg_up, wg_down, wg_left, wg_right,
            wp_up, wp_down, wp_left, wp_right,
            power_timer_norm,
        ]

    # -----------------------------------------------------------------------
    # STAN GRY DLA FRONTENDU (PEŁNA REPREZENTACJA)
    # -----------------------------------------------------------------------
    def get_game_state(self) -> dict:
        """
        Zwraca pełny stan gry jako słownik JSON do wysłania do przeglądarki.
        Frontend używa tych danych do renderowania gry na canvas.
        """
        return {
            "maze":         self.maze,
            "ghost":        {"row": self.ghost_row, "col": self.ghost_col},
            "pacman":       {"row": self.pacman_row, "col": self.pacman_col},
            "score":        self.score,
            "lives":        self.lives,
            "done":         self.done,
            "game_won":     self.game_won,
            "power_mode":   self.power_mode,
            "power_timer":  self.power_timer,
            "level":        self.level,
            "step_count":   self.step_count,
            "manhattan":    int(self._manhattan()),
        }

    # -----------------------------------------------------------------------
    # AUTOPILOT DLA PAC-MANA (TRYB TRENINGU)
    # -----------------------------------------------------------------------
    def get_autopilot_action(self) -> int:
        """
        Prosta heurystyka ucieczki Pac-Mana do użytku w trybie automatycznego
        treningu (bez udziału gracza).

        Strategia:
        1. Wyznacz kierunek od duszka do Pac-Mana.
        2. Preferuj ruch w kierunku przeciwnym do duszka (ucieczka).
        3. Jeśli ucieczka jest zablokowana ścianą, wybierz losowy dostępny ruch.
        4. Nie zawracaj, jeśli masz inną opcję.

        Returns:
            Akcja (0–3) dla Pac-Mana.
        """
        pr, pc = self.pacman_row, self.pacman_col
        gr, gc = self.ghost_row,  self.ghost_col

        dr = pr - gr  # ujemna → duszek jest poniżej Pac-Mana
        dc = pc - gc  # ujemna → duszek jest po prawej Pac-Mana

        # Kierunki ucieczki (od duszka): jeśli duszek jest na dole,
        # Pac-Man chce iść w górę (dr > 0 → akcja góra = 2)
        escape_actions = []
        if dr > 0:
            escape_actions.append(2)   # góra (uciekaj od duszka poniżej)
        elif dr < 0:
            escape_actions.append(3)   # dół
        if dc > 0:
            escape_actions.append(1)   # lewo (uciekaj od duszka po prawej)
        elif dc < 0:
            escape_actions.append(0)   # prawo

        # Filtruj do dostępnych (niesciennych) ruchów
        valid_actions = [a for a in range(ACTION_SIZE)
                         if self._can_pacman_move(a)]

        # Nie zawracaj, jeśli masz inną opcję
        opposite = OPPOSITE_ACTION.get(self.pacman_last_action, -1)
        non_backtrack = [a for a in valid_actions if a != opposite]
        if non_backtrack:
            valid_actions = non_backtrack

        # Preferuj kierunki ucieczki — wybierz ten który maksymalizuje dystans od duszka
        preferred = [a for a in escape_actions if a in valid_actions]
        if preferred:
            best, best_dist = preferred[0], -1
            for a in preferred:
                dr2, dc2 = ACTION_DELTAS[a]
                nr = (self.pacman_row + dr2) % ROWS
                nc = (self.pacman_col + dc2) % COLS
                d = abs(nr - self.ghost_row) + abs(nc - self.ghost_col)
                if d > best_dist:
                    best_dist = d
                    best = a
            return best

        # Fallback: wybierz ruch który najbardziej oddala od duszka
        if valid_actions:
            best, best_dist = valid_actions[0], -1
            for a in valid_actions:
                dr2, dc2 = ACTION_DELTAS[a]
                nr = (self.pacman_row + dr2) % ROWS
                nc = (self.pacman_col + dc2) % COLS
                d = abs(nr - self.ghost_row) + abs(nc - self.ghost_col)
                if d > best_dist:
                    best_dist = d
                    best = a
            return best

        # Ostateczność: oddaj poprzednią akcję
        return self.pacman_last_action

    # -----------------------------------------------------------------------
    # METODY POMOCNICZE – RUCH
    # -----------------------------------------------------------------------
    def _try_move_ghost(self, action: int) -> bool:
        """
        Próbuje przesunąć duszka w podanym kierunku.

        Returns:
            True jeśli ruch był możliwy (nie ściana), False w przeciwnym razie.
        """
        dr, dc = ACTION_DELTAS[action]
        new_r  = self.ghost_row + dr
        new_c  = self.ghost_col + dc

        # Obsługa tunelu (zawijanie poziome)
        new_c = new_c % COLS

        if self._is_wall(new_r, new_c):
            return False   # ściana – ruch zablokowany

        self.ghost_row = new_r
        self.ghost_col = new_c
        return True

    def _try_move_pacman(self, action: int) -> bool:
        """
        Próbuje przesunąć Pac-Mana w podanym kierunku.

        Returns:
            True jeśli ruch był możliwy (nie ściana), False w przeciwnym razie.
        """
        dr, dc = ACTION_DELTAS[action]
        new_r  = self.pacman_row + dr
        new_c  = self.pacman_col + dc

        # Obsługa tunelu
        new_c = new_c % COLS

        if self._is_wall(new_r, new_c):
            return False

        self.pacman_row = new_r
        self.pacman_col = new_c
        self.pacman_last_action = action
        return True

    def _can_pacman_move(self, action: int) -> bool:
        """Sprawdza, czy Pac-Man może wykonać dany ruch (nie ściana)."""
        dr, dc = ACTION_DELTAS[action]
        new_r  = (self.pacman_row + dr)
        new_c  = (self.pacman_col + dc) % COLS
        return not self._is_wall(new_r, new_c)

    # -----------------------------------------------------------------------
    # METODY POMOCNICZE – WARUNKI GRY
    # -----------------------------------------------------------------------
    def _is_wall(self, row: int, col: int) -> bool:
        """
        Sprawdza, czy komórka (row, col) jest ścianą lub jest poza planszą.

        Zawijanie poziome (tunel) jest obsługiwane przez % COLS przed wywołaniem.
        """
        if row < 0 or row >= ROWS:
            return True    # pionowe ograniczenia planszy
        col = col % COLS   # poziomy tunel
        return self.maze[row][col] == 1

    def _check_collision(self, ghost_prev_r=None, ghost_prev_c=None,
                         pacman_prev_r=None, pacman_prev_c=None) -> bool:
        """Sprawdza kolizję: ta sama komórka lub minięcie się w jednym kroku."""
        # Klasyczna kolizja: ta sama komórka
        if self.ghost_row == self.pacman_row and self.ghost_col == self.pacman_col:
            return True
        # Kolizja krzyżowa: zamiana miejsc w jednym kroku
        if ghost_prev_r is not None:
            if (self.ghost_row == pacman_prev_r and
                    self.ghost_col == pacman_prev_c and
                    self.pacman_row == ghost_prev_r and
                    self.pacman_col == ghost_prev_c):
                return True
        return False

    def _manhattan(self) -> float:
        """
        Oblicza dystans Manhattana między duszkiem a Pac-Manem,
        uwzględniając zawijanie tunelu w poziomie.
        """
        dr  = abs(self.ghost_row - self.pacman_row)
        dc_direct  = abs(self.ghost_col - self.pacman_col)
        dc_tunnel  = COLS - dc_direct   # odległość przez tunel
        dc = min(dc_direct, dc_tunnel)
        return float(dr + dc)

    def _all_dots_eaten(self) -> bool:
        """Sprawdza, czy Pac-Man zebrał wszystkie kropki i power pellety."""
        for row in self.maze:
            for cell in row:
                if cell in (0, 2):
                    return False
        return True

    # -----------------------------------------------------------------------
    # DOSTĘPNE AKCJE DLA DUSZKA
    # -----------------------------------------------------------------------
    def get_valid_ghost_actions(self) -> list[int]:
        """
        Zwraca listę akcji, które nie prowadzą duszka w ścianę.
        Używane do debugowania i wizualizacji.
        """
        valid = []
        for a in range(ACTION_SIZE):
            dr, dc = ACTION_DELTAS[a]
            new_r = self.ghost_row + dr
            new_c = (self.ghost_col + dc) % COLS
            if not self._is_wall(new_r, new_c):
                valid.append(a)
        return valid
