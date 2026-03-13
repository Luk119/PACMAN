"""
=============================================================================
BUFOR DOŚWIADCZEŃ (EXPERIENCE REPLAY BUFFER)
=============================================================================

Experience Replay to kluczowy komponent algorytmu DQN wprowadzony przez
DeepMind w 2013 roku (Mnih et al.). Rozwiązuje dwa fundamentalne problemy:

1. KORELACJA SEKWENCYJNA (Sequential Correlation):
   Kolejne próbki ze środowiska są silnie skorelowane (s_t → s_{t+1} → s_{t+2}).
   Trenowanie sieci na takich danych narusza założenie o niezależności
   próbek (i.i.d.) i prowadzi do niestabilnego uczenia.
   Rozwiązanie: losowe próbkowanie z bufora łamie korelację.

2. EFEKTYWNOŚĆ PRÓBEK (Sample Efficiency):
   Każde doświadczenie (s, a, r, s', done) może być użyte wielokrotnie
   do trenowania, zamiast tylko raz w kolejności zbierania.

IMPLEMENTACJA:
    Bufor działa jako cykliczne kolejka (deque) z maksymalną pojemnością.
    Gdy bufor jest pełny, najstarsze doświadczenia są automatycznie usuwane.
    Trenowanie odbywa się losowo próbkowanymi minibatchami.

Parametry zalecane:
    capacity  = 10 000 – 100 000 (większy = lepsza dywersyfikacja)
    batch_size = 32 – 128          (kompromis między stabilnością a szybkością)
=============================================================================
"""

import random
import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Cykliczny bufor przechowujący doświadczenia agenta w postaci krotek
    (state, action, reward, next_state, done).

    Umożliwia losowe próbkowanie minibatchy do trenowania sieci DQN.
    """

    def __init__(self, capacity: int = 10_000):
        """
        Args:
            capacity: Maksymalna liczba doświadczeń przechowywanych w buforze.
                     Gdy bufor jest pełny, najstarsze doświadczenia są usuwane.
        """
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state:      list[float],
        action:     int,
        reward:     float,
        next_state: list[float],
        done:       bool,
    ) -> None:
        """
        Dodaje jedno doświadczenie do bufora.

        Args:
            state:      Wektor stanu s_t (lista floatów, długość STATE_SIZE).
            action:     Wybrana akcja a_t (int, 0–3).
            reward:     Otrzymana nagroda r_t (float).
            next_state: Wektor następnego stanu s_{t+1}.
            done:       Czy epizod się zakończył (bool).
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Losuje minibatch doświadczeń z bufora.

        Args:
            batch_size: Liczba próbek do pobrania.

        Returns:
            Krotka pięciu tablic NumPy:
                states:      (batch_size, state_size)  float32
                actions:     (batch_size,)              int64
                rewards:     (batch_size,)              float32
                next_states: (batch_size, state_size)  float32
                dones:       (batch_size,)              float32  (0.0 lub 1.0)

        Raises:
            ValueError: Jeśli bufor ma mniej próbek niż batch_size.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Bufor zawiera {len(self.buffer)} próbek, "
                f"a wymagane batch_size = {batch_size}."
            )

        # Losowe próbkowanie bez powtórzeń
        batch = random.sample(self.buffer, batch_size)

        # Rozpakowanie krotek do osobnych list
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        """Zwraca aktualną liczbę doświadczeń w buforze."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Sprawdza, czy bufor zawiera wystarczająco dużo próbek
        do wykonania jednego kroku treningu.

        Args:
            batch_size: Wymagana minimalna liczba próbek.

        Returns:
            True jeśli bufor jest gotowy do próbkowania.
        """
        return len(self.buffer) >= batch_size

    @property
    def fill_ratio(self) -> float:
        """Zwraca stopień zapełnienia bufora jako liczbę z zakresu [0, 1]."""
        return len(self.buffer) / self.capacity
