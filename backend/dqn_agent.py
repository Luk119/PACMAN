"""
=============================================================================
AGENT DEEP Q-LEARNING (DQN)
=============================================================================

Implementacja algorytmu DQN (Deep Q-Network) na podstawie artykułu:
"Human-level control through deep reinforcement learning"
Mnih et al., Nature 2015.

ALGORYTM DQN – PRZEGLĄD:
─────────────────────────────────────────────────────────────────────────────
1. Inicjalizacja:
   - Sieć Q (online network) z losowymi wagami θ
   - Sieć docelowa (target network) z wagami θ⁻ = θ
   - Bufor doświadczeń D

2. Dla każdego kroku t:
   a. Z prawdopodobieństwem ε wybierz losową akcję (eksploracja),
      w przeciwnym razie wybierz a = argmax_a Q(s_t, a; θ) (eksploatacja).
   b. Wykonaj akcję a, obserwuj nagrodę r i następny stan s_{t+1}.
   c. Zapisz krotkę (s_t, a, r, s_{t+1}, done) w buforze D.
   d. Losuj minibatch z D.
   e. Wyznacz cel:
      y_j = r_j                          jeśli done
      y_j = r_j + γ · max_{a'} Q(s'_j, a'; θ⁻)   w przeciwnym razie
   f. Minimalizuj stratę: L = (y_j - Q(s_j, a_j; θ))²
   g. Co N kroków: θ⁻ ← θ  (aktualizacja target network)
   h. Zmniejsz ε (ε-decay)

SIEĆ DOCELOWA (Target Network):
    Używamy zamrożonej kopii sieci Q do obliczania celów treningowych.
    Bez tego mechanizmu, aktualizacja Q zmieniałaby jednocześnie
    cele i predykcje, co prowadzi do niestabilności (moving target problem).
    Co N kroków kopiujemy wagi Q → target.

PARAMETRY HIPERPARAMETRYCZNE:
    gamma             = 0.99   – współczynnik dyskontowania
    epsilon_start     = 1.00   – początk. prawdopodobieństwo eksploracji
    epsilon_min       = 0.01   – minimalne prawdopodobieństwo eksploracji
    epsilon_decay     = 0.995  – mnożnik ε po każdym kroku (wykładniczy)
    learning_rate     = 0.001  – szybkość uczenia optymalizatora Adam
    batch_size        = 64     – rozmiar minibatcha
    target_update_freq= 100    – co ile kroków kopiujemy Q → target
    buffer_capacity   = 10000  – pojemność bufora doświadczeń
    grad_clip         = 1.0    – przycinanie gradientów (stabilność)
=============================================================================
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Agent uczący się sterowania duszkiem w grze Pac-Man
    przy użyciu algorytmu Deep Q-Learning.

    Przechowuje:
    - sieć online (Q-network) – aktualizowana co krok,
    - sieć docelową (target network) – kopiowana co N kroków,
    - bufor doświadczeń (ReplayBuffer),
    - stan ε-greedy (epsilon i jego harmonogram zaniku).
    """

    def __init__(
        self,
        state_size:          int   = 16,
        action_size:         int   = 4,
        gamma:               float = 0.99,
        epsilon_start:       float = 1.0,
        epsilon_min:         float = 0.01,
        epsilon_decay:       float = 0.9997,
        learning_rate:       float = 1e-3,
        batch_size:          int   = 64,
        target_update_freq:  int   = 50,
        buffer_capacity:     int   = 10_000,
        grad_clip:           float = 1.0,
    ):
        """
        Args:
            state_size:         Wymiar wektora stanu (15 cech).
            action_size:        Liczba akcji (4: prawo, lewo, góra, dół).
            gamma:              Współczynnik dyskontowania przyszłych nagród.
            epsilon_start:      Początkowe prawdopodobieństwo losowej akcji.
            epsilon_min:        Minimalne prawdopodobieństwo losowej akcji.
            epsilon_decay:      Mnożnik ε po każdym kroku uczenia.
            learning_rate:      Szybkość uczenia optymalizatora Adam.
            batch_size:         Rozmiar minibatcha do uczenia.
            target_update_freq: Co ile kroków aktualizować target network.
            buffer_capacity:    Pojemność bufora doświadczeń.
            grad_clip:          Maksymalna norma gradientu (gradient clipping).
        """
        self.state_size         = state_size
        self.action_size        = action_size
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip          = grad_clip

        # Urządzenie obliczeniowe: GPU jeśli dostępne, CPU w przeciwnym razie
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- SIECI NEURONOWE ------------------------------------------------
        # Sieć online: parametry θ aktualizowane każdy krok uczenia
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)

        # Sieć docelowa: parametry θ⁻ kopiowane z Q co target_update_freq kroków
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()   # target network tylko do inferencji

        # --- OPTYMALIZATOR --------------------------------------------------
        # Adam: adaptacyjny optymalizator, dobrze sprawdza się w DQN
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        # --- FUNKCJA STRATY -------------------------------------------------
        # Huber loss (smooth L1) – mniej wrażliwa na outliers niż MSE
        self.loss_fn = nn.SmoothL1Loss()

        # --- BUFOR DOŚWIADCZEŃ ----------------------------------------------
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # --- LICZNIKI -------------------------------------------------------
        self.training_steps  = 0   # łączna liczba kroków uczenia
        self.total_steps     = 0   # łączna liczba kroków środowiska

        # Historia strat (do monitorowania)
        self.loss_history: list[float] = []

    # =======================================================================
    # WYBÓR AKCJI (ε-GREEDY)
    # =======================================================================
    def select_action(
        self,
        state:    list[float],
        training: bool = False,
    ) -> int:
        """
        Wybiera akcję zgodnie z polityką ε-greedy.

        W trybie treningu (training=True):
        - Z prawdopodobieństwem ε → losowa akcja (eksploracja)
        - Z prawdopodobieństwem (1-ε) → akcja zachłanna z Q-sieci (eksploatacja)

        W trybie inferencji (training=False):
        - Zawsze akcja zachłanna (ε=0).

        Args:
            state:    Wektor stanu (lista 15 floatów).
            training: Czy stosować ε-greedy (True) czy zawsze greedy (False).

        Returns:
            Indeks wybranej akcji (0–3).
        """
        self.total_steps += 1

        if training and random.random() < self.epsilon:
            # EKSPLORACJA: losowa akcja
            return random.randint(0, self.action_size - 1)

        # EKSPLOATACJA: akcja z najwyższą Q-wartością
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def select_action_with_qvalues(
        self,
        state: list[float],
    ) -> tuple[int, list[float]]:
        """
        Zwraca akcję zachłanną oraz Q-wartości dla wszystkich akcji.
        Używane do wizualizacji w frontendzie.

        Args:
            state: Wektor stanu.

        Returns:
            Krotka (action, q_values_list).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        q_list  = q_values.squeeze(0).cpu().tolist()
        action  = int(q_values.argmax(dim=1).item())
        return action, q_list

    # =======================================================================
    # ZAPAMIĘTYWANIE DOŚWIADCZENIA
    # =======================================================================
    def remember(
        self,
        state:      list[float],
        action:     int,
        reward:     float,
        next_state: list[float],
        done:       bool,
    ) -> None:
        """
        Zapisuje krotkę doświadczenia do bufora.

        Args:
            state:      Stan przed akcją s_t.
            action:     Wykonana akcja a_t.
            reward:     Otrzymana nagroda r_t.
            next_state: Stan po akcji s_{t+1}.
            done:       Czy epizod się zakończył.
        """
        self.memory.push(state, action, reward, next_state, done)

    # =======================================================================
    # KROK UCZENIA
    # =======================================================================
    def train_step(self) -> float | None:
        """
        Wykonuje jeden krok aktualizacji sieci Q.

        Algorytm:
        1. Pobierz losowy minibatch z bufora doświadczeń.
        2. Oblicz aktualne Q-wartości: Q(s, a; θ).
        3. Oblicz cele Bellmana: y = r + γ · max_{a'} Q(s', a'; θ⁻).
        4. Oblicz stratę Hubera: L = Huber(Q(s,a;θ), y).
        5. Propagacja wsteczna i aktualizacja θ.
        6. Co target_update_freq kroków: θ⁻ ← θ.
        7. ε-decay: ε ← max(ε·decay, ε_min).

        Returns:
            Wartość straty (float) lub None jeśli bufor nie jest gotowy.
        """
        if not self.memory.is_ready(self.batch_size):
            return None

        # --- PRÓBKOWANIE MINIBATCHA ----------------------------------------
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Konwersja na tensory PyTorch
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # --- AKTUALNE Q-WARTOŚCI -------------------------------------------
        # Q(s_j, a_j; θ) – Q-wartość dla faktycznie podjętej akcji
        current_q = self.q_network(states_t) \
                        .gather(1, actions_t.unsqueeze(1)) \
                        .squeeze(1)

        # --- CELE BELLMANA --------------------------------------------------
        # Obliczane przez ZAMROŻONĄ sieć docelową (θ⁻)
        with torch.no_grad():
            # max_{a'} Q(s'_j, a'; θ⁻)
            next_q_max = self.target_network(next_states_t).max(dim=1)[0]
            # y_j = r_j + γ · max Q(s'_j, a'; θ⁻) · (1 - done_j)
            target_q = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # --- OBLICZENIE STRATY ----------------------------------------------
        loss = self.loss_fn(current_q, target_q)

        # --- PROPAGACJA WSTECZNA -------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping – stabilizuje trening (zapobiega eksplozji gradientów)
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.grad_clip,
        )

        self.optimizer.step()

        # --- AKTUALIZACJA LICZNIKÓW ----------------------------------------
        self.training_steps += 1
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        # Przechowuj max 1000 ostatnich strat
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]

        # --- AKTUALIZACJA TARGET NETWORK ------------------------------------
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # --- ε-DECAY --------------------------------------------------------
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

        return loss_val

    # =======================================================================
    # ZAPIS I WCZYTANIE MODELU
    # =======================================================================
    def save(self, path: str = "models/ghost_dqn.pth") -> None:
        """
        Zapisuje stan agenta do pliku .pth (format PyTorch).

        Zapisuje:
        - wagi sieci online (q_network),
        - wagi sieci docelowej (target_network),
        - stan optymalizatora,
        - aktualny epsilon,
        - licznik kroków uczenia.

        Args:
            path: Ścieżka do pliku zapisu.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "q_network":      self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer":      self.optimizer.state_dict(),
                "epsilon":        self.epsilon,
                "training_steps": self.training_steps,
                "total_steps":    self.total_steps,
                "loss_history":   self.loss_history[-200:],
            },
            path,
        )
        print(f"[DQNAgent] Model zapisany: {path}")

    def load(self, path: str = "models/ghost_dqn.pth") -> bool:
        """
        Wczytuje stan agenta z pliku .pth.

        Args:
            path: Ścieżka do pliku modelu.

        Returns:
            True jeśli wczytano pomyślnie, False jeśli plik nie istnieje.
        """
        if not os.path.exists(path):
            print(f"[DQNAgent] Plik modelu nie istnieje: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon        = checkpoint.get("epsilon",        self.epsilon_min)
        self.training_steps = checkpoint.get("training_steps", 0)
        self.total_steps    = checkpoint.get("total_steps",    0)
        self.loss_history   = checkpoint.get("loss_history",   [])
        print(f"[DQNAgent] Model wczytany: {path}, ε={self.epsilon:.4f}")
        return True

    # =======================================================================
    # WŁAŚCIWOŚCI DIAGNOSTYCZNE
    # =======================================================================
    @property
    def avg_loss(self) -> float:
        """Średnia strata z ostatnich 100 kroków uczenia."""
        if not self.loss_history:
            return 0.0
        recent = self.loss_history[-100:]
        return float(np.mean(recent))

    def get_info(self) -> dict:
        """
        Zwraca słownik z metadanymi agenta (do logowania i API).
        """
        return {
            "epsilon":        round(self.epsilon, 4),
            "training_steps": self.training_steps,
            "total_steps":    self.total_steps,
            "buffer_size":    len(self.memory),
            "buffer_ready":   self.memory.is_ready(self.batch_size),
            "avg_loss":       round(self.avg_loss, 6),
            "device":         str(self.device),
        }
