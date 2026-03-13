"""
=============================================================================
SIEĆ NEURONOWA DQN – ARCHITEKTURA
=============================================================================

Deep Q-Network (DQN) to sieć neuronowa, która aproksymuje funkcję wartości
akcji Q(s, a). Na wejście przyjmuje wektor stanu s, a na wyjściu zwraca
wartości Q dla każdej z 4 akcji jednocześnie.

ARCHITEKTURA:
    Wejście:  state_size  (15 cech)
    Warstwa 1: Linear(15 → 256) + ReLU
    Warstwa 2: Linear(256 → 128) + ReLU
    Warstwa 3: Linear(128 → 64) + ReLU
    Wyjście:  Linear(64 → 4)  (bez aktywacji – surowe Q-wartości)

Dlaczego ta architektura?
- Trzy warstwy ukryte dają wystarczającą pojemność do aproksymacji
  funkcji Q w stosunkowo prostym środowisku 19×19.
- Brak BatchNorm i Dropout – środowisko jest deterministyczne,
  a regularyzacja przez epsilon-greedy zapobiega przeuczeniu.
- ReLU zamiast Sigmoid/Tanh – eliminuje problem zanikającego gradientu
  i jest szybsze obliczeniowo.

RÓWNANIE BELLMANA (podstawa aktualizacji DQN):
    Q(s, a) ← r + γ · max_{a'} Q_target(s', a')

gdzie:
    r      – natychmiastowa nagroda,
    γ      – współczynnik dyskontowania (gamma, 0 < γ ≤ 1),
    s'     – stan następny,
    Q_target – sieć docelowa (target network, zamrożona kopia Q-sieci).
=============================================================================
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Sieć neuronowa aproksymująca funkcję Q(s, a).

    Przyjmuje wektor stanu (batch_size × state_size) i zwraca
    Q-wartości dla każdej z action_size akcji (batch_size × action_size).
    """

    def __init__(self, state_size: int = 16, action_size: int = 4):
        """
        Args:
            state_size:  Wymiar wektora wejściowego (liczba cech stanu).
            action_size: Liczba możliwych akcji (wyjście sieci).
        """
        super(DQNNetwork, self).__init__()

        self.state_size  = state_size
        self.action_size = action_size

        # ---------------------------------------------------------------
        # WARSTWY SIECI
        # ---------------------------------------------------------------
        # Warstwa wejściowa: state_size → 256
        self.fc1 = nn.Linear(state_size, 256)
        self.relu1 = nn.ReLU()

        # Warstwa ukryta 1: 256 → 128
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()

        # Warstwa ukryta 2: 128 → 64
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()

        # Warstwa wyjściowa: 64 → action_size (surowe Q-wartości)
        self.fc4 = nn.Linear(64, action_size)

        # Inicjalizacja wag metodą He (Xavier dla ReLU)
        self._init_weights()

    def _init_weights(self):
        """
        Inicjalizacja wag sieci metodą He (kaiming_uniform).
        Poprawia stabilność treningu dzięki odpowiedniej skali gradientów
        na początku trenowania.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        # Warstwa wyjściowa – mała inicjalizacja dla stabilności Q-wartości
        nn.init.uniform_(self.fc4.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Przepływ w przód (forward pass).

        Args:
            x: Tensor kształtu (batch_size, state_size).

        Returns:
            Tensor Q-wartości kształtu (batch_size, action_size).
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)

    def get_action(self, state_tensor: torch.Tensor) -> int:
        """
        Wygodna metoda zwracająca akcję zachłanną (argmax Q-wartości)
        bez obliczania gradientów.

        Args:
            state_tensor: Tensor kształtu (1, state_size) lub (state_size,).

        Returns:
            Indeks akcji z najwyższą Q-wartością.
        """
        with torch.no_grad():
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.argmax(dim=1).item()
