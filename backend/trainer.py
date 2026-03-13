"""
=============================================================================
TRENER DQN – AUTOMATYCZNY TRENING BEZ UDZIAŁU GRACZA
=============================================================================

Moduł odpowiada za uruchamianie wieloepizodowego treningu agenta DQN
w tle (osobny wątek), bez potrzeby interakcji użytkownika z grą.

SCHEMAT EPIZODU TRENINGOWEGO:
    1. reset() → nowy stan środowiska
    2. Pętla kroków:
       a. Pac-Man: heurystyka ucieczki (autopilot)
       b. Duszek: ε-greedy DQN (eksploracja/eksploatacja)
       c. Środowisko: step(ghost_action, pacman_action)
       d. remember(s, a, r, s', done) → zapis do bufora
       e. train_step() → uczenie sieci (jeśli bufor gotowy)
    3. Zapis logów epizodu
    4. Powtórz dla następnego epizodu

OPCJE KONFIGURACJI:
    max_steps_per_episode: Maksymalna liczba kroków epizodu (awaryjne zakończenie)
    save_interval:         Co ile epizodów zapisywać model automatycznie
    log_interval:          Co ile epizodów logować do konsoli

DOSTĘPNE OPERACJE:
    start_training(episodes)  – uruchom trening w tle
    stop_training()           – zatrzymaj trening (gracefully)
    get_status()              – pobierz bieżący status i logi
    save_plot()               – zapisz wykres reward vs epizod (matplotlib)
=============================================================================
"""

import threading
import time
import os
from typing import TYPE_CHECKING

# Warunkowy import dla adnotacji typów (unikamy cyrkulacji)
if TYPE_CHECKING:
    from dqn_agent import DQNAgent

from environment import PacmanEnvironment


class Trainer:
    """
    Zarządza procesem automatycznego treningu agenta DQN.

    Trening odbywa się w osobnym wątku demona (daemon thread),
    dzięki czemu serwer Flask pozostaje responsywny podczas uczenia.
    """

    def __init__(
        self,
        agent:             "DQNAgent",
        max_steps_per_ep:  int = 500,
        save_interval:     int = 100,
        log_interval:      int = 10,
        model_path:        str = "models/ghost_dqn.pth",
        level:             int = 1,
    ):
        """
        Args:
            agent:            Instancja agenta DQNAgent do wytrenowania.
            max_steps_per_ep: Maksymalna liczba kroków na epizod.
            save_interval:    Co ile epizodów automatycznie zapisywać model.
            log_interval:     Co ile epizodów drukować postęp do konsoli.
            model_path:       Ścieżka zapisu modelu.
            level:            Poziom labiryntu (1–3).
        """
        self.agent            = agent
        self.max_steps_per_ep = max_steps_per_ep
        self.save_interval    = save_interval
        self.log_interval     = log_interval
        self.model_path       = model_path
        self.level            = level

        # --- STAN TRENINGU ---------------------------------------------------
        self.is_training    = False
        self.current_ep     = 0
        self.total_episodes = 0

        # Logi epizodów: lista słowników z metadanymi każdego epizodu
        self.episode_logs:  list[dict] = []

        # Numer sesji treningowej (rośnie przy każdym start_training)
        self.session_id: int = 0

        # Statystyki bieżące (szybki dostęp przez API)
        self.best_reward:   float = float("-inf")
        self.recent_rewards: list[float] = []  # ostatnie 100 epizodów

        # Wątek treningu
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Blokada dla bezpiecznego dostępu do współdzielonych zasobów
        self._lock = threading.Lock()

        # Podgląd na żywo: bieżący stan środowiska treningowego
        # Aktualizowany przez wątek treningu, czytany przez endpoint /training_live_state
        self.live_state: dict | None = None

    # =======================================================================
    # URUCHAMIANIE I ZATRZYMYWANIE TRENINGU
    # =======================================================================
    def start_training(self, episodes: int = 1000) -> bool:
        """
        Uruchamia trening w tle (osobny wątek demona).

        Args:
            episodes: Liczba epizodów do trenowania.

        Returns:
            True jeśli trening został uruchomiony, False jeśli już trwa.
        """
        if self.is_training:
            return False

        self._stop_event.clear()
        self.total_episodes = episodes
        self.current_ep     = 0
        self.session_id    += 1

        self._thread = threading.Thread(
            target=self._training_loop,
            args=(episodes,),
            daemon=True,     # wątek demona: kończy się z procesem głównym
            name="DQNTrainer",
        )
        self._thread.start()
        return True

    def stop_training(self) -> None:
        """Wysyła sygnał zatrzymania do wątku treningu (graceful stop)."""
        self._stop_event.set()
        self.is_training = False

    def is_alive(self) -> bool:
        """Sprawdza, czy wątek treningu jest nadal aktywny."""
        return self._thread is not None and self._thread.is_alive()

    # =======================================================================
    # PĘTLA TRENINGOWA (wykonywana w wątku tła)
    # =======================================================================
    def _training_loop(self, episodes: int) -> None:
        """
        Główna pętla treningowa. Każda iteracja to jeden epizod gry.
        Wywoływana w osobnym wątku.

        Args:
            episodes: Całkowita liczba epizodów do rozegrania.
        """
        self.is_training = True
        print(f"\n[Trainer] ▶ Rozpoczynam trening: {episodes} epizodów"
              f" | max_steps={self.max_steps_per_ep}"
              f" | device={self.agent.device}")

        # Nowe środowisko dla wątku treningowego (bez power mode — ghost uczy się gonować)
        env = PacmanEnvironment(level=self.level, training_mode=True)

        for ep in range(1, episodes + 1):
            # Sprawdzenie sygnału zatrzymania
            if self._stop_event.is_set():
                print(f"[Trainer] ⏹ Trening zatrzymany na epizodzie {ep}.")
                break

            # --- RESET ŚRODOWISKA ------------------------------------------
            state = env.reset()
            total_reward = 0.0
            step         = 0
            done         = False
            caught       = False   # czy duszek złapał Pac-Mana

            # --- PĘTLA KROKÓW EPIZODU --------------------------------------
            while not done and step < self.max_steps_per_ep:
                # Akcja duszka: ε-greedy DQN
                ghost_action = self.agent.select_action(state, training=True)

                # Akcja Pac-Mana: autopilot ucieczki
                pacman_action = env.get_autopilot_action()

                # Krok środowiska
                next_state, reward, done, info = env.step(
                    ghost_action, pacman_action
                )

                # Sprawdź czy duszek złapał
                if info.get("caught_pacman"):
                    caught = True

                # Aktualizuj podgląd na żywo (bez blokady – odczyt jest tolerancyjny)
                gs = env.get_game_state()
                gs["current_episode"] = ep
                gs["ghost_action"]    = ghost_action
                gs["step"]            = step
                self.live_state = gs

                # Zapis doświadczenia do bufora replay
                self.agent.remember(state, ghost_action, reward, next_state, done)

                # Złapanie: zapisz wielokrotnie żeby sieć zapamiętała ten sygnał
                if info.get("caught_pacman"):
                    for _ in range(4):
                        self.agent.remember(state, ghost_action, reward, next_state, done)

                # Krok uczenia (minibatch backprop)
                self.agent.train_step()

                state         = next_state
                total_reward += reward
                step         += 1

            # --- ZAKOŃCZENIE EPIZODU ---------------------------------------
            with self._lock:
                self.current_ep = ep

                # Zapis loga epizodu
                log_entry = {
                    "episode":  ep,
                    "reward":   round(total_reward, 2),
                    "epsilon":  round(self.agent.epsilon, 4),
                    "steps":    step,
                    "caught":   caught,
                    "avg_loss": round(self.agent.avg_loss, 6),
                    "session":  self.session_id,
                }
                self.episode_logs.append(log_entry)

                # Śledzenie najlepszego wyniku
                if total_reward > self.best_reward:
                    self.best_reward = total_reward

                # Okno ruchome dla ostatnich nagród (do obliczenia średniej)
                self.recent_rewards.append(total_reward)
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)

            # --- LOGOWANIE DO KONSOLI --------------------------------------
            if ep % self.log_interval == 0 or ep == 1:
                avg_r = (sum(self.recent_rewards) / len(self.recent_rewards)
                         if self.recent_rewards else 0.0)
                caught_str = "✓" if caught else "✗"
                print(
                    f"[Trainer] Ep {ep:5d}/{episodes}"
                    f" | R={total_reward:8.2f}"
                    f" | AvgR(100)={avg_r:8.2f}"
                    f" | ε={self.agent.epsilon:.4f}"
                    f" | steps={step:4d}"
                    f" | złapał={caught_str}"
                    f" | loss={self.agent.avg_loss:.5f}"
                )

            # --- AUTOMATYCZNY ZAPIS MODELU ---------------------------------
            if ep % self.save_interval == 0:
                self.agent.save(self.model_path)
                print(f"[Trainer] 💾 Model zapisany (ep {ep}): {self.model_path}")

        # Zapis końcowy
        self.agent.save(self.model_path)
        self.is_training = False
        print(f"\n[Trainer] ✅ Trening zakończony."
              f" Najlepszy wynik: {self.best_reward:.2f}"
              f" | Model: {self.model_path}")

    # =======================================================================
    # STATUS TRENINGU (dla API /training_status)
    # =======================================================================
    def get_status(self) -> dict:
        """
        Zwraca bieżący status treningu jako słownik JSON.

        Returns:
            Słownik zawierający:
                is_training:      bool – czy trening aktualnie trwa
                current_episode:  int  – numer bieżącego epizodu
                total_episodes:   int  – docelowa liczba epizodów
                progress_pct:     float – postęp w procentach
                best_reward:      float – najlepszy łączny wynik epizodu
                avg_reward_100:   float – średnia z ostatnich 100 epizodów
                recent_logs:      list  – ostatnie 50 logów epizodów
                agent_info:       dict  – informacje o agencie (ε, kroki itp.)
        """
        with self._lock:
            recent_logs = self.episode_logs[-50:]
            total_ep    = self.total_episodes
            current_ep  = self.current_ep
            best_r      = self.best_reward
            recent_r    = list(self.recent_rewards)

        progress = (current_ep / total_ep * 100) if total_ep > 0 else 0.0
        avg_r    = (sum(recent_r) / len(recent_r)) if recent_r else 0.0

        return {
            "is_training":    self.is_training,
            "current_episode": current_ep,
            "total_episodes":  total_ep,
            "progress_pct":    round(progress, 1),
            "best_reward":     round(best_r, 2) if best_r != float("-inf") else None,
            "avg_reward_100":  round(avg_r, 2),
            "recent_logs":     recent_logs,
            "agent_info":      self.agent.get_info(),
        }

    # =======================================================================
    # GENEROWANIE WYKRESU POSTĘPU
    # =======================================================================
    def save_plot(self, path: str = "models/training_plot.png") -> bool:
        """
        Generuje i zapisuje wykres nagrody vs numer epizodu (matplotlib).

        Args:
            path: Ścieżka zapisu wykresu (PNG).

        Returns:
            True jeśli wykres został zapisany, False jeśli nie ma danych
            lub matplotlib nie jest dostępny.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # backend bez GUI
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Trainer] matplotlib nie jest zainstalowany.")
            return False

        with self._lock:
            logs = list(self.episode_logs)

        if not logs:
            return False

        # Grupuj logi według sesji
        sessions = {}
        global_ep = 0
        for e in logs:
            sid = e.get("session", 1)
            if sid not in sessions:
                sessions[sid] = {"episodes": [], "rewards": [], "epsilons": []}
            global_ep += 1
            sessions[sid]["episodes"].append(global_ep)
            sessions[sid]["rewards"].append(e["reward"])
            sessions[sid]["epsilons"].append(e["epsilon"])

        palette = ["steelblue", "tomato", "mediumseagreen", "mediumpurple",
                   "darkorange", "deeppink", "teal", "goldenrod"]

        window = 20

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("DQN Ghost Training Progress", fontsize=14, fontweight="bold")

        for idx, (sid, data) in enumerate(sorted(sessions.items())):
            color = palette[idx % len(palette)]
            eps   = data["episodes"]
            rews  = data["rewards"]
            epsls = data["epsilons"]

            # Ruchome średnie w obrębie sesji
            smoothed = []
            for i in range(len(rews)):
                start = max(0, i - window + 1)
                smoothed.append(sum(rews[start:i+1]) / (i - start + 1))

            label = f"Sesja {sid}"
            ax1.plot(eps, rews, alpha=0.25, color=color)
            ax1.plot(eps, smoothed, color=color, linewidth=2,
                     label=f"{label} – śr. krocząca ({window} ep)")

            ax2.plot(eps, epsls, color=color, linewidth=2, label=label)

            # Pionowa linia oddzielająca sesje
            if eps:
                ax1.axvline(x=eps[0], color=color, linestyle=":", alpha=0.5)
                ax2.axvline(x=eps[0], color=color, linestyle=":", alpha=0.5)

        ax1.set_xlabel("Epizod (globalny)")
        ax1.set_ylabel("Łączna nagroda")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        ax2.set_xlabel("Epizod (globalny)")
        ax2.set_ylabel("Epsilon (ε)")
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_title("Zanik eksploracji (ε-decay)")

        plt.tight_layout()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[Trainer] Wykres zapisany: {path}")
        return True
