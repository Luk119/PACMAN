"""
=============================================================================
SERWER FLASK – REST API DLA GRY PAC-MAN Z DQN
=============================================================================

Aplikacja Flask obsługująca dwa tryby działania:

TRYB GRY INTERAKTYWNEJ (/play):
    Gracz steruje Pac-Manem przez klawiaturę (frontend → API).
    Duszek sterowany jest przez wytrenowany model DQN.
    Opcjonalnie można włączyć uczenie online podczas gry.

TRYB AUTOMATYCZNEGO TRENINGU (/train):
    Pac-Man i duszek działają w pełni automatycznie (bez gracza).
    Środowisko resetuje się po każdym epizodzie.
    Trening odbywa się w tle (daemon thread).

ENDPOINTY REST:
    GET  /                    → serwuje frontend (index.html)
    GET  /game_state          → pełny stan gry (maze, pozycje, score)
    POST /step                → krok gry { pacman_action: 0-3 }
    POST /reset               → reset gry do stanu początkowego
    GET  /get_action          → akcja duszka dla bieżącego stanu (bez kroku)
    POST /train               → start treningu { episodes: int, level: int }
    POST /stop_training       → zatrzymanie treningu
    GET  /training_status     → status treningu (logi, epsilon, avg_reward)
    POST /save_model          → zapisz model { path: str (opcjonalnie) }
    POST /load_model          → wczytaj model { path: str (opcjonalnie) }
    GET  /training_plot       → wykres postępu treningu (PNG)
    POST /toggle_online_learn → przełącz uczenie online { enabled: bool }
    GET  /agent_info          → informacje o stanie agenta
=============================================================================
"""

import os
import sys
import json
import threading
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

# Dodaj katalog backend do PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

from environment import PacmanEnvironment
from dqn_agent   import DQNAgent
from trainer     import Trainer

# ---------------------------------------------------------------------------
# INICJALIZACJA APLIKACJI
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend"),
    static_url_path="",
)
CORS(app)   # Zezwól na CORS dla wszystkich endpointów (dev)

# ---------------------------------------------------------------------------
# GLOBALNE SINGLETONY
# ---------------------------------------------------------------------------
# Środowisko gry (dla trybu interaktywnego)
game_env = PacmanEnvironment(level=1)

# Agent DQN (wspólny dla trybu gry i treningu)
agent = DQNAgent(
    state_size         = 16,
    action_size        = 4,
    gamma              = 0.99,
    epsilon_start      = 1.0,
    epsilon_min        = 0.01,
    epsilon_decay      = 0.9997,
    learning_rate      = 1e-3,
    batch_size         = 64,
    target_update_freq = 50,
    buffer_capacity    = 50_000,
    grad_clip          = 1.0,
)

# Trener (obsługuje wątek treningu w tle)
trainer = Trainer(
    agent           = agent,
    max_steps_per_ep= 500,
    save_interval   = 100,
    log_interval    = 10,
    model_path      = os.path.join(os.path.dirname(__file__), "models", "ghost_dqn.pth"),
    level           = 1,
)

# Flagi trybu
online_learning_enabled = False   # uczenie podczas interaktywnej gry
game_lock = threading.Lock()      # blokada dostępu do game_env

# ---------------------------------------------------------------------------
# TABLICA WYNIKÓW – PERSISTENCJA
# ---------------------------------------------------------------------------
_SCORES_PATH = os.path.join(os.path.dirname(__file__), "models", "scores.json")
_scores_lock = threading.Lock()


def _load_scores() -> dict:
    if os.path.exists(_SCORES_PATH):
        try:
            with open(_SCORES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"1": [], "2": [], "3": [], "guest_counter": 0}


def _save_scores(scores: dict) -> None:
    os.makedirs(os.path.dirname(_SCORES_PATH), exist_ok=True)
    with open(_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

# Spróbuj wczytać istniejący model przy starcie
_model_path = os.path.join(os.path.dirname(__file__), "models", "ghost_dqn.pth")
if os.path.exists(_model_path):
    agent.load(_model_path)
    print(f"[App] Model wczytany automatycznie: {_model_path}")
else:
    print("[App] Brak zapisanego modelu – agent rozpoczyna od zera (ε=1.0).")


# ===========================================================================
# ENDPOINTY: FRONTEND
# ===========================================================================
@app.route("/")
def index():
    """Serwuje stronę główną gry (index.html z katalogu frontend)."""
    return send_from_directory(app.static_folder, "index.html")


# ===========================================================================
# ENDPOINTY: TRYB GRY INTERAKTYWNEJ
# ===========================================================================
@app.route("/game_state", methods=["GET"])
def get_game_state():
    """
    Zwraca pełny stan gry jako JSON.

    Response:
        {
            maze:       [[int, ...], ...],  – 19×19 labirynt (0-3)
            ghost:      {row, col},
            pacman:     {row, col},
            score:      int,
            lives:      int,
            done:       bool,
            game_won:   bool,
            power_mode: bool,
            level:      int,
            step_count: int,
            manhattan:  int,
            agent_info: {epsilon, training_steps, ...}
        }
    """
    with game_lock:
        state_dict = game_env.get_game_state()
    state_dict["agent_info"]            = agent.get_info()
    state_dict["online_learning_enabled"] = online_learning_enabled
    return jsonify(state_dict)


@app.route("/step", methods=["POST"])
def step():
    """
    Wykonuje jeden krok gry.

    Request JSON:
        { "pacman_action": int }   – akcja gracza: 0=prawo, 1=lewo, 2=góra, 3=dół

    Response:
        Pełny stan gry po kroku (jak /game_state) + pole "reward".
    """
    global online_learning_enabled

    data          = request.get_json(force=True, silent=True) or {}
    pacman_action = int(data.get("pacman_action", -1))

    # Walidacja akcji (0–3); -1 = brak akcji (Pac-Man stoi w miejscu)
    if pacman_action not in (0, 1, 2, 3):
        pacman_action = game_env.pacman_last_action

    with game_lock:
        if game_env.done:
            state_dict = game_env.get_game_state()
            state_dict["reward"] = 0.0
            return jsonify(state_dict)

        # Pobierz bieżący stan dla agenta
        current_state = game_env.get_state()

        # Akcja duszka z modelu DQN (ε=0 → zawsze greedy w trybie gry)
        ghost_action, q_values = agent.select_action_with_qvalues(current_state)

        # Krok środowiska
        next_state, reward, done, info = game_env.step(ghost_action, pacman_action)

        # Opcjonalne uczenie online podczas gry
        if online_learning_enabled:
            agent.remember(current_state, ghost_action, reward, next_state, done)
            agent.train_step()

        # Jeśli gra zakończona – automatyczny reset po chwili
        state_dict = game_env.get_game_state()

    state_dict["reward"]    = round(reward, 3)
    state_dict["q_values"]  = [round(q, 3) for q in q_values]
    state_dict["ghost_action"] = ghost_action
    state_dict["agent_info"] = agent.get_info()
    state_dict["online_learning_enabled"] = online_learning_enabled
    return jsonify(state_dict)


@app.route("/reset", methods=["POST"])
def reset_game():
    """
    Resetuje grę do stanu początkowego.

    Request JSON (opcjonalnie):
        { "level": int }  – poziom labiryntu (1–3), domyślnie 1

    Response:
        Pełny stan gry po resecie.
    """
    data  = request.get_json(force=True, silent=True) or {}
    level = int(data.get("level", 1))
    if level not in (1, 2, 3):
        level = 1

    with game_lock:
        # Utwórz nowe środowisko dla wybranego poziomu i skopiuj jego stan
        _new_env = PacmanEnvironment(level=level)
        game_env.__dict__.update(_new_env.__dict__)

    state_dict = game_env.get_game_state()
    state_dict["agent_info"] = agent.get_info()
    return jsonify(state_dict)


@app.route("/get_action", methods=["GET"])
def get_action():
    """
    Zwraca akcję duszka dla bieżącego stanu gry BEZ wykonywania kroku.
    Używane do podglądu decyzji modelu.

    Response:
        {
            action:    int,          – wybrana akcja (0–3)
            action_name: str,        – nazwa akcji
            q_values:  [float, ...], – Q-wartości dla wszystkich akcji
        }
    """
    with game_lock:
        state = game_env.get_state()
    action, q_values = agent.select_action_with_qvalues(state)
    action_names = ["right", "left", "up", "down"]
    return jsonify({
        "action":      action,
        "action_name": action_names[action],
        "q_values":    [round(q, 4) for q in q_values],
    })


@app.route("/toggle_online_learn", methods=["POST"])
def toggle_online_learn():
    """
    Włącza lub wyłącza uczenie online podczas interaktywnej gry.

    Request JSON:
        { "enabled": bool }

    Response:
        { "online_learning_enabled": bool }
    """
    global online_learning_enabled
    data    = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", not online_learning_enabled))
    online_learning_enabled = enabled
    return jsonify({"online_learning_enabled": online_learning_enabled})


# ===========================================================================
# ENDPOINTY: TRENING
# ===========================================================================
@app.route("/train", methods=["POST"])
def start_training():
    """
    Uruchamia automatyczny trening DQN w tle.

    Request JSON:
        {
            "episodes": int,   – liczba epizodów (domyślnie 500)
            "level":    int,   – poziom labiryntu (1–3, domyślnie 1)
        }

    Response:
        { "message": str, "episodes": int, "level": int }
    """
    data     = request.get_json(force=True, silent=True) or {}
    episodes = int(data.get("episodes", 500))
    level    = int(data.get("level",    1))

    if episodes < 1 or episodes > 100_000:
        return jsonify({"error": "episodes musi być w zakresie 1–100000"}), 400
    if level not in (1, 2, 3):
        level = 1

    if trainer.is_training:
        return jsonify({"error": "Trening już trwa.", "is_training": True}), 409

    # Skonfiguruj trener dla wybranego poziomu
    trainer.level = level
    success = trainer.start_training(episodes=episodes)

    if success:
        return jsonify({
            "message":  f"Trening uruchomiony: {episodes} epizodów, poziom {level}.",
            "episodes": episodes,
            "level":    level,
        })
    return jsonify({"error": "Nie udało się uruchomić treningu."}), 500


@app.route("/stop_training", methods=["POST"])
def stop_training():
    """
    Zatrzymuje bieżący trening.

    Response:
        { "message": str, "stopped": bool }
    """
    trainer.stop_training()
    return jsonify({
        "message": "Trening zostanie zatrzymany.",
        "stopped": True,
    })


@app.route("/training_status", methods=["GET"])
def training_status():
    """
    Zwraca bieżący status treningu.

    Response:
        {
            is_training:      bool,
            current_episode:  int,
            total_episodes:   int,
            progress_pct:     float,
            best_reward:      float | null,
            avg_reward_100:   float,
            recent_logs:      [{episode, reward, epsilon, steps, caught, avg_loss}, ...],
            agent_info:       {epsilon, training_steps, buffer_size, ...}
        }
    """
    return jsonify(trainer.get_status())


# ===========================================================================
# ENDPOINTY: ZAPIS / WCZYTANIE MODELU
# ===========================================================================
@app.route("/save_model", methods=["POST"])
def save_model():
    """
    Zapisuje model DQN do pliku.

    Request JSON (opcjonalnie):
        { "path": str }  – ścieżka zapisu, domyślnie models/ghost_dqn.pth

    Response:
        { "message": str, "path": str }
    """
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path", os.path.join(
        os.path.dirname(__file__), "models", "ghost_dqn.pth"
    ))
    try:
        agent.save(path)
        return jsonify({"message": f"Model zapisany: {path}", "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/load_model", methods=["POST"])
def load_model():
    """
    Wczytuje model DQN z pliku.

    Request JSON (opcjonalnie):
        { "path": str }  – ścieżka do pliku .pth

    Response:
        { "message": str, "loaded": bool, "agent_info": {...} }
    """
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path", os.path.join(
        os.path.dirname(__file__), "models", "ghost_dqn.pth"
    ))
    loaded = agent.load(path)
    return jsonify({
        "message":    f"Model wczytany: {path}" if loaded else f"Plik nie istnieje: {path}",
        "loaded":     loaded,
        "agent_info": agent.get_info(),
    })


@app.route("/training_live_state", methods=["GET"])
def training_live_state():
    """
    Zwraca aktualny stan środowiska treningowego (do podglądu na żywo).

    Dostępne tylko gdy trening jest aktywny. Frontend renderuje ten stan
    na canvasie, umożliwiając obserwację uczenia się duszka w czasie rzeczywistym.

    Response:
        Pełny stan gry jak /game_state + pole current_episode i step,
        lub { available: false } jeśli trening nie jest aktywny.
    """
    if not trainer.is_training or trainer.live_state is None:
        return jsonify({
            "available": False,
            "is_training": trainer.is_training,
        })
    state = dict(trainer.live_state)
    state["available"]   = True
    state["agent_info"]  = agent.get_info()
    return jsonify(state)


@app.route("/training_plot", methods=["GET"])
def training_plot():
    """
    Generuje wykres postępu treningu (nagroda vs epizod) i zwraca jako PNG.
    Wymaga zainstalowanego matplotlib.

    Response:
        Plik PNG lub JSON z błędem.
    """
    plot_path = os.path.join(os.path.dirname(__file__), "models", "training_plot.png")
    success   = trainer.save_plot(path=plot_path)
    if success:
        return send_file(plot_path, mimetype="image/png")
    return jsonify({"error": "Brak danych treningu lub brak matplotlib."}), 404


# ===========================================================================
# ENDPOINT: INFORMACJE O AGENCIE
# ===========================================================================
@app.route("/agent_info", methods=["GET"])
def agent_info():
    """
    Zwraca informacje diagnostyczne o agencie DQN.

    Response:
        {
            epsilon:        float,
            training_steps: int,
            total_steps:    int,
            buffer_size:    int,
            buffer_ready:   bool,
            avg_loss:       float,
            device:         str,
        }
    """
    return jsonify(agent.get_info())


# ===========================================================================
# ENDPOINTY: TABLICA WYNIKÓW
# ===========================================================================
@app.route("/scores", methods=["GET"])
def get_scores():
    """
    Zwraca top-10 wyników dla wybranego poziomu.

    Query params:
        level: int  – poziom (1–3), domyślnie 1

    Response:
        { level: int, scores: [{nick, score, time, date}, ...] }
    """
    level = request.args.get("level", "1")
    if level not in ("1", "2", "3"):
        level = "1"
    with _scores_lock:
        data = _load_scores()
    entries = data.get(level, [])
    entries = sorted(entries, key=lambda x: (-x["score"], x.get("time", 99999)))
    return jsonify({
        "level":         int(level),
        "scores":        entries[:10],
        "guest_counter": data.get("guest_counter", 0),
    })


@app.route("/scores", methods=["POST"])
def post_score():
    """
    Zapisuje wynik gracza.

    Request JSON:
        {
            "nick":  str   – opcjonalny nick (jeśli brak → Guest_N),
            "score": int   – wynik,
            "level": int   – poziom (1–3),
            "time":  float – czas ukończenia w sekundach
        }

    Response:
        { message: str, nick: str, scores: [...top 10...] }
    """
    body  = request.get_json(force=True, silent=True) or {}
    score = int(body.get("score", 0))
    level = str(int(body.get("level", 1)))
    time_s = float(body.get("time", 0))
    nick   = str(body.get("nick", "")).strip()

    if level not in ("1", "2", "3"):
        level = "1"

    with _scores_lock:
        scores = _load_scores()
        if not nick:
            scores["guest_counter"] = scores.get("guest_counter", 0) + 1
            nick = f"Guest_{scores['guest_counter']}"

        entry = {
            "nick":  nick,
            "score": score,
            "time":  round(time_s, 1),
            "date":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        scores.setdefault(level, []).append(entry)
        _save_scores(scores)

        entries = sorted(scores[level], key=lambda x: (-x["score"], x.get("time", 99999)))

    return jsonify({
        "message": "Wynik zapisany!",
        "nick":    nick,
        "scores":  entries[:10],
    })


# ===========================================================================
# PUNKT WEJŚCIA
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PAC-MAN DQN – SERWER FLASK")
    print("=" * 60)
    print(f"  Frontend: http://localhost:5001/")
    print(f"  API docs: endpointy dostępne na http://localhost:5001/")
    print(f"  Model:    {_model_path}")
    print("=" * 60)

    app.run(
        host  = "0.0.0.0",
        port  = 5001,
        debug = False,    # debug=False bo używamy wątków
        threaded = True,  # obsługa wielu requestów jednocześnie
    )
