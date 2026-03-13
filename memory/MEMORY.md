# Project Memory – Pac-Man DQN

## Project Overview
**Title:** Implementacja algorytmu Deep Q-Learning w grze webowej (Pac-Man)
**Working dir:** /Users/lukaszkundzicz/PycharmProjects/Licencjat/
**Original game:** v2.py (pygame, 3 levels, 19×19 grid, cell_size=30)

## Architecture
- **backend/environment.py** – pure Python game logic (no pygame), OpenAI Gym-style
- **backend/model.py** – PyTorch DQNNetwork (15→256→128→64→4)
- **backend/replay_buffer.py** – Experience Replay (deque, capacity=10000)
- **backend/dqn_agent.py** – DQNAgent (ε-greedy, target network, Adam, SmoothL1)
- **backend/trainer.py** – background thread trainer, logs, matplotlib plot
- **backend/app.py** – Flask REST API (port 5000)
- **frontend/index.html** – game UI with play/train panels
- **frontend/style.css** – dark arcade theme
- **frontend/game.js** – canvas renderer, REST client, keyboard handler
- **requirements.txt** – flask, flask-cors, torch, numpy, matplotlib

## Game Constants (from v2.py)
- Grid: 19×19, CELL_SIZE=30
- Maze values: 0=dot, 1=wall, 2=power pellet, 3=empty
- GHOST_START = (8, 9), PACMAN_START = (14, 9)
- Actions: 0=right(dc+1), 1=left(dc-1), 2=up(dr-1), 3=down(dr+1)

## DQN State Vector (15 features)
0-1: ghost_row/col normalized
2-3: pacman_row/col normalized
4-5: signed delta (direction ghost→pacman)
6: manhattan distance normalized
7-10: walls around ghost (up/down/left/right)
11-14: walls around pacman (up/down/left/right)

## Rewards
+100: caught pacman, +1/-1: closer/farther, -5: wall hit, -0.1: per step, -50: ghost eaten (power mode)

## REST API Endpoints
- GET /game_state – full game state for rendering
- POST /step {pacman_action} – game step
- POST /reset {level} – reset game
- POST /train {episodes, level} – start background training
- POST /stop_training – stop training
- GET /training_status – training logs and stats
- POST /save_model / load_model
- GET /training_plot – matplotlib PNG

## How to Run
```bash
cd backend
pip install -r ../requirements.txt
python app.py
# Open http://localhost:5000
```

## User Preferences
- Language: Polish (academic context)
- PyTorch (not TensorFlow)
- Flask (not FastAPI)
