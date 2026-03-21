/**
 * =============================================================================
 * PAC-MAN DQN – LOGIKA FRONTENDU
 * =============================================================================
 *
 * Plik odpowiada za:
 *  1. Renderowanie gry na elemencie <canvas> (labirynt, Pac-Man, duszek, HUD).
 *  2. Obsługę klawiatury gracza (WASD / strzałki).
 *  3. Komunikację REST z backendem Flask:
 *       GET  /game_state       – pobranie stanu gry
 *       POST /step             – krok gry z akcją gracza
 *       POST /reset            – reset gry
 *       POST /train            – start treningu
 *       POST /stop_training    – zatrzymanie treningu
 *       GET  /training_status  – status treningu
 *       POST /save_model       – zapis modelu
 *       POST /load_model       – wczytanie modelu
 *       POST /toggle_online_learn – przełącznik uczenia online
 *  4. Wyświetlania panelu treningu: logi, statystyki, postęp.
 *  5. Wizualizacji Q-wartości sieci DQN.
 *
 * STAŁE GRY:
 *   CELL_SIZE = 30 px  (rozmiar komórki siatki 19×19)
 *   COLS = ROWS = 19
 *   Akcje: 0=prawo, 1=lewo, 2=góra, 3=dół
 * =============================================================================
 */

"use strict";

// ---------------------------------------------------------------------------
// STAŁE
// ---------------------------------------------------------------------------
const CELL_SIZE   = 30;
const ROWS        = 19;
const COLS        = 19;
const CANVAS_W    = CELL_SIZE * COLS;   // 570 px
const CANVAS_H    = CELL_SIZE * ROWS;   // 570 px
const API_BASE    = "";                  // serwer Flask na tym samym hoście:porcie
const TICK_MS     = 150;                 // ~6.7 FPS dopasowane do tempa v2.py (~12 FPS)
const TRAIN_POLL  = 1000;               // co ile ms odpytujemy status treningu

// Akcje: 0=prawo, 1=lewo, 2=góra, 3=dół
const ACTION_NAMES = ["→", "←", "↑", "↓"];
const ACTION_LABELS = ["Right", "Left", "Up", "Down"];

// ---------------------------------------------------------------------------
// STAN APLIKACJI
// ---------------------------------------------------------------------------
const App = {
  mode:             "play",       // "play" | "train"
  gameState:        null,         // ostatni stan gry z backendu
  playerAction:     -1,           // akcja gracza do wysłania przy następnym kroku
  currentDirection: 0,            // aktualny kierunek Pac-Mana
  animFrame:        0,            // licznik animacji Pac-Mana
  mouthOpen:        true,
  lastTickTime:     0,
  gameLoopId:       null,
  trainPollId:      null,
  livePollId:       null,         // podgląd na żywo treningu
  liveWatchEnabled: false,        // czy podgląd na żywo jest włączony
  isWaitingForStep: false,        // blokada współbieżnych requestów /step
  currentLevel:     1,
  qValues:          [0, 0, 0, 0], // ostatnie Q-wartości duszka
  keyState:         {},           // aktualnie wciśnięte klawisze
  detailsOn:        false,        // czy widać szczegółowy HUD (dystans, kroki, ε, online)
  analysisOn:       false,        // czy widać panele analityczne (Q-wartości, opis DQN)
  gameStartTime:    null,         // czas startu bieżącej rozgrywki (Date.now())
  scoreSubmitted:   false,        // czy wynik tego epizodu już wysłano
  lbLevel:          1,            // aktywna zakładka tablicy wyników
  gameReady:        false,        // true dopiero po zakończeniu resetu — blokuje fałszywy handleGameOver
  gameStarted:      false,        // true gdy gracz nacisnął P — od tego momentu liczy się czas
  countdown:        null,         // null lub liczba 3/2/1/0 podczas odliczania
  flashAlpha:       0,            // czerwony flash gdy duszek złapie Pac-Mana
  waitingForStart:  false,        // true gdy czekamy na naciśnięcie P (INSERT COIN)
  prevScore:        0,            // poprzedni wynik (do wykrywania zjedzenia kropki)
  countdownStartTime: null,       // czas startu bieżącej sekundy odliczania
};

// ---------------------------------------------------------------------------
// DŹWIĘKI RETRO (Web Audio API)
// ---------------------------------------------------------------------------
let _audioCtx = null;
function _getAudioCtx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}

function playSound(type) {
  try {
    const ctx = _getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    const t = ctx.currentTime;
    if (type === "dot") {
      osc.type = "square";
      osc.frequency.setValueAtTime(880, t);
      gain.gain.setValueAtTime(0.08, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
      osc.start(t); osc.stop(t + 0.08);
    } else if (type === "power") {
      osc.type = "square";
      osc.frequency.setValueAtTime(440, t);
      osc.frequency.linearRampToValueAtTime(880, t + 0.2);
      gain.gain.setValueAtTime(0.12, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
      osc.start(t); osc.stop(t + 0.25);
    } else if (type === "catch") {
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(600, t);
      osc.frequency.linearRampToValueAtTime(80, t + 0.5);
      gain.gain.setValueAtTime(0.15, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
      osc.start(t); osc.stop(t + 0.5);
    } else if (type === "win") {
      osc.type = "square";
      osc.frequency.setValueAtTime(523, t);
      osc.frequency.setValueAtTime(659, t + 0.1);
      osc.frequency.setValueAtTime(784, t + 0.2);
      osc.frequency.setValueAtTime(1047, t + 0.3);
      gain.gain.setValueAtTime(0.1, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
      osc.start(t); osc.stop(t + 0.5);
    } else if (type === "countdown") {
      osc.type = "square";
      osc.frequency.setValueAtTime(440, t);
      gain.gain.setValueAtTime(0.1, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
      osc.start(t); osc.stop(t + 0.12);
    } else if (type === "go") {
      osc.type = "square";
      osc.frequency.setValueAtTime(880, t);
      gain.gain.setValueAtTime(0.15, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
      osc.start(t); osc.stop(t + 0.2);
    }
  } catch(e) {}
}

// ---------------------------------------------------------------------------
// ELEMENTY DOM
// ---------------------------------------------------------------------------
let canvas, ctx;

function initDOM() {
  canvas = document.getElementById("gameCanvas");
  canvas.width  = CANVAS_W;
  canvas.height = CANVAS_H;
  ctx = canvas.getContext("2d");
}

// ---------------------------------------------------------------------------
// KOMUNIKACJA Z BACKENDEM (REST API)
// ---------------------------------------------------------------------------
async function apiGet(endpoint) {
  const res = await fetch(API_BASE + endpoint);
  if (!res.ok) throw new Error(`GET ${endpoint} → ${res.status}`);
  return res.json();
}

async function apiPost(endpoint, body = {}) {
  const res = await fetch(API_BASE + endpoint, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${endpoint} → ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// INICJALIZACJA GRY (start lub reset)
// ---------------------------------------------------------------------------
async function initGame(level = 1) {
  try {
    App.currentLevel     = level;
    App.playerAction     = -1;
    App.isWaitingForStep = false;
    App.gameReady        = false;
    App.scoreSubmitted   = false;

    const data = await apiPost("/reset", { level });
    App.gameStartTime = null;
    App.gameStarted   = false;
    App.countdown     = null;
    App.waitingForStart = false;
    App.gameReady     = true;
    updateGameState(data);
    updateHUD(data);
    render();
    App.waitingForStart = true;
    _startInsertCoinLoop();
  } catch (e) {
    setStatus("❌ Błąd połączenia z backendem: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// PĘTLA GRY (tryb PLAY)
// ---------------------------------------------------------------------------
function tickMsForLevel(level) {
  if (level === 2) return 110;
  if (level === 3) return 75;
  return 150; // level 1
}

function startGameLoop() {
  stopGameLoop();
  App.gameLoopId = setInterval(gameTick, tickMsForLevel(App.currentLevel));
}

function stopGameLoop() {
  if (App.gameLoopId !== null) {
    clearInterval(App.gameLoopId);
    App.gameLoopId = null;
  }
}

async function gameTick() {
  if (App.mode !== "play") return;
  if (!App.gameStarted) return;    // czekaj na naciśnięcie P
  if (App.isWaitingForStep) return;

  const gs = App.gameState;
  if (gs && gs.done) return;   // czekaj na reset gracza

  // Wyślij aktualną akcję gracza
  const action = App.playerAction >= 0 ? App.playerAction : App.currentDirection;

  App.isWaitingForStep = true;
  try {
    const data = await apiPost("/step", { pacman_action: action });
    updateGameState(data);
    updateHUD(data);

    // Zaktualizuj Q-wartości
    if (data.q_values) {
      App.qValues = data.q_values;
      updateQDisplay(data.q_values, data.ghost_action);
    }

    // Resetuj bufor akcji (zapamiętujemy ostatni kierunek)
    App.playerAction = -1;

    render();
  } catch (e) {
    console.warn("Błąd kroku:", e.message);
  } finally {
    App.isWaitingForStep = false;
  }
}

// ---------------------------------------------------------------------------
// AKTUALIZACJA STANU GRY
// ---------------------------------------------------------------------------
function updateGameState(data) {
  App.gameState = data;

  // Animacja gęby Pac-Mana
  App.animFrame++;
  App.mouthOpen = (App.animFrame % 4) < 2;

  // Dźwięki zjadania kropek (tylko podczas aktywnej gry gracza)
  if (App.gameStarted && data.score > (App.prevScore ?? 0)) {
    const diff = data.score - (App.prevScore ?? 0);
    playSound(diff >= 50 ? "power" : "dot");
  }
  App.prevScore = data.score ?? 0;

  // Reakcja na koniec gry (tylko gdy gra faktycznie wystartowała)
  if (data.done && App.gameReady) {
    handleGameOver(data);
  }
}

function handleGameOver(data) {
  const isWin = data.game_won;
  const msg   = isWin ? "PAC-MAN WYGRYWA!" : "DUSZEK ZŁAPAŁ PAC-MANA!";
  drawGameOverOverlay(msg, isWin);

  // Dźwięk końca gry
  playSound(isWin ? "win" : "catch");

  // Czerwony flash przy przegranej
  if (!isWin) {
    App.flashAlpha = 1.0;
    function fadeFlash() {
      App.flashAlpha -= 0.06;
      if (App.flashAlpha < 0) App.flashAlpha = 0;
      render();
      if (App.flashAlpha > 0) requestAnimationFrame(fadeFlash);
    }
    requestAnimationFrame(fadeFlash);
  }

  // Zapisz wynik na tablicy (wygrana i przegrana)
  if (!App.scoreSubmitted) {
    App.scoreSubmitted = true;
    const elapsed = App.gameStartTime ? (Date.now() - App.gameStartTime) / 1000 : 0;
    submitScore(data.score, App.currentLevel, elapsed);
  }
}

// ---------------------------------------------------------------------------
// PALETA KOLORÓW CANVASA (zależna od motywu)
// ---------------------------------------------------------------------------
function canvasPalette() {
  const light = document.documentElement.getAttribute("data-theme") === "light";
  return light ? {
    bg:        "#dde1f0",
    wallOuter: "#2244aa",
    wallInner: "#3355c8",
    dot:       "#4a5a9a",
    overlay:   "rgba(210,215,235,0.80)",
    light:     true,
  } : {
    bg:        "#000000",
    wallOuter: "#0022aa",
    wallInner: "#1133cc",
    dot:       "#ddddff",
    overlay:   "rgba(0,0,0,0.65)",
    light:     false,
  };
}

// ---------------------------------------------------------------------------
// RENDEROWANIE CANVASA
// ---------------------------------------------------------------------------
/** Rysuje pasek power-up na górze canvasu (kurczy się w miarę upływu czasu) */
function drawPowerBar(timer) {
  const MAX_TIMER = 50;
  const BAR_H     = 14;
  const BAR_Y     = 0;
  const ratio     = Math.max(0, timer / MAX_TIMER);
  const barW      = Math.round(CANVAS_W * ratio);

  // tło paska
  ctx.fillStyle = "#222222";
  ctx.fillRect(0, BAR_Y, CANVAS_W, BAR_H);

  // wypełnienie (kolor przechodzi z zielonego w czerwony gdy kończy się czas)
  const r = Math.round(255 * (1 - ratio));
  const g = Math.round(200 * ratio);
  ctx.fillStyle = `rgb(${r}, ${g}, 0)`;
  ctx.fillRect(0, BAR_Y, barW, BAR_H);

  // etykieta
  ctx.fillStyle = "#ffffff";
  ctx.font      = "bold 10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("POWER", CANVAS_W / 2, BAR_Y + BAR_H - 1);
  ctx.textAlign = "left";
}

function render() {
  if (!ctx || !App.gameState) return;
  const gs = App.gameState;
  const p  = canvasPalette();

  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  drawMaze(gs.maze, p);
  drawPacman(gs.pacman, App.mouthOpen, App.currentDirection);
  drawGhost(gs.ghost, gs.power_mode);
  if (gs.power_mode) drawPowerBar(gs.power_timer);

  if (App.flashAlpha > 0) {
    ctx.fillStyle = `rgba(255,0,0,${App.flashAlpha})`;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  }

  if (App.countdown !== null) {
    drawCountdownOverlay();
  } else if (gs.done) {
    drawGameOverOverlay(gs.game_won ? "PAC-MAN WYGRYWA!" : "GAME OVER",
                        gs.game_won, p);
  }
}

/** Rysuje labirynt (ściany, kropki, power pellety) */
function drawMaze(maze, p) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = c * CELL_SIZE;
      const y = r * CELL_SIZE;
      const cell = maze[r][c];

      if (cell === 1) {
        // Ściana – niebieski gradient z cienką krawędzią
        ctx.fillStyle = p.wallOuter;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.fillStyle = p.wallInner;
        ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

      } else if (cell === 0) {
        // Kropka
        ctx.fillStyle = p.dot;
        ctx.beginPath();
        ctx.arc(x + CELL_SIZE / 2, y + CELL_SIZE / 2, 3, 0, Math.PI * 2);
        ctx.fill();

      } else if (cell === 2) {
        // Power pellet – pulsujący złoty
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 200);
        const r2 = 6 + pulse * 3;
        const pcx = x + CELL_SIZE / 2;
        const pcy = y + CELL_SIZE / 2;
        ctx.fillStyle = `rgba(255, 215, 0, ${0.7 + pulse * 0.3})`;
        ctx.shadowColor = "#ffd700";
        ctx.shadowBlur  = p.light ? 0 : 8;
        ctx.beginPath();
        ctx.arc(pcx, pcy, r2, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        // Ciemny kontur w trybie jasnym
        if (p.light) {
          ctx.strokeStyle = `rgba(160, 100, 0, ${0.5 + pulse * 0.4})`;
          ctx.lineWidth   = 1.5;
          ctx.stroke();
        }
      }
      // cell === 3 → wolne pole (dom duszka, tunel) – nie rysujemy nic
    }
  }
}

/** Rysuje Pac-Mana z animowaną gębą i okiem */
function drawPacman(pos, mouthOpen, direction) {
  const cx = pos.col * CELL_SIZE + CELL_SIZE / 2;
  const cy = pos.row * CELL_SIZE + CELL_SIZE / 2;
  const radius = CELL_SIZE / 2 - 2;
  const light  = canvasPalette().light;

  // Kąt obrotu w zależności od kierunku
  const startAngles = [0, Math.PI, -Math.PI / 2, Math.PI / 2]; // prawo, lewo, góra, dół
  const startAngle  = startAngles[direction] || 0;
  const mouthAngle  = mouthOpen ? Math.PI / 4 : Math.PI / 12;

  ctx.fillStyle   = "#ffee00";
  ctx.shadowColor = "#ffee00";
  ctx.shadowBlur  = light ? 0 : 8;

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, radius,
    startAngle + mouthAngle,
    startAngle + 2 * Math.PI - mouthAngle);
  ctx.closePath();
  ctx.fill();

  // Kontur w trybie jasnym
  if (light) {
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Oko — 60° od kierunku ruchu, zawsze w górnej połowie ekranu
  const opt1 = startAngle - Math.PI / 3;
  const opt2 = startAngle + Math.PI / 3;
  const eyeAngle = Math.sin(opt1) <= Math.sin(opt2) ? opt1 : opt2;
  const eyeDist  = radius * 0.52;
  const ex = cx + Math.cos(eyeAngle) * eyeDist;
  const ey = cy + Math.sin(eyeAngle) * eyeDist;

  // Oko — zwykła kropka
  ctx.fillStyle = "#111111";
  ctx.beginPath();
  ctx.arc(ex, ey, 2.5, 0, Math.PI * 2);
  ctx.fill();
}

/** Rysuje duszka (kółko z oczami) */
function drawGhost(pos, powerMode) {
  const cx     = pos.col * CELL_SIZE + CELL_SIZE / 2;
  const cy     = pos.row * CELL_SIZE + CELL_SIZE / 2;
  const radius = CELL_SIZE / 2 - 2;

  const color = powerMode ? "#4444ff" : "#ff2244";
  const glow  = powerMode ? "#4444ff" : "#ff2244";

  ctx.shadowColor = glow;
  ctx.shadowBlur  = 10;

  // Ciało duszka – koło + prostokąt z ząbkami na dole
  ctx.fillStyle = color;
  ctx.beginPath();

  // Górna półkula
  ctx.arc(cx, cy, radius, Math.PI, 0);

  // Dół z ząbkami (3 trójkąty)
  const bottom = cy + radius;
  const top    = cy;
  const left   = cx - radius;
  const right  = cx + radius;
  const tw     = (right - left) / 3;

  ctx.lineTo(right, bottom);
  ctx.lineTo(right - tw * 0.5, top + radius * 0.4);
  ctx.lineTo(right - tw, bottom);
  ctx.lineTo(right - tw * 1.5, top + radius * 0.4);
  ctx.lineTo(left + tw, bottom);
  ctx.lineTo(left + tw * 0.5, top + radius * 0.4);
  ctx.lineTo(left, bottom);
  ctx.closePath();
  ctx.fill();

  ctx.shadowBlur = 0;

  // Oczy (tylko gdy nie power mode)
  if (!powerMode) {
    const eyeR = radius / 4;
    const eyeOffX = radius / 3;
    const eyeOffY = radius / 4;

    // Białe oczy
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(cx - eyeOffX, cy - eyeOffY, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + eyeOffX, cy - eyeOffY, eyeR, 0, Math.PI * 2);
    ctx.fill();

    // Niebieskie źrenice
    ctx.fillStyle = "#0022ff";
    ctx.beginPath();
    ctx.arc(cx - eyeOffX + 2, cy - eyeOffY + 2, eyeR / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + eyeOffX + 2, cy - eyeOffY + 2, eyeR / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    // Tryb mocy – przestraszone oczy (x_x)
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth   = 2;
    const ex = radius / 3;
    const ey = radius / 4;
    // Lewe oko ×
    ctx.beginPath();
    ctx.moveTo(cx - ex - 4, cy - ey - 4);
    ctx.lineTo(cx - ex + 4, cy - ey + 4);
    ctx.moveTo(cx - ex + 4, cy - ey - 4);
    ctx.lineTo(cx - ex - 4, cy - ey + 4);
    ctx.stroke();
    // Prawe oko ×
    ctx.beginPath();
    ctx.moveTo(cx + ex - 4, cy - ey - 4);
    ctx.lineTo(cx + ex + 4, cy - ey + 4);
    ctx.moveTo(cx + ex + 4, cy - ey - 4);
    ctx.lineTo(cx + ex - 4, cy - ey + 4);
    ctx.stroke();
    ctx.lineWidth = 1;
  }
}

/** Nakładka INSERT COIN (miga co 600ms) */
function drawWaitForStartOverlay() {
  if (!ctx) return;
  const p = canvasPalette();
  ctx.fillStyle = p.overlay;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";

  // INSERT COIN — blink every 600ms
  const blink = Math.floor(Date.now() / 600) % 2 === 0;
  if (blink) {
    ctx.font      = "bold 32px 'Courier New', monospace";
    ctx.fillStyle = p.light ? "#e06500" : "#ffee00";
    ctx.shadowColor = p.light ? "#e06500" : "#ffee00";
    ctx.shadowBlur  = p.light ? 0 : 16;
    ctx.fillText("INSERT COIN", CANVAS_W / 2, CANVAS_H / 2 - 40);
    ctx.shadowBlur = 0;
  }

  ctx.font      = "16px 'Courier New', monospace";
  ctx.fillStyle = p.light ? "#1a2252" : "#aaaacc";
  ctx.fillText("Naciśnij [P] lub kliknij START", CANVAS_W / 2, CANVAS_H / 2 + 5);

  const nick = (document.getElementById("playerNick")?.value ?? "").trim();
  ctx.font      = "14px 'Courier New', monospace";
  ctx.fillStyle = nick ? (p.light ? "#006b3c" : "#00dd74") : (p.light ? "#3a468a" : "#555878");
  ctx.fillText(nick ? `GRACZ: ${nick}` : "Wpisz nick w panelu →", CANVAS_W / 2, CANVAS_H / 2 + 40);
}

let _insertCoinRafId = null;
function _startInsertCoinLoop() {
  if (_insertCoinRafId) cancelAnimationFrame(_insertCoinRafId);
  function loop() {
    if (!App.waitingForStart) { _insertCoinRafId = null; return; }
    if (!App.gameStarted && App.countdown === null) drawWaitForStartOverlay();
    _insertCoinRafId = requestAnimationFrame(loop);
  }
  _insertCoinRafId = requestAnimationFrame(loop);
}

function startGame() {
  if (!App.gameReady || App.gameStarted) return;
  if (App.mode !== "play") return;
  const gs = App.gameState;
  if (gs && gs.done) return;
  if (App.countdown !== null) return;
  App.waitingForStart = false;
  App.countdown = 3;
  App.countdownStartTime = Date.now();
  render();
  function _runCountdownRender() {
    if (App.countdown === null) return;
    drawCountdownOverlay();
    requestAnimationFrame(_runCountdownRender);
  }
  _runCountdownRender();
  const tick = setInterval(() => {
    playSound(App.countdown > 0 ? "countdown" : "go");
    App.countdown--;
    App.countdownStartTime = Date.now();
    if (App.countdown <= 0) {
      clearInterval(tick);
      App.countdown     = null;
      App.gameStarted   = true;
      App.gameStartTime = Date.now();
    }
    render();
  }, 1000);
}

function drawCountdownOverlay() {
  const p = canvasPalette();
  ctx.fillStyle = p.overlay;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  const elapsed = App.countdownStartTime ? (Date.now() - App.countdownStartTime) / 1000 : 0;
  // scale: starts big (1.6), shrinks to 1.0
  const scale = 1.6 - 0.6 * Math.min(elapsed, 1.0);
  const label = App.countdown > 0 ? String(App.countdown) : "START";
  const baseSize = App.countdown > 0 ? 120 : 72;
  const size = Math.round(baseSize * scale);

  ctx.save();
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.font        = `bold ${size}px 'Courier New', monospace`;
  ctx.fillStyle   = App.countdown > 0 ? "#f0d400" : "#00dd74";
  ctx.shadowColor = App.countdown > 0 ? "#f0d400" : "#00dd74";
  ctx.shadowBlur  = 30;
  // fade in: alpha goes from 0.3 to 1.0 quickly
  const alpha = Math.min(1, elapsed * 5);
  ctx.globalAlpha = alpha;
  ctx.fillText(label, CANVAS_W / 2, CANVAS_H / 2);
  ctx.restore();
  ctx.shadowBlur  = 0;
}

/** Nakładka GAME OVER / WIN na canvasie */
function drawGameOverOverlay(message, isWin, p) {
  p = p || canvasPalette();
  ctx.fillStyle = p.overlay;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  ctx.textAlign  = "center";
  ctx.textBaseline = "middle";

  ctx.font      = "bold 36px 'Courier New', monospace";
  ctx.fillStyle = isWin ? "#ffee00" : "#ff2244";
  ctx.shadowColor = isWin ? "#ffee00" : "#ff2244";
  ctx.shadowBlur  = 16;
  ctx.fillText(message, CANVAS_W / 2, CANVAS_H / 2 - 20);

  ctx.shadowBlur  = 0;
  const score   = App.gameState?.score ?? 0;
  const elapsed = App.gameStartTime ? (Date.now() - App.gameStartTime) / 1000 : 0;
  const timeStr = formatTime(elapsed);
  ctx.font      = "bold 16px 'Courier New', monospace";
  ctx.fillStyle = p.light ? "#1a2252" : "#f0d400";
  ctx.fillText(`Wynik: ${score}   Czas: ${timeStr}`, CANVAS_W / 2, CANVAS_H / 2 + 18);

  ctx.font      = "16px 'Courier New', monospace";
  ctx.fillStyle = p.light ? "#445" : "#aaaacc";
  ctx.fillText("Naciśnij [R] lub kliknij Reset", CANVAS_W / 2, CANVAS_H / 2 + 44);
}

// ---------------------------------------------------------------------------
// AKTUALIZACJA HUD
// ---------------------------------------------------------------------------
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function updateHUD(data) {
  safeText("hudScore",     data.score ?? 0);
  safeText("hudManhattan", data.manhattan ?? "?");
  safeText("hudLevel",     data.level ?? 1);
  safeText("hudEpsilon",   data.agent_info?.epsilon?.toFixed(3) ?? "?");
  safeText("hudSteps",     data.step_count ?? 0);

  // Czas trwania rozgrywki
  if (App.gameStarted && App.gameStartTime) {
    const elapsed = (Date.now() - App.gameStartTime) / 1000;
    safeText("hudTime", formatTime(elapsed));
  }

  // Kolor online learning
  const olBadge = document.getElementById("onlineLearningBadge");
  if (olBadge) {
    const isOn = data.online_learning_enabled;
    olBadge.textContent = isOn ? "ON" : "OFF";
    olBadge.className   = "badge " + (isOn ? "badge-training" : "badge-idle");
  }
}

// ---------------------------------------------------------------------------
// PANEL Q-WARTOŚCI
// ---------------------------------------------------------------------------
function updateQDisplay(qVals, bestAction) {
  App.qValues = qVals || [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    const cell = document.getElementById(`qCell${i}`);
    if (!cell) continue;
    // obsługuje zarówno starą klasę (.q-cell-value) jak i nową (.q-val)
    const valEl = cell.querySelector(".q-val") || cell.querySelector(".q-cell-value");
    if (valEl) valEl.textContent = App.qValues[i]?.toFixed(3) ?? "0.000";
    cell.classList.toggle("best", i === bestAction);
  }
}

// ---------------------------------------------------------------------------
// OBSŁUGA KLAWIATURY
// ---------------------------------------------------------------------------
const KEY_TO_ACTION = {
  ArrowRight: 0, KeyD: 0,
  ArrowLeft:  1, KeyA: 1,
  ArrowUp:    2, KeyW: 2,
  ArrowDown:  3, KeyS: 3,
};

function setupKeyboard() {
  document.addEventListener("keydown", (e) => {
    // Nie przechwytuj klawiszy gdy focus jest na polu tekstowym
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    if (e.code in KEY_TO_ACTION) {
      e.preventDefault();
      const action = KEY_TO_ACTION[e.code];
      App.playerAction  = action;
      App.currentDirection = action;
      highlightKey(action);
    }

    // R – reset
    if (e.code === "KeyR") {
      initGame(App.currentLevel);
    }

    // P – start gry (uruchomienie zegara)
    if (e.code === "KeyP") {
      startGame();
    }
  });

  document.addEventListener("keyup", (e) => {
    clearKeyHighlights();
  });
}

function highlightKey(action) {
  // Highlight klawiszy w panelu sterowania
  ["W","A","S","D","UP","DOWN","LEFT","RIGHT"].forEach(k => {
    const el = document.getElementById("key" + k);
    if (el) el.classList.remove("active");
  });
  const keyMap = { 0: ["D","RIGHT"], 1: ["A","LEFT"], 2: ["W","UP"], 3: ["S","DOWN"] };
  (keyMap[action] || []).forEach(k => {
    const el = document.getElementById("key" + k);
    if (el) el.classList.add("active");
  });
}

function clearKeyHighlights() {
  ["W","A","S","D","UP","DOWN","LEFT","RIGHT"].forEach(k => {
    const el = document.getElementById("key" + k);
    if (el) el.classList.remove("active");
  });
}

// ---------------------------------------------------------------------------
// OBSŁUGA TRYBU (PLAY / TRAIN)
// ---------------------------------------------------------------------------
function setMode(mode) {
  App.mode = mode;

  document.getElementById("btnModePlay").classList.toggle("active",  mode === "play");
  document.getElementById("btnModeTrain").classList.toggle("active", mode === "train");

  // WAŻNE: panele to elementy blokowe (.panel), więc display:"block", nie "flex"
  const playPanel  = document.getElementById("playPanel");
  const qPanel     = document.getElementById("qPanel");
  const trainPanel = document.getElementById("trainPanel");

  const showPlay  = mode === "play";
  const showTrain = mode === "train";

  if (playPanel)  playPanel.style.display  = showPlay  ? "block" : "none";
  if (qPanel)     qPanel.style.display     = (showPlay && App.analysisOn) ? "block" : "none";
  if (trainPanel) trainPanel.style.display = showTrain ? "block" : "none";

  const leaderboardPanel = document.getElementById("leaderboardPanel");
  if (leaderboardPanel) leaderboardPanel.style.display = showPlay ? "block" : "none";

  const hud = document.querySelector(".hud");
  if (hud) hud.style.display = showPlay ? "flex" : "none";

  if (mode === "play") {
    stopLivePolling();
    startGameLoop();
    stopTrainPolling();
    // Wróć do canvasa gry (pobierz aktualny stan)
    apiGet("/game_state").then(d => { updateGameState(d); render(); }).catch(() => {});
  } else {
    stopGameLoop();
    startTrainPolling();
    // Jeśli watch był włączony, uruchom polling podglądu
    if (App.liveWatchEnabled) startLivePolling();
    setStatus("");
  }
}

// ---------------------------------------------------------------------------
// POLLING STATUSU TRENINGU
// ---------------------------------------------------------------------------
function startTrainPolling() {
  stopTrainPolling();
  fetchTrainingStatus();   // natychmiastowe pierwsze zapytanie
  App.trainPollId = setInterval(fetchTrainingStatus, TRAIN_POLL);
}

function stopTrainPolling() {
  if (App.trainPollId !== null) {
    clearInterval(App.trainPollId);
    App.trainPollId = null;
  }
}

// ---------------------------------------------------------------------------
// PODGLĄD NA ŻYWO – polling stanu środowiska treningowego
// ---------------------------------------------------------------------------
function startLivePolling() {
  stopLivePolling();
  App.livePollId = setInterval(fetchLiveState, 120);  // ~8 fps
}

function stopLivePolling() {
  if (App.livePollId !== null) {
    clearInterval(App.livePollId);
    App.livePollId = null;
  }
}

async function fetchLiveState() {
  try {
    const data = await apiGet("/training_live_state");

    if (!data.available) {
      // Trening nie jest aktywny lub jeszcze się nie zaczął
      updateLiveOverlay(false);
      return;
    }

    // Zaktualizuj canvas stanem treningowym
    App.animFrame++;
    App.mouthOpen = (App.animFrame % 6) < 3;
    renderTrainingState(data);
    updateLiveOverlay(true, data);

  } catch (e) {
    // Cichy błąd – nie przeszkadzaj w treningu
  }
}

/** Renderuje stan środowiska treningowego (duszek + pacman autopilot) na canvasie */
function renderTrainingState(gs) {
  if (!ctx) return;
  const p = canvasPalette();

  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  drawMaze(gs.maze, p);
  drawPacman(gs.pacman, App.mouthOpen, 0);
  drawGhost(gs.ghost, gs.power_mode || false);

  // Pasek informacyjny na dole canvasu
  ctx.save();
  ctx.fillStyle = p.light ? "rgba(200,210,230,0.88)" : "rgba(0,0,0,0.72)";
  ctx.fillRect(0, CANVAS_H - 20, CANVAS_W, 20);
  ctx.font = "11px 'Courier New'";
  ctx.fillStyle = p.light ? "#006b3c" : "#00ff88";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(`▶  EP ${gs.current_episode}    STEP ${gs.step}`, 10, CANVAS_H - 10);
  ctx.restore();
}

/** Aktualizuje etykietę statusu podglądu w panelu treningu */
function updateLiveOverlay(active, gs) {
  const el = document.getElementById("liveWatchStatus");
  if (!el) return;
  if (active && gs) {
    el.textContent = `● EP ${gs.current_episode}  s=${gs.step}  d=${gs.manhattan ?? '?'}`;
    el.style.color  = "var(--green)";
  } else {
    el.textContent  = "Trening nie jest aktywny";
    el.style.color  = "var(--text-dim)";
  }
}

/** Obsługa przełącznika podglądu na żywo (wywołanie z HTML) */
function onToggleLiveWatch(checkbox) {
  App.liveWatchEnabled = checkbox.checked;
  if (App.liveWatchEnabled && App.mode === "train") {
    startLivePolling();
    setStatus("");
  } else {
    stopLivePolling();
    updateLiveOverlay(false);
    if (App.mode === "train") {
      // Przywróć pusty canvas z napisem
      const p = canvasPalette();
      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = p.dot;
      ctx.font = "14px 'Courier New'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Włącz podgląd na żywo lub wróć do trybu gry", CANVAS_W/2, CANVAS_H/2);
      ctx.textAlign = "left";
    }
    setStatus("Podgląd wyłączony.");
  }
}

async function fetchTrainingStatus() {
  try {
    const data = await apiGet("/training_status");
    updateTrainingPanel(data);
  } catch (e) {
    console.warn("Błąd polling status:", e.message);
  }
}

function updateTrainingPanel(data) {
  // Pasek postępu
  const pct = data.progress_pct ?? 0;
  const bar = document.getElementById("trainProgressBar");
  if (bar) bar.style.width = pct + "%";

  safeText("trainEpNum",    `${data.current_episode ?? 0} / ${data.total_episodes ?? 0}`);
  safeText("trainAvgR",     data.avg_reward_100?.toFixed(2) ?? "-");
  safeText("trainBestR",    data.best_reward?.toFixed(2) ?? "-");
  safeText("trainEpsilon",  data.agent_info?.epsilon?.toFixed(4) ?? "-");
  safeText("trainBufSize",  data.agent_info?.buffer_size ?? 0);
  safeText("trainLoss",     data.agent_info?.avg_loss?.toFixed(5) ?? "-");
  safeText("trainSteps",    data.agent_info?.training_steps ?? 0);
  safeText("trainProgress", `${pct.toFixed(1)}%`);

  // Badge statusu
  const badge = document.getElementById("trainStatusBadge");
  if (badge) {
    if (data.is_training) {
      badge.textContent = "TRAINING";
      badge.className   = "badge badge-training";
    } else if (data.current_episode > 0) {
      badge.textContent = "DONE";
      badge.className   = "badge badge-done";
    } else {
      badge.textContent = "IDLE";
      badge.className   = "badge badge-idle";
    }
  }

  // Przyciski
  const btnStart = document.getElementById("btnStartTrain");
  const btnStop  = document.getElementById("btnStopTrain");
  if (btnStart) btnStart.disabled = data.is_training;
  if (btnStop)  btnStop.disabled  = !data.is_training;

  // Log epizodów (ostatnie wpisy na górze)
  appendTrainLogs(data.recent_logs || []);
}

// Śledź ostatni wyświetlony epizod, by nie duplikować wpisów
let lastLoggedEpisode = 0;

function appendTrainLogs(logs) {
  const container = document.getElementById("trainLog");
  if (!container) return;

  // Tylko nowe wpisy
  const newLogs = logs.filter(l => l.episode > lastLoggedEpisode);
  if (newLogs.length === 0) return;

  newLogs.forEach(entry => {
    const div = document.createElement("div");
    div.className = "log-entry";
    const caughtHtml = entry.caught
      ? `<span class="caught">✓ZŁAPAŁ</span>`
      : `<span class="missed">✗chybił</span>`;
    div.innerHTML =
      `<span class="ep-num">Ep ${entry.episode}</span>` +
      ` R=<span class="reward-v">${entry.reward}</span>` +
      ` ε=<span class="eps-v">${entry.epsilon}</span>` +
      ` s=${entry.steps}` +
      ` ${caughtHtml}`;
    container.prepend(div);   // najnowsze na górze
  });

  lastLoggedEpisode = Math.max(...logs.map(l => l.episode), lastLoggedEpisode);

  // Ogranicz do 200 wpisów w DOM
  while (container.children.length > 200) {
    container.removeChild(container.lastChild);
  }
}

// ---------------------------------------------------------------------------
// OBSŁUGA PRZYCISKÓW UI
// ---------------------------------------------------------------------------
async function onStartTraining() {
  const episodes = parseInt(document.getElementById("trainEpisodes")?.value ?? "500");
  const level    = parseInt(document.getElementById("trainLevel")?.value    ?? "1");

  if (isNaN(episodes) || episodes < 1) {
    setStatus("Podaj prawidłową liczbę epizodów.");
    return;
  }

  lastLoggedEpisode = 0;
  const logEl = document.getElementById("trainLog");
  if (logEl) logEl.innerHTML = "";

  try {
    setStatus("");
    const data = await apiPost("/train", { episodes, level });
    setStatus("");
    startTrainPolling();
  } catch (e) {
    setStatus("❌ Błąd uruchamiania treningu: " + e.message);
  }
}

async function onStopTraining() {
  try {
    await apiPost("/stop_training");
    setStatus("⏹ Trening zostanie zatrzymany.");
  } catch (e) {
    setStatus("❌ Błąd zatrzymywania: " + e.message);
  }
}

async function onSaveModel() {
  try {
    const data = await apiPost("/save_model");
    setStatus("💾 " + data.message);
  } catch (e) {
    setStatus("❌ Błąd zapisu: " + e.message);
  }
}

async function onLoadModel() {
  try {
    const data = await apiPost("/load_model");
    setStatus("📂 " + data.message);
    if (data.agent_info) {
      safeText("hudEpsilon", data.agent_info.epsilon?.toFixed(3) ?? "?");
    }
  } catch (e) {
    setStatus("❌ Błąd wczytania: " + e.message);
  }
}

async function onToggleOnlineLearning(checkbox) {
  try {
    const data = await apiPost("/toggle_online_learn", { enabled: checkbox.checked });
    const isOn = data.online_learning_enabled;
    const olBadge = document.getElementById("onlineLearningBadge");
    if (olBadge) {
      olBadge.textContent = isOn ? "ON" : "OFF";
      olBadge.className   = "badge " + (isOn ? "badge-training" : "badge-idle");
    }
  } catch (e) {
    setStatus("❌ Błąd toggle: " + e.message);
  }
}

function onLevelSelect(level) {
  App.currentLevel = level;
  document.querySelectorAll(".level-btn").forEach((btn, i) => {
    btn.classList.toggle("active", i + 1 === level);
  });
  initGame(level);
}

async function onOpenPlot() {
  window.open(API_BASE + "/training_plot", "_blank");
}

// ---------------------------------------------------------------------------
// NARZĘDZIA
// ---------------------------------------------------------------------------
function safeText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// ---------------------------------------------------------------------------
// WIDOK — prosty / szczegółowy
// ---------------------------------------------------------------------------
function initViewMode() {
  setDetailsView(localStorage.getItem("details") === "1");
  setAnalysisView(localStorage.getItem("analysis") === "1");
}

function setDetailsView(on) {
  App.detailsOn = on;
  localStorage.setItem("details", on ? "1" : "0");
  const toggleEl = document.getElementById("btnToggleDetails");
  if (toggleEl) toggleEl.checked = on;
  document.querySelectorAll(".hud-detail").forEach(el => {
    el.style.display = on ? "flex" : "none";
  });
}

function setAnalysisView(on) {
  App.analysisOn = on;
  localStorage.setItem("analysis", on ? "1" : "0");
  document.getElementById("btnToggleAnalysis")?.classList.toggle("active", on);
  const qPanel    = document.getElementById("qPanel");
  const dqnPanel  = document.getElementById("dqnPanel");
  const guidePanel = document.getElementById("guidePanel");
  if (dqnPanel)  dqnPanel.style.display  = on ? "block" : "none";
  if (guidePanel) guidePanel.style.display = on ? "block" : "none";
  // qPanel widoczny tylko w trybie play + analiza włączona
  if (qPanel) qPanel.style.display = (on && App.mode === "play") ? "block" : "none";
}

// ---------------------------------------------------------------------------
// MOTYW — jasny / ciemny
// ---------------------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  const toggle = document.getElementById("toggleTheme");
  if (toggle) toggle.checked = (saved === "light");
}

function onToggleTheme(checkbox) {
  const theme = checkbox.checked ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

// ---------------------------------------------------------------------------
// INSTRUKCJA OBSŁUGI — TOGGLE (zwijanie / rozwijanie panelu)
// ---------------------------------------------------------------------------
function toggleGuide() {
  const panel = document.getElementById("guidePanel");
  const icon  = document.getElementById("guideToggleIcon");
  if (!panel) return;
  const open = panel.classList.toggle("open");
  if (icon) icon.textContent = open ? "▲" : "▼";
}

function setStatus(msg) {
  const el = document.getElementById("statusBar");
  if (el) el.textContent = msg;
}

// ---------------------------------------------------------------------------
// TABLICA WYNIKÓW
// ---------------------------------------------------------------------------
async function submitScore(score, level, timeSec) {
  const nick = (document.getElementById("playerNick")?.value ?? "").trim();
  try {
    const data = await apiPost("/scores", { nick, score, level, time: timeSec });
    App.lbLevel = level;
    updateLbTabs(level);
    await fetchLeaderboard(level);   // odśwież tablicę + placeholder
  } catch (e) {
    console.warn("Błąd zapisu wyniku:", e.message);
  }
}

async function updateHeaderHighscore() {
  try {
    let best = 0;
    for (const lvl of [1, 2, 3]) {
      const data = await apiGet(`/scores?level=${lvl}`);
      if (data.scores && data.scores.length > 0) {
        best = Math.max(best, data.scores[0].score ?? 0);
      }
    }
    const el = document.getElementById("headerHighscore");
    if (el) el.textContent = best > 0 ? best : "––––";
  } catch (e) {}
}

async function fetchLeaderboard(level) {
  try {
    const data = await apiGet(`/scores?level=${level}`);
    renderLeaderboard(level, data.scores);
    updateGuestPlaceholder(data.guest_counter ?? 0);
    updateHeaderHighscore();
  } catch (e) {
    console.warn("Błąd pobierania tablicy:", e.message);
  }
}

function updateGuestPlaceholder(counter) {
  const input = document.getElementById("playerNick");
  if (input) input.placeholder = `Guest_${counter + 1}`;
}

function renderLeaderboard(level, scores) {
  const tbody = document.getElementById("lbBody");
  if (!tbody) return;

  if (!scores || scores.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-dim);text-align:center;padding:10px;">
      Brak wyników dla poziomu ${level}</td></tr>`;
    return;
  }

  // Nick aktualnego gracza (do podświetlenia)
  const myNick = (document.getElementById("playerNick")?.value ?? "").trim();

  const medalClass = ["lb-gold", "lb-silver", "lb-bronze"];

  tbody.innerHTML = scores.map((entry, i) => {
    const timeStr = (entry.time != null && entry.time > 0) ? `${entry.time.toFixed(1)}s` : "–";
    const isMe    = myNick && entry.nick === myNick;
    const rankCls = medalClass[i] ?? "lb-normal";
    const cls     = isMe ? `${rankCls} lb-highlight` : rankCls;
    const rawDate = entry.date ?? "";
    const dateShort = rawDate.length >= 10
      ? rawDate.slice(8,10) + "." + rawDate.slice(5,7) + "." + rawDate.slice(2,4) + (rawDate.length >= 16 ? " " + rawDate.slice(11,16) : "")
      : rawDate;
    return `<tr class="${cls}">
      <td>${i === 0 ? '<span style="position:relative;left:-3px;">🏆</span>' : i + 1}</td>
      <td>${escHtml(entry.nick)}</td>
      <td>${entry.score}</td>
      <td>${timeStr}</td>
      <td style="color:var(--text-mid);">${dateShort}</td>
    </tr>`;
  }).join("");
}

function updateLbTabs(level) {
  [1, 2, 3].forEach(l => {
    const btn = document.getElementById(`lbTab${l}`);
    if (btn) btn.classList.toggle("active", l === level);
  });
}

function onLbTab(level) {
  App.lbLevel = level;
  updateLbTabs(level);
  fetchLeaderboard(level);
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// START
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  initViewMode();
  initDOM();
  setupKeyboard();
  setMode("play");

  // Rysuj pusty ekran podczas łączenia
  ctx.fillStyle = canvasPalette().bg;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  await initGame(1);
  startGameLoop();
  fetchLeaderboard(1);   // załaduj tablicę wyników przy starcie
});
