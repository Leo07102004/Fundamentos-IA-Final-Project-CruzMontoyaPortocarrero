"""
Agente Montoya – Connect-4
===========================
Basado íntegramente en los conceptos del curso:

  Clase 11 (RL Basics):
    - GPI con q-values estimados por Monte Carlo (FVMC-style).
    - Utility estimation desde trials.

  Clase 12 (Competitive MDPs):
    - Alternating Markov Game de suma cero (r ∈ {-1, 0, 1}, γ=1).
    - UCB exploration:  u(s,a) = q̂(s,a) + sqrt(log(ΣN[s,a']) / N[s,a])
      (slide 22, clase 12) — usada tanto en el tree policy como en
      la selección de acción final.

  Clase 13 (Online Policy Improvement / MCTS):
    - Trial-Based Online Policy Improvement: en cada estado del outer
      trial se lanza un sub-proceso MCTS limitado en tiempo.
    - MCTS con sus 4 fases: Selection → Expansion → Simulation
      → Backpropagation (slide 25, clase 13).
    - Tree policy = UCB; Default policy = heurística posicional
      (equivale a la "default/rollout policy" del curso).
    - q̂' del sub-proceso NO contamina el q̂ global (slide 13, clase 13).
    - Warmup: se inicializa la raíz vacía antes de la partida,
      aprovechando el budget de tiempo libre para poblar el árbol
      desde el estado inicial (Exploring Starts desde s0, clase 11).

Diferencias clave respecto a Cruz y Portocarrero:
  1. Bitboards (representación interna): 8-17x más iteraciones en el
     mismo tiempo — no es un concepto nuevo, es optimización de impl.
  2. Default policy guiada por score posicional + amenazas de 3-en-raya,
     en vez de rollout completamente aleatorio (Cruz/Portocarrero).
  3. UCB de clase 12 para la política de exploración en árbol, con C
     configurable que controla el trade-off exploit/explore.
  4. Warm-up en mount(): Exploring Starts desde posición vacía consume
     el 88% del budget para pre-poblar el árbol antes del primer movimiento.

Variable numérica de configuración: UCB_C (parámetro C de UCB).
  - C bajo  → más explotador (converge rápido, puede perder diversidad).
  - C alto  → más explorador (cubre más ramas, puede ser superficial).
  Se puede activar/desactivar el warmup para mostrar su impacto.

Versiones para análisis de desempeño:
  MontoyaAgent(ucb_c=1.41, warmup=True)   ← versión completa (V-FULL)
  MontoyaAgent(ucb_c=1.41, warmup=False)  ← sin warmup (V-NOWARM)
  MontoyaAgent(ucb_c=0.0,  warmup=True)   ← puramente explotador (V-GREEDY)
"""

import math
import time
import random
import numpy as np
from connect4.policy import Policy

# ── Hiperparámetros ──────────────────────────────────────────────────────
DEFAULT_BUDGET = 28.0
DEFAULT_UCB_C  = 1.41          # C del UCB (clase 12, slide 22)
ROWS, COLS     = 6, 7
# Orden de columnas de más central a más lateral (heurística de apertura)
COL_ORDER      = [3, 2, 4, 1, 5, 0, 6]

# Tabla de pesos posicionales por columna×fila (mayor valor = mejor casilla)
# Justificación: el centro tiene más posibilidades de completar 4 en raya.
_POS_COL = [
    [3, 4, 5, 5, 4, 3],   # col 0
    [4, 6, 7, 7, 6, 4],   # col 1
    [5, 8, 9, 9, 8, 5],   # col 2
    [7,10,11,11,10, 7],   # col 3 (centro)
    [5, 8, 9, 9, 8, 5],   # col 4
    [4, 6, 7, 7, 6, 4],   # col 5
    [3, 4, 5, 5, 4, 3],   # col 6
]
_BIT_WEIGHT = [0] * 49
for _c in range(7):
    for _r in range(6):
        _BIT_WEIGHT[_c * 7 + _r] = _POS_COL[_c][_r]


# ══════════════════════════════════════════════════════════════════════════
#  REPRESENTACIÓN BITBOARD
#  bit = col*7 + fila_desde_abajo  (fila 0 = abajo)
#  Detecta victoria con 4 shifts de bits (7=horiz, 1=vert, 6=diag/, 8=diag\)
# ══════════════════════════════════════════════════════════════════════════

def _has_won(mask: int) -> bool:
    """Detecta 4 en raya en O(8 ops de bit). Sin numpy."""
    m = mask & (mask >> 7)
    if m & (m >> 14): return True   # horizontal
    m = mask & (mask >> 1)
    if m & (m >> 2):  return True   # vertical
    m = mask & (mask >> 6)
    if m & (m >> 12): return True   # diagonal /
    m = mask & (mask >> 8)
    if m & (m >> 16): return True   # diagonal backslash
    return False


def _board_to_bits(board: np.ndarray):
    """Convierte tablero numpy → (red_bits, yel_bits, heights[7])."""
    red = 0; yel = 0; heights = [0] * 7
    for c in range(7):
        for rfb in range(6):
            val = board[5 - rfb, c]
            if val == -1:
                red |= 1 << (c * 7 + rfb)
                if rfb + 1 > heights[c]: heights[c] = rfb + 1
            elif val == 1:
                yel |= 1 << (c * 7 + rfb)
                if rfb + 1 > heights[c]: heights[c] = rfb + 1
    return red, yel, heights


def _drop_bit(red, yel, heights, col, player):
    """Coloca ficha en col para player. Devuelve (new_r, new_y, new_h) o None."""
    h = heights[col]
    if h >= 6: return None
    bit = col * 7 + h
    nh = heights[:]
    nh[col] = h + 1
    if player == -1:
        return (red | (1 << bit), yel, nh)
    return (red, yel | (1 << bit), nh)


def _free_cols(heights) -> list:
    """Columnas libres ordenadas de más central a lateral."""
    return [c for c in COL_ORDER if heights[c] < 6]


def _immediate_win(red, yel, heights, player):
    """Devuelve columna de victoria inmediata, o None."""
    mask = red if player == -1 else yel
    for c in COL_ORDER:
        h = heights[c]
        if h < 6 and _has_won(mask | (1 << (c * 7 + h))):
            return c
    return None


def _gives_opp_win(red, yel, heights, col, player) -> bool:
    """True si jugar col regala victoria al oponente en su siguiente turno."""
    res = _drop_bit(red, yel, heights, col, player)
    if res is None: return False
    nr, ny, nh = res
    return _immediate_win(nr, ny, nh, -player) is not None


# ── Score posicional (suma de pesos de casillas propias menos ajenas) ─────
def _pos_score(red, yel, player) -> float:
    own = red if player == -1 else yel
    opp = yel if player == -1 else red
    s = 0
    tmp = own
    while tmp:
        lsb = tmp & (-tmp)
        s += _BIT_WEIGHT[lsb.bit_length() - 1]
        tmp ^= lsb
    tmp = opp
    while tmp:
        lsb = tmp & (-tmp)
        s -= _BIT_WEIGHT[lsb.bit_length() - 1]
        tmp ^= lsb
    return float(s)


# ── Default Policy (rollout heurístico) ───────────────────────────────────
# Concepto: "default policy" del MCTS (clase 13).
# En vez de aleatoria pura (Cruz/Portocarrero), usa heurística posicional:
#   1. Victoria inmediata  2. Bloqueo  3. Top-2 por score posicional (+ ruido)
def _rollout(red, yel, heights, player) -> float:
    """
    Simulación desde (red,yel,heights) con jugador activo=player.
    Devuelve +1 si gana player inicial, -1 si pierde, 0 empate.
    Default policy guiada (no aleatoria pura) — clase 13.
    """
    r, y, h, p = red, yel, heights, player
    for _ in range(42):
        fc = _free_cols(h)
        if not fc: break

        # 1. Victoria inmediata
        wc = _immediate_win(r, y, h, p)
        if wc is not None:
            _drop_bit(r, y, h, wc, p)   # solo para terminar el loop
            break

        # 2. Bloqueo inmediato
        bc = _immediate_win(r, y, h, -p)
        if bc is not None:
            res = _drop_bit(r, y, h, bc, p)
            if res: r, y, h = res
        else:
            # 3. Evaluar score posicional, elegir entre los 2 mejores
            scored = []
            for c in fc:
                res = _drop_bit(r, y, h, c, p)
                if res:
                    nr, ny, nh = res
                    scored.append((c, _pos_score(nr, ny, p), nr, ny, nh))
            if not scored: break
            scored.sort(key=lambda x: -x[1])
            best = random.choice(scored[:2])   # top-2 con algo de ruido
            r, y, h = best[2], best[3], best[4]

        if _has_won(r) or _has_won(y) or not _free_cols(h):
            break
        p = -p

    if _has_won(r if player == -1 else y): return  1.0
    if _has_won(y if player == -1 else r): return -1.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════
#  NODO MCTS
#  Implementa UCB de clase 12: u(s,a) = q̂(s,a) + C*sqrt(log(ΣN)/N[a])
# ══════════════════════════════════════════════════════════════════════════

class _Node:
    __slots__ = ("red", "yel", "heights", "key", "player",
                 "parent", "children", "N", "Q", "untried")

    def __init__(self, red, yel, heights, player, parent=None):
        self.red     = red
        self.yel     = yel
        self.heights = heights
        self.key     = (red, yel)
        self.player  = player      # jugador que ACABA DE MOVER (estado resultante)
        self.parent  = parent
        self.children = {}         # col → _Node
        self.N = 0
        self.Q = 0.0               # suma acumulada de utilidades (perspectiva root)

        fc = _free_cols(heights)
        terminal = _has_won(red) or _has_won(yel) or not fc
        if terminal:
            self.untried = []
        else:
            # Pruning: descartar movimientos que regalan victoria al rival
            mover = -player
            safe  = [c for c in fc
                     if not _gives_opp_win(red, yel, heights, c, mover)]
            self.untried = safe if safe else fc[:]

    def ucb_score(self, c_val: float, sign: float) -> float:
        """
        UCB de clase 12 (slide 22):
          u(s,a) = q̂(s,a) + sqrt(log(ΣN[s,a']) / N[s,a])
        sign = +1 si el nodo mueve en dirección de maximizar para el jugador
               que empezó la búsqueda; -1 en caso contrario.
        """
        if self.N == 0:
            return float("inf")
        exploit = sign * self.Q / self.N
        explore = c_val * math.sqrt(math.log(self.parent.N + 1) / self.N)
        return exploit + explore

    def best_child(self, c_val: float, sign: float) -> "_Node":
        return max(self.children.values(),
                   key=lambda ch: ch.ucb_score(c_val, sign))

    def expand(self, col: int, table: dict) -> "_Node":
        mover = -self.player
        res   = _drop_bit(self.red, self.yel, self.heights, col, mover)
        if res is None:
            if col in self.untried: self.untried.remove(col)
            return self
        nr, ny, nh = res
        child = _Node(nr, ny, nh, mover, parent=self)
        self.children[col] = child
        if col in self.untried: self.untried.remove(col)
        table[child.key] = child
        return child


# ══════════════════════════════════════════════════════════════════════════
#  MOTOR MCTS
#  Implementa el loop: Selection → Expansion → Simulation → Backpropagation
#  (clase 13, slide 25/26)
# ══════════════════════════════════════════════════════════════════════════

class _MCTS:
    def __init__(self, c: float = DEFAULT_UCB_C):
        self.c      = c
        self._table = {}   # key → _Node  (árbol persistente entre movidas)
        self.root   = None

    # ── Warmup (Exploring Starts desde posición vacía, clase 11) ──────────
    def warm_up(self, seconds: float):
        """
        Pre-pobla el árbol MCTS desde el tablero vacío antes de que empiece
        la partida. Esto implementa la idea de Exploring Starts (clase 11):
        comenzar trials desde un estado aleatorio inicial.
        Aquí el "estado inicial" es siempre s0 = tablero vacío, y se exploran
        distintas acciones iniciales gracias al UCB.
        """
        root = _Node(0, 0, [0]*7, 1)   # player=1: el que "acaba de mover" es 1
        self.root   = root
        self._table = {root.key: root}
        deadline    = time.time() + seconds
        while time.time() < deadline:
            self._iterate(root)

    # ── Búsqueda desde estado actual ──────────────────────────────────────
    def search(self, red, yel, heights, active_player, seconds: float) -> int:
        """
        Ejecuta el sub-proceso MCTS (inner trials, clase 13) desde el estado
        actual durante 'seconds' segundos. Devuelve la columna con mayor N.
        """
        key  = (red, yel)
        node = self._table.get(key)
        if node is None:
            node = _Node(red, yel, heights, -active_player)
            self._table[key] = node
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._iterate(node)
        if not node.children:
            fc = _free_cols(heights)
            return fc[0] if fc else 0
        # Acción final: columna con más visitas N (más confiable, clase 13)
        return max(node.children, key=lambda col: node.children[col].N)

    # ── Una iteración MCTS: Sel → Exp → Sim → Backprop ────────────────────
    def _iterate(self, root: _Node):
        node  = root
        final = _has_won(node.red) or _has_won(node.yel) or \
                not _free_cols(node.heights)

        # 1. SELECTION: bajar por el árbol usando UCB (tree policy)
        while not final and not node.untried and node.children:
            mover = -node.player
            sign  = 1.0 if mover == -1 else -1.0
            node  = node.best_child(self.c, sign)
            final = _has_won(node.red) or _has_won(node.yel) or \
                    not _free_cols(node.heights)

        # 2. EXPANSION: añadir un nodo hijo no visitado
        if not final and node.untried:
            col  = random.choice(node.untried)
            node = node.expand(col, self._table)

        # 3. SIMULATION: rollout con default policy heurística
        reward = _rollout(node.red, node.yel, node.heights, -node.player)

        # 4. BACKPROPAGATION: actualizar N y Q en el camino hacia la raíz
        #    Como es un Alternating Markov Game de suma cero (clase 12),
        #    la utilidad se invierte en cada nivel: r → -r → r → ...
        n, r = node, reward
        while n is not None:
            n.N += 1
            n.Q += r
            r    = -r
            n    = n.parent


# ══════════════════════════════════════════════════════════════════════════
#  POLÍTICA EXPORTABLE
# ══════════════════════════════════════════════════════════════════════════

class MontoyaAgent(Policy):
    """
    Parámetros configurables:
      ucb_c   : coeficiente C del UCB (clase 12). Default=1.41.
                C=0 → puramente explotador; C>2 → muy explorador.
      warmup  : si True, usa el 88% del budget pre-poblando el árbol
                antes de la primera jugada (Exploring Starts, clase 11).
                Si False, el árbol empieza vacío en la primera jugada.

    Versiones de análisis:
      MontoyaAgent(ucb_c=1.41, warmup=True)  → V-FULL (referencia)
      MontoyaAgent(ucb_c=1.41, warmup=False) → V-NOWARM
      MontoyaAgent(ucb_c=0.0,  warmup=True)  → V-GREEDY
    """

    def __init__(self,
                 default_budget: float = DEFAULT_BUDGET,
                 ucb_c: float = DEFAULT_UCB_C,
                 warmup: bool = True):
        self.default_budget = default_budget
        self.ucb_c          = ucb_c
        self.warmup         = warmup
        self._mcts          = None
        self._game_start    = None
        self._total_budget  = None

    def mount(self, time_budget=None):
        """
        Invocado antes de la partida con el budget total de tiempo.
        Si warmup=True, dedica el 88% a pre-poblar el árbol (Exploring Starts).
        """
        if time_budget is None:
            budget = self.default_budget
        elif isinstance(time_budget, (int, float)):
            budget = float(time_budget)
        elif isinstance(time_budget, (tuple, list)):
            try:   budget = float(time_budget[0])
            except Exception: budget = self.default_budget
        else:
            try:   budget = float(time_budget)
            except Exception: budget = self.default_budget

        self._total_budget = budget
        self._game_start   = time.time()
        self._mcts         = _MCTS(c=self.ucb_c)

        if self.warmup:
            warmup_secs = max(0.5, budget * 0.88)
            self._mcts.warm_up(warmup_secs)

    def act(self, s: np.ndarray) -> int:
        """
        Selecciona acción para el estado s.
        Orden de prioridades:
          1. Victoria inmediata (detectada por bitboard, O(8 ops))
          2. Bloqueo inmediato del rival
          3. MCTS con budget de tiempo restante proporcional a movidas restantes
        Los puntos 1 y 2 son casos especiales del GPI donde q(s,a_win)=1
        es determinísticamente conocido sin necesidad de simulaciones.
        """
        fc = [c for c in range(COLS) if s[0, c] == 0]
        if not fc: return 0

        # Determinar quién es el jugador activo en este estado
        reds    = int(np.sum(s == -1))
        yellows = int(np.sum(s == 1))
        active  = -1 if reds <= yellows else 1

        red, yel, heights = _board_to_bits(s)
        pieces = reds + yellows

        # 1. Victoria inmediata
        wc = _immediate_win(red, yel, heights, active)
        if wc is not None:
            return int(wc)

        # 2. Bloqueo inmediato
        bc = _immediate_win(red, yel, heights, -active)
        if bc is not None:
            return int(bc)

        # 3. MCTS (sub-proceso online, clase 13)
        if self._mcts is None:
            # Fallback si mount() no fue llamado: score posicional
            scored = [(c, _pos_score(*_drop_bit(red, yel, heights, c, active)[:2], active))
                      for c in fc
                      if _drop_bit(red, yel, heights, c, active) is not None]
            return int(max(scored, key=lambda x: x[1])[0]) if scored else fc[0]

        # Calcular tiempo por movida: distribuye el tiempo restante entre
        # las movidas que faltan (estimado como 21 - piezas/2)
        elapsed    = time.time() - self._game_start if self._game_start else 0.0
        total      = self._total_budget or DEFAULT_BUDGET
        time_left  = max(0.05, total - elapsed - 0.05)
        moves_left = max(1, 21 - pieces // 2)
        per_move   = min(time_left / moves_left, 2.0)
        per_move   = max(per_move, 0.05)

        col = self._mcts.search(red, yel, heights, active, per_move)
        if heights[col] >= 6:
            col = fc[0]
        return int(col)
