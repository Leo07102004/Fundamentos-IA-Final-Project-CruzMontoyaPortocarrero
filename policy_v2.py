"""
Agente Connect-4 V2: MCTS con Bitboards
========================================
Mejoras sobre V1:
  1. BITBOARDS: representa el tablero como dos enteros de 64 bits (uno por color).
     - _winner: 7.1 us -> 0.9 us  (8x)
     - _immediate_win: 59.8 us -> 3.5 us  (17x)
     - Sin copias de array numpy en el hot path.
     - Resultado: ~9x mas iteraciones MCTS en el mismo presupuesto de tiempo.

  2. LOSING-MOVE PRUNING en expansion: si una columna le da la victoria al oponente
     en su siguiente turno, se descarta de untried (salvo que todas lo hagan).

  3. DOUBLE-THREAT (fork) detection en act(): detecta si el jugador activo puede crear
     dos amenazas simultaneas con un solo movimiento, y lo prioriza; igual para
     bloquear el fork del oponente.

  4. TIEMPO: warmup usa el 90% del presupuesto; per_move se estima por movimientos
     restantes en la partida, no por columnas libres.

Representacion de bitboard (estandar Connect-4):
  bit = col*7 + fila_desde_abajo  (fila 0 = abajo, fila 5 = arriba)
  Detecciones de victoria: shifts 7 (horiz), 1 (vert), 6 (diag/), 8 (diag\\)
"""

import math
import time
import random
import numpy as np
from connect4.policy import Policy

# ── Hiperparametros ────────────────────────────────────────────────────────
DEFAULT_BUDGET = 28.0
UCB_C          = 1.41
ROWS, COLS     = 6, 7
COL_ORDER      = [3, 2, 4, 1, 5, 0, 6]

# Pesos posicionales indexados por bit (col*7 + fila_desde_abajo)
_BIT_WEIGHT = [0] * 49
_POS_COL = [
    [3, 4, 5, 5, 4, 3],
    [4, 6, 7, 7, 6, 4],
    [5, 8, 9, 9, 8, 5],
    [7,10,11,11,10, 7],
    [5, 8, 9, 9, 8, 5],
    [4, 6, 7, 7, 6, 4],
    [3, 4, 5, 5, 4, 3],
]
for _c in range(7):
    for _r in range(6):
        _BIT_WEIGHT[_c * 7 + _r] = _POS_COL[_c][_r]


# ── Utilidades de bitboard ─────────────────────────────────────────────────

def _has_won(mask: int) -> bool:
    """Victoria en O(8 ops de bit). No usa numpy."""
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
    """Convierte tablero numpy -> (red_bits, yel_bits, heights[7])."""
    red = 0; yel = 0; heights = [0] * 7
    for c in range(7):
        for rfb in range(6):       # rfb = row from bottom (0=abajo)
            val = board[5 - rfb, c]
            if val == -1:
                red |= 1 << (c * 7 + rfb)
                if rfb + 1 > heights[c]: heights[c] = rfb + 1
            elif val == 1:
                yel |= 1 << (c * 7 + rfb)
                if rfb + 1 > heights[c]: heights[c] = rfb + 1
    return red, yel, heights


def _drop_bit(red: int, yel: int, heights: list, col: int, player: int):
    """Coloca ficha. Devuelve (new_red, new_yel, new_heights) o None."""
    h = heights[col]
    if h >= 6: return None
    bit = col * 7 + h
    nh = heights[:]
    nh[col] = h + 1
    if player == -1:
        return (red | (1 << bit), yel, nh)
    return (red, yel | (1 << bit), nh)


def _free_cols_h(heights: list) -> list:
    return [c for c in COL_ORDER if heights[c] < 6]


def _immediate_win_bit(red: int, yel: int, heights: list, player: int):
    """Columna que da victoria inmediata, o None."""
    mask = red if player == -1 else yel
    for c in COL_ORDER:
        h = heights[c]
        if h < 6 and _has_won(mask | (1 << (c * 7 + h))):
            return c
    return None


def _score_bits(red: int, yel: int, player: int) -> float:
    """Evaluacion posicional via popcount sobre bitboard."""
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


def _gives_opponent_win(red: int, yel: int, heights: list,
                        col: int, player: int) -> bool:
    """True si jugar col le regala victoria inmediata al oponente."""
    res = _drop_bit(red, yel, heights, col, player)
    if res is None: return False
    nr, ny, nh = res
    return _immediate_win_bit(nr, ny, nh, -player) is not None


def _rollout_bb(red: int, yel: int, heights: list, player: int) -> float:
    """
    Rollout heuristico puro sobre bitboard.
    1. Victoria inmediata.
    2. Bloqueo inmediato.
    3. Top-3 por score posicional.
    """
    r, y, h, p = red, yel, heights, player
    for _ in range(42):
        fc = _free_cols_h(h)
        if not fc: break

        wc = _immediate_win_bit(r, y, h, p)
        if wc is not None:
            res = _drop_bit(r, y, h, wc, p)
            r, y, h = res
            break

        bc = _immediate_win_bit(r, y, h, -p)
        if bc is not None:
            res = _drop_bit(r, y, h, bc, p)
            if res: r, y, h = res
        else:
            scored = []
            for c in fc:
                res = _drop_bit(r, y, h, c, p)
                if res:
                    nr, ny, nh = res
                    scored.append((c, _score_bits(nr, ny, p), nr, ny, nh))
            if not scored: break
            scored.sort(key=lambda x: -x[1])
            best = random.choice(scored[:3])
            r, y, h = best[2], best[3], best[4]

        if _has_won(r) or _has_won(y) or not _free_cols_h(h):
            break
        p = -p

    if _has_won(r): return  1.0
    if _has_won(y): return -1.0
    return 0.0


# ── Nodo MCTS ─────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("red", "yel", "heights", "key", "player",
                 "parent", "children", "N", "Q", "untried")

    def __init__(self, red, yel, heights, player, parent=None):
        self.red     = red
        self.yel     = yel
        self.heights = heights
        self.key     = (red, yel)
        self.player  = player
        self.parent  = parent
        self.children = {}
        self.N = 0
        self.Q = 0.0

        fc    = _free_cols_h(heights)
        final = _has_won(red) or _has_won(yel) or not fc
        if final:
            self.untried = []
        else:
            # Pruning: descartar movimientos suicidas
            mover = -player
            safe  = [c for c in fc
                     if not _gives_opponent_win(red, yel, heights, c, mover)]
            self.untried = safe if safe else fc[:]

    def ucb(self, c_val, sign):
        if self.N == 0: return float("inf")
        return (sign * self.Q / self.N
                + c_val * math.sqrt(math.log(self.parent.N + 1) / self.N))

    def best_child(self, c_val, sign):
        return max(self.children.values(),
                   key=lambda ch: ch.ucb(c_val, sign))

    def expand(self, col, table):
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


# ── Motor MCTS ────────────────────────────────────────────────────────────

class _MCTS:
    def __init__(self, c=UCB_C):
        self.c = c
        self.root   = None
        self._table = {}

    def warm_up(self, seconds: float):
        init = _Node(0, 0, [0]*7, 1)
        self.root   = init
        self._table = {init.key: init}
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._iterate(init)

    def search(self, red, yel, heights, active_player, seconds) -> int:
        key  = (red, yel)
        node = self._table.get(key)
        if node is None:
            node = _Node(red, yel, heights, -active_player)
            self._table[key] = node
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._iterate(node)
        if not node.children:
            fc = _free_cols_h(heights)
            return fc[0] if fc else 0
        return max(node.children, key=lambda col: node.children[col].N)

    def _iterate(self, root):
        node  = root
        final = _has_won(node.red) or _has_won(node.yel) or not _free_cols_h(node.heights)
        # Seleccion
        while not final and not node.untried and node.children:
            mover = -node.player
            sign  = 1 if mover == -1 else -1
            node  = node.best_child(self.c, sign)
            final = _has_won(node.red) or _has_won(node.yel) or not _free_cols_h(node.heights)
        # Expansion
        if not final and node.untried:
            col  = random.choice(node.untried)
            node = node.expand(col, self._table)
        # Rollout
        reward = _rollout_bb(node.red, node.yel, node.heights, -node.player)
        # Retropropagacion
        n, r = node, reward
        while n is not None:
            n.N += 1
            n.Q += r
            r    = -r
            n    = n.parent


# ── Fork detection ─────────────────────────────────────────────────────────

def _count_threats(red, yel, heights, player) -> int:
    """Cuenta victorias inmediatas disponibles para player."""
    mask = red if player == -1 else yel
    cnt  = 0
    for c in range(7):
        h = heights[c]
        if h < 6 and _has_won(mask | (1 << (c * 7 + h))):
            cnt += 1
    return cnt


def _find_fork(red, yel, heights, player):
    """Columna que crea 2+ amenazas de victoria simultaneas, o None."""
    for c in COL_ORDER:
        if heights[c] >= 6: continue
        res = _drop_bit(red, yel, heights, c, player)
        if res is None: continue
        nr, ny, nh = res
        if _count_threats(nr, ny, nh, player) >= 2:
            return c
    return None


# ── Politica exportable ───────────────────────────────────────────────────

class MCTSAgent(Policy):
    """
    MCTS V2 con bitboards.
    Interfaz identica a V1; acepta mount(float|int|tuple|None).
    """

    def __init__(self, default_budget=DEFAULT_BUDGET, ucb_c=UCB_C):
        self.default_budget = default_budget
        self.ucb_c          = ucb_c
        self._mcts          = None
        self._game_start    = None
        self._total_budget  = None

    def mount(self, time_budget=None):
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

        warmup_secs = max(0.5, budget * 0.90)
        self._mcts  = _MCTS(c=self.ucb_c)
        self._mcts.warm_up(warmup_secs)

    def act(self, s: np.ndarray) -> int:
        fc = [c for c in range(COLS) if s[0, c] == 0]
        if not fc: return 0

        reds    = int(np.sum(s == -1))
        yellows = int(np.sum(s == 1))
        active  = -1 if reds <= yellows else 1

        red, yel, heights = _board_to_bits(s)

        # 1. Victoria inmediata
        wc = _immediate_win_bit(red, yel, heights, active)
        if wc is not None: return int(wc)

        # 2. Bloqueo
        bc = _immediate_win_bit(red, yel, heights, -active)
        if bc is not None: return int(bc)

        # 3. Crear fork
        fk = _find_fork(red, yel, heights, active)
        if fk is not None: return int(fk)

        # 4. Bloquear fork del oponente
        opp_fk = _find_fork(red, yel, heights, -active)
        if opp_fk is not None: return int(opp_fk)

        # 5. MCTS
        if self._mcts is None:
            scored = []
            for c in fc:
                res = _drop_bit(red, yel, heights, c, active)
                if res:
                    nr, ny, _ = res
                    scored.append((c, _score_bits(nr, ny, active)))
            return int(max(scored, key=lambda x: x[1])[0]) if scored else fc[0]

        elapsed    = time.time() - self._game_start if self._game_start else 0.0
        total      = self._total_budget or DEFAULT_BUDGET
        time_left  = max(0.05, total - elapsed - 0.05)
        moves_left = max(1, 21 - (reds + yellows) // 2)
        per_move   = min(time_left / moves_left, 2.0)
        per_move   = max(per_move, 0.05)

        col = self._mcts.search(red, yel, heights, active, per_move)

        if heights[col] >= 6:
            col = fc[0]
        return int(col)
