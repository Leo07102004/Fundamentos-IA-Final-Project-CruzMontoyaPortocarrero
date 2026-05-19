"""
Agente Connect-4: MCTS con Q-values y rollout heuristico
=========================================================
Estrategia:
  - Self-play offline en mount() usando Monte-Carlo Tree Search (MCTS) con UCB1.
  - Tabla hash O(1) para reutilizar nodos entre turnos del mismo juego.
  - Rollout heuristico: gana/bloquea inmediato + evaluacion posicional.
  - Propagacion bipolar de recompensas (Alternating Markov Game, sum-zero).
  - mount(time_budget) acepta float, int, tuple o None (compatible con Gradescope).

Diferencial clave frente a estrategias comunes:
  1. MCTS con UCB1 sobre Q-values compartidos (no Minimax, no puro random).
  2. Self-play en mount() -> Q-values reflejan ambos lados del juego.
  3. Reutilizacion de arbol entre movimientos via hash del tablero.
  4. Rollout con heuristica posicional y bloqueo/amenaza inmediata.
"""

import math
import time
import random
import numpy as np
from connect4.policy import Policy

# ── Hiperparametros configurables (para analisis empirico) ────────────────
DEFAULT_BUDGET = 28.0    # segundos totales por partida
UCB_C = 1.41             # constante UCB1 (sqrt(2))
ROWS, COLS = 6, 7
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]   # centro primero

# Pesos posicionales (centro y filas bajas valen mas)
_POS_WEIGHT = np.array([
    [3, 4, 5, 7, 5, 4, 3],
    [4, 6, 8, 10, 8, 6, 4],
    [5, 7, 9, 11, 9, 7, 5],
    [5, 7, 9, 11, 9, 7, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3],
], dtype=float)


# ── Utilidades de tablero ─────────────────────────────────────────────────

def _free_cols(board):
    return [c for c in COL_ORDER if board[0, c] == 0]


def _drop(board, col, player):
    """Devuelve nuevo tablero con ficha colocada, o None si columna llena."""
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            nb = board.copy()
            nb[r, col] = player
            return nb
    return None


def _winner(board):
    b = board
    for r in range(ROWS):
        for c in range(COLS):
            p = b[r, c]
            if p == 0:
                continue
            if c + 3 < COLS and b[r, c+1] == p and b[r, c+2] == p and b[r, c+3] == p:
                return p
            if r + 3 < ROWS and b[r+1, c] == p and b[r+2, c] == p and b[r+3, c] == p:
                return p
            if (r + 3 < ROWS and c + 3 < COLS
                    and b[r+1, c+1] == p and b[r+2, c+2] == p and b[r+3, c+3] == p):
                return p
            if (r + 3 < ROWS and c - 3 >= 0
                    and b[r+1, c-1] == p and b[r+2, c-2] == p and b[r+3, c-3] == p):
                return p
    return 0


def _is_final(board):
    return _winner(board) != 0 or all(board[0, c] != 0 for c in range(COLS))


def _immediate_win_col(board, player):
    """Columna que da victoria inmediata a player, o None."""
    for c in COL_ORDER:
        if board[0, c] != 0:
            continue
        nb = _drop(board, c, player)
        if nb is not None and _winner(nb) == player:
            return c
    return None


def _score_board(board, player):
    """Evaluacion posicional: fichas propias menos oponente ponderadas por posicion."""
    own = (board == player).astype(float)
    opp = (board == -player).astype(float)
    return float(np.sum(_POS_WEIGHT * own) - np.sum(_POS_WEIGHT * opp))


def _rollout(board, player):
    """
    Rollout heuristico:
    1. Gana inmediatamente si puede.
    2. Bloquea victoria inmediata del oponente.
    3. Elige entre las 3 mejores columnas por peso posicional.
    Devuelve +1 si gana Rojo(-1), -1 si gana Amarillo(1), 0 empate.
    """
    b = board.copy()
    p = player
    for _ in range(42):
        fc = _free_cols(b)
        if not fc:
            break
        win_col = _immediate_win_col(b, p)
        if win_col is not None:
            b = _drop(b, win_col, p)
            break
        blk_col = _immediate_win_col(b, -p)
        if blk_col is not None:
            nb = _drop(b, blk_col, p)
            if nb is not None:
                b = nb
        else:
            scored = []
            for c in fc:
                nb = _drop(b, c, p)
                if nb is not None:
                    scored.append((c, _score_board(nb, p)))
            if not scored:
                break
            scored.sort(key=lambda x: -x[1])
            top = scored[:3]
            chosen = random.choice(top)[0]
            nb = _drop(b, chosen, p)
            if nb is not None:
                b = nb
        if _is_final(b):
            break
        p = -p
    w = _winner(b)
    if w == -1:
        return 1.0
    if w == 1:
        return -1.0
    return 0.0


# ── Nodo MCTS ─────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("board", "key", "player", "parent", "children", "N", "Q", "untried")

    def __init__(self, board, player, parent=None):
        self.board = board
        self.key = board.tobytes()
        self.player = player        # quien acaba de mover (convencion interna)
        self.parent = parent
        self.children = {}          # col -> _Node
        self.N = 0
        self.Q = 0.0
        self.untried = _free_cols(board) if not _is_final(board) else []

    def ucb(self, c, sign):
        if self.N == 0:
            return float("inf")
        return sign * self.Q / self.N + c * math.sqrt(math.log(self.parent.N + 1) / self.N)

    def best_child(self, c, sign):
        return max(self.children.values(), key=lambda ch: ch.ucb(c, sign))

    def expand(self, col, table):
        mover = -self.player
        nb = _drop(self.board, col, mover)
        child = _Node(nb, mover, parent=self)
        self.children[col] = child
        self.untried.remove(col)
        table[child.key] = child
        return child


# ── Motor MCTS ────────────────────────────────────────────────────────────

class _MCTS:
    def __init__(self, c=UCB_C):
        self.c = c
        self.root = None
        self._table = {}

    def warm_up(self, seconds):
        """Self-play desde tablero vacio durante `seconds` segundos."""
        board = np.zeros((ROWS, COLS), dtype=int)
        self.root = _Node(board, 1)   # player=1: nadie movio; el primero en mover sera -1
        self._table = {self.root.key: self.root}
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._iterate(self.root)

    def search(self, board, active_player, seconds):
        """Busca la mejor columna para active_player dado el tablero."""
        key = board.tobytes()
        node = self._table.get(key)
        if node is None:
            node = _Node(board, -active_player)
            self._table[key] = node
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._iterate(node)
        if not node.children:
            fc = _free_cols(board)
            return fc[0] if fc else 0
        return max(node.children, key=lambda col: node.children[col].N)

    def _iterate(self, root):
        node = root
        # Seleccion: UCB1 hasta nodo hoja
        while not _is_final(node.board) and not node.untried and node.children:
            mover = -node.player
            sign = 1 if mover == -1 else -1
            node = node.best_child(self.c, sign)
        # Expansion
        if not _is_final(node.board) and node.untried:
            col = random.choice(node.untried)
            node = node.expand(col, self._table)
        # Simulacion (rollout heuristico)
        reward = _rollout(node.board, -node.player)
        # Retropropagacion bipolar
        n, r = node, reward
        while n is not None:
            n.N += 1
            n.Q += r
            r = -r
            n = n.parent


# ── Politica exportable (interfaz del torneo) ─────────────────────────────

class MCTSAgent(Policy):
    """
    Agente MCTS con Q-values para Connect-4.

    mount(time_budget) es compatible con Gradescope:
    acepta float, int, tuple, list o None.

    Parametros para analisis empirico:
        default_budget : segundos de warmup si no se recibe argumento
        ucb_c          : constante UCB1
    """

    def __init__(self, default_budget=DEFAULT_BUDGET, ucb_c=UCB_C):
        self.default_budget = default_budget
        self.ucb_c = ucb_c
        self._mcts = None
        self._game_start = None
        self._total_budget = None

    def mount(self, time_budget=None):
        """
        Precalienta el agente con self-play MCTS.

        time_budget puede ser:
          - None          -> usa default_budget
          - float / int   -> usa ese valor en segundos
          - tuple / list  -> usa el primer elemento
          - cualquier otro -> intenta float(), si falla usa default_budget
        """
        if time_budget is None:
            budget = self.default_budget
        elif isinstance(time_budget, (int, float)):
            budget = float(time_budget)
        elif isinstance(time_budget, (tuple, list)):
            try:
                budget = float(time_budget[0])
            except Exception:
                budget = self.default_budget
        else:
            try:
                budget = float(time_budget)
            except Exception:
                budget = self.default_budget

        self._total_budget = budget
        self._game_start = time.time()

        # Usar casi todo el presupuesto en warmup; dejar ~2.5 s para los act()
        warmup_secs = max(0.5, budget - 2.5)
        self._mcts = _MCTS(c=self.ucb_c)
        self._mcts.warm_up(warmup_secs)

    def act(self, s):
        """
        Elige la columna. Aplica logica tactica rapida antes de invocar MCTS.
        """
        fc = [c for c in range(COLS) if s[0, c] == 0]
        if not fc:
            return 0

        # Detectar jugador activo por conteo de fichas
        reds = int(np.sum(s == -1))
        yellows = int(np.sum(s == 1))
        active = -1 if reds <= yellows else 1

        # Tactica inmediata (sin coste MCTS)
        win_col = _immediate_win_col(s, active)
        if win_col is not None:
            return int(win_col)
        blk_col = _immediate_win_col(s, -active)
        if blk_col is not None:
            return int(blk_col)

        # Si mount() no se llamo (fallback de seguridad)
        if self._mcts is None:
            scored = []
            for c in fc:
                nb = _drop(s, c, active)
                if nb is not None:
                    scored.append((c, _score_board(nb, active)))
            if scored:
                return int(max(scored, key=lambda x: x[1])[0])
            return fc[0]

        # Presupuesto por movimiento (distribuir tiempo restante)
        elapsed = time.time() - self._game_start if self._game_start else 0.0
        total = self._total_budget if self._total_budget else DEFAULT_BUDGET
        time_left = max(0.05, total - elapsed - 0.1)
        per_move = min(time_left / max(1, len(fc)), 1.5)
        per_move = max(per_move, 0.05)

        col = self._mcts.search(s, active, per_move)

        # Validacion de seguridad
        if s[0, col] != 0:
            col = fc[0]
        return int(col)
        