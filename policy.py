import math
import time
import numpy as np

from connect4.policy import Policy


# Hiperparametros del agente
ITERATION_BUDGET = 800     # simulaciones MCTS por jugada (subido desde 400)
TIME_BUDGET_S = 0.45       # limite duro de tiempo (segundos)
UCB_C = 1.4                # exploracion UCB1 (calibrado para factor de rama 7)
ROLLOUT_HEURISTIC = True   # rollout con gana/bloquea inmediato

ROWS, COLS = 6, 7
# Orden centro-primero: las columnas centrales tienen mas diagonales y son
# estrategicamente superiores. Esto sesga la expansion del arbol.
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]


# --------------------------------------------------------------------------
# Utilidades de tablero (numpy plano, sin ConnectState)
# --------------------------------------------------------------------------

def _free_cols(board):
    return [c for c in COL_ORDER if board[0, c] == 0]


def _drop(board, col, player):
    """Coloca ficha y devuelve (nuevo_tablero, fila) o (None, -1)."""
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            nb = board.copy()
            nb[r, col] = player
            return nb, r
    return None, -1


def _wins_at(b, r, c, p):
    """True si la pieza p en (r,c) cierra un 4-en-linea."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr, cc] == p:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr, cc] == p:
            count += 1
            rr -= dr
            cc -= dc
        if count >= 4:
            return True
    return False


def _winner(board):
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            if c + 3 < COLS and board[r, c+1] == p and board[r, c+2] == p and board[r, c+3] == p:
                return p
            if r + 3 < ROWS and board[r+1, c] == p and board[r+2, c] == p and board[r+3, c] == p:
                return p
            if r + 3 < ROWS and c + 3 < COLS and board[r+1, c+1] == p and board[r+2, c+2] == p and board[r+3, c+3] == p:
                return p
            if r + 3 < ROWS and c - 3 >= 0 and board[r+1, c-1] == p and board[r+2, c-2] == p and board[r+3, c-3] == p:
                return p
    return 0


def _is_final(board):
    return _winner(board) != 0 or all(board[0, c] != 0 for c in range(COLS))


def _initial_heights(board):
    h = [0] * COLS
    for c in range(COLS):
        for r in range(ROWS):
            if board[r, c] != 0:
                h[c] = ROWS - r
                break
    return h


def _count_immediate_wins(board, heights, player):
    """Cuenta cuantas columnas darian victoria inmediata a `player`.
    Usa el vector de alturas para evitar copiar el tablero. Solo escribe y
    revierte la celda donde caeria la ficha."""
    wins = 0
    winning_cols = []
    for c in COL_ORDER:
        if heights[c] >= ROWS:
            continue
        r = ROWS - 1 - heights[c]
        board[r, c] = player
        if _wins_at(board, r, c, player):
            wins += 1
            winning_cols.append(c)
        board[r, c] = 0
    return wins, winning_cols


# --------------------------------------------------------------------------
# Nodo MCTS
# --------------------------------------------------------------------------

class _Node:
    __slots__ = ("board", "player", "parent", "action", "children", "untried", "N", "W")

    def __init__(self, board, player, parent=None, action=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.children = {}
        # Expansion en orden centro -> bordes (mejora calidad MCTS gratis)
        self.untried = _free_cols(board) if not _is_final(board) else []
        self.N = 0
        self.W = 0.0

    @property
    def is_fully_expanded(self):
        return len(self.untried) == 0

    @property
    def is_terminal(self):
        return _is_final(self.board)

    @property
    def q_hat(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def ucb1_child(self, c):
        log_n = math.log(self.N)
        best = None
        best_score = -float("inf")
        for child in self.children.values():
            exploit = -child.q_hat   # suma cero: invertir para perspectiva del padre
            explore = c * math.sqrt(log_n / child.N)
            score = exploit + explore
            if score > best_score:
                best_score, best = score, child
        return best


# --------------------------------------------------------------------------
# Motor MCTS
# --------------------------------------------------------------------------

class _MCTS:
    def __init__(self, rng, c=UCB_C):
        self.rng = rng
        self.c = c

    def _rollout_choose(self, board, heights, player):
        """Elige columna en el rollout: prioriza ganar inmediato, luego bloquear,
        luego centro-aleatorio. Reduce varianza vs aleatorio puro."""
        # 1. Ganar inmediato si puedo
        for c in COL_ORDER:
            if heights[c] >= ROWS:
                continue
            r = ROWS - 1 - heights[c]
            board[r, c] = player
            won = _wins_at(board, r, c, player)
            board[r, c] = 0
            if won:
                return c
        # 2. Bloquear si el rival gana inmediato
        for c in COL_ORDER:
            if heights[c] >= ROWS:
                continue
            r = ROWS - 1 - heights[c]
            board[r, c] = -player
            opp_won = _wins_at(board, r, c, -player)
            board[r, c] = 0
            if opp_won:
                return c
        # 3. Aleatorio (orden centro-primero ya esta implicito en legal)
        legal = [c for c in range(COLS) if heights[c] < ROWS]
        return legal[int(self.rng.integers(len(legal)))]

    def _simulate(self, board, player, root_player):
        b = board.copy()
        heights = _initial_heights(b)
        p = player
        total = int(np.count_nonzero(b))

        while True:
            if all(h >= ROWS for h in heights):
                return 0.0
            if ROLLOUT_HEURISTIC:
                a = self._rollout_choose(b, heights, p)
            else:
                legal = [c for c in range(COLS) if heights[c] < ROWS]
                a = legal[int(self.rng.integers(len(legal)))]
            r = ROWS - 1 - heights[a]
            b[r, a] = p
            heights[a] += 1
            total += 1
            if _wins_at(b, r, a, p):
                return 1.0 if p == root_player else -1.0
            if total >= ROWS * COLS:
                return 0.0
            p = -p

    def _select_and_expand(self, root):
        node = root
        while not node.is_terminal:
            if not node.is_fully_expanded:
                # Expandir en orden centro-primero (pop(0)) en vez de aleatorio
                action = node.untried.pop(0)
                nb, _ = _drop(node.board, action, node.player)
                child = _Node(nb, -node.player, parent=node, action=action)
                node.children[action] = child
                return child
            node = node.ucb1_child(self.c)
        return node

    def _backpropagate(self, leaf, reward_root, root_player):
        node = leaf
        while node is not None:
            sign = 1.0 if node.player == root_player else -1.0
            node.N += 1
            node.W += sign * reward_root
            node = node.parent

    def search(self, board, current_player, iterations, time_budget_s,
               allowed_actions=None):
        root = _Node(board, current_player)
        if allowed_actions is not None:
            root.untried = [a for a in root.untried if a in allowed_actions]
        root_player = current_player
        deadline = time.monotonic() + time_budget_s if time_budget_s > 0 else None

        for _ in range(iterations):
            leaf = self._select_and_expand(root)
            if leaf.is_terminal:
                w = _winner(leaf.board)
                reward = 0.0 if w == 0 else (1.0 if w == root_player else -1.0)
            else:
                reward = self._simulate(leaf.board, leaf.player, root_player)
            self._backpropagate(leaf, reward, root_player)
            if deadline is not None and time.monotonic() > deadline:
                break

        if not root.children:
            choices = allowed_actions if allowed_actions else _free_cols(board)
            return int(self.rng.choice(choices))
        # Robust child: accion mas visitada (estandar MCTS)
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]


# --------------------------------------------------------------------------
# Politica expuesta al torneo
# --------------------------------------------------------------------------

class Portocarrero(Policy):

    def __init__(self):
        self._rng = None
        self._mcts = None

    def _ensure_ready(self):
        if self._mcts is None:
            self._rng = np.random.default_rng()
            self._mcts = _MCTS(self._rng, c=UCB_C)

    def mount(self, time_budget=None) -> None:
        self._ensure_ready()

    def act(self, s: np.ndarray) -> int:
        self._ensure_ready()
        board = s.copy()
        n_red = int((board == -1).sum())
        n_yel = int((board == 1).sum())
        current_player = -1 if n_red == n_yel else 1

        heights = _initial_heights(board)
        legal = _free_cols(board)
        if not legal:
            return 0

        # ============ Capa tactica 0: ganar inmediato ===================
        for a in legal:
            r = ROWS - 1 - heights[a]
            board[r, a] = current_player
            won = _wins_at(board, r, a, current_player)
            board[r, a] = 0
            if won:
                return int(a)

        # ============ Capa tactica 1: bloquear amenaza inmediata ========
        opp_wins, opp_winning_cols = _count_immediate_wins(
            board, heights, -current_player
        )
        if opp_wins >= 1:
            # Si hay 2+ amenazas, no podemos bloquear todas: estamos perdidos,
            # bloquea cualquiera y reza (MCTS no salvara nada aqui).
            return int(opp_winning_cols[0])

        # ============ Capa tactica 2: DOBLE AMENAZA (lookahead 2-ply) ===
        # Si puedo jugar una columna que me deja con 2 victorias en 1, gano
        # forzado (el rival solo puede bloquear una). Esto detecta jugadas
        # ganadoras que MCTS podria no descubrir con presupuesto bajo.
        for a in legal:
            r = ROWS - 1 - heights[a]
            board[r, a] = current_player
            heights[a] += 1
            my_wins, _ = _count_immediate_wins(board, heights, current_player)
            # Tambien hay que verificar que el rival no gane primero
            opp_threats, _ = _count_immediate_wins(board, heights, -current_player)
            heights[a] -= 1
            board[r, a] = 0
            if my_wins >= 2 and opp_threats == 0:
                return int(a)

        # ============ Capa tactica 3: filtrar jugadas suicidas ==========
        # Descartar columnas que regalan al rival victoria inmediata (1-ply)
        # o doble amenaza (2-ply, el rival nos forzaria a perder).
        safe = []
        for a in legal:
            r = ROWS - 1 - heights[a]
            board[r, a] = current_player
            heights[a] += 1
            # ¿el rival gana inmediato tras mi jugada?
            opp_wins_now, _ = _count_immediate_wins(
                board, heights, -current_player
            )
            # ¿el rival puede crear doble amenaza tras mi jugada?
            opp_creates_double = False
            if opp_wins_now == 0:
                for b in range(COLS):
                    if heights[b] >= ROWS:
                        continue
                    rb = ROWS - 1 - heights[b]
                    board[rb, b] = -current_player
                    heights[b] += 1
                    opp_after, _ = _count_immediate_wins(
                        board, heights, -current_player
                    )
                    heights[b] -= 1
                    board[rb, b] = 0
                    if opp_after >= 2:
                        opp_creates_double = True
                        break
            heights[a] -= 1
            board[r, a] = 0
            if opp_wins_now == 0 and not opp_creates_double:
                safe.append(a)

        if not safe:
            safe = legal
        allowed = safe if len(safe) < len(legal) else None

        # ============ Capa estrategica: MCTS-UCB1 =======================
        action = self._mcts.search(
            board,
            current_player,
            iterations=ITERATION_BUDGET,
            time_budget_s=TIME_BUDGET_S,
            allowed_actions=allowed,
        )
        return int(action)
