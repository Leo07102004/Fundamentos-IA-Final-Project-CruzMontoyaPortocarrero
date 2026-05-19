import math
import time
import numpy as np

from connect4.policy import Policy


ITERATION_BUDGET = 400
TIME_BUDGET_S = 0.35
UCB_C = math.sqrt(2)

ROWS, COLS = 6, 7
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]


def _free_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def _drop(board, col, player):
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            nb = board.copy()
            nb[r, col] = player
            return nb, r
    return None, -1


def _wins_at(b, r, c, p):
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


class _Node:
    __slots__ = ("board", "player", "parent", "action", "children", "untried", "N", "W")

    def __init__(self, board, player, parent=None, action=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.children = {}
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
            exploit = -child.q_hat
            explore = c * math.sqrt(log_n / child.N)
            score = exploit + explore
            if score > best_score:
                best_score, best = score, child
        return best


class _MCTS:
    def __init__(self, rng, c=UCB_C):
        self.rng = rng
        self.c = c

    def _simulate(self, board, player, root_player):
        b = board.copy()
        heights = _initial_heights(b)
        p = player
        total = int(np.count_nonzero(b))

        while True:
            legal = [c for c in range(COLS) if heights[c] < ROWS]
            if not legal:
                return 0.0
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
                idx = int(self.rng.integers(len(node.untried)))
                action = node.untried.pop(idx)
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

    def search(self, board, current_player, iterations, time_budget_s, allowed_actions=None):
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
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]


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

        legal = _free_cols(board)
        if not legal:
            return 0

        # (a) jugada ganadora inmediata
        for a in legal:
            nb, r = _drop(board, a, current_player)
            if nb is not None and _wins_at(nb, r, a, current_player):
                return int(a)

        # (b) bloquear jugada ganadora del rival
        for a in legal:
            nb, _ = _drop(board, a, current_player)
            if nb is None or _is_final(nb):
                continue
            opp_legal = _free_cols(nb)
            for b in opp_legal:
                nb2, r2 = _drop(nb, b, -current_player)
                if nb2 is not None and _wins_at(nb2, r2, b, -current_player):
                    if b in legal:
                        return int(b)
                    break

        # (c) descartar jugadas suicidas
        safe = []
        for a in legal:
            nb, _ = _drop(board, a, current_player)
            if nb is None:
                continue
            if _is_final(nb):
                safe.append(a)
                continue
            opp_wins = False
            for b in _free_cols(nb):
                nb2, r2 = _drop(nb, b, -current_player)
                if nb2 is not None and _wins_at(nb2, r2, b, -current_player):
                    opp_wins = True
                    break
            if not opp_wins:
                safe.append(a)
        if not safe:
            safe = legal
        allowed = safe if len(safe) < len(legal) else None

        action = self._mcts.search(
            board,
            current_player,
            iterations=ITERATION_BUDGET,
            time_budget_s=TIME_BUDGET_S,
            allowed_actions=allowed,
        )
        return int(action)