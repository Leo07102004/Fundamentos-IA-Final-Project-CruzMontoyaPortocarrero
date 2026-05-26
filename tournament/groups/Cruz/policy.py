import math
import time
import pickle
import numpy as np
from pathlib import Path

from connect4.policy import Policy


ITERATION_BUDGET = 350
TIME_BUDGET_S = 0.20
UCB_C = math.sqrt(2)

ROWS, COLS = 6, 7
CENTER_COL = 3
MODEL_FILE = "q_values.pkl"


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


def _board_key(board, player):
    return (tuple(board.flatten().tolist()), int(player))


def _winning_moves(board, player):
    ans = []
    for a in _free_cols(board):
        nb, r = _drop(board, a, player)
        if nb is not None and _wins_at(nb, r, a, player):
            ans.append(a)
    return ans


def _opponent_can_win_next(board, player):
    opp = -player
    for a in _free_cols(board):
        nb, r = _drop(board, a, opp)
        if nb is not None and _wins_at(nb, r, a, opp):
            return True
    return False


def _safe_moves(board, player):
    safe = []
    for a in _free_cols(board):
        nb, _ = _drop(board, a, player)
        if nb is None:
            continue
        if _winner(nb) == player:
            safe.append(a)
            continue
        if not _opponent_can_win_next(nb, player):
            safe.append(a)
    return safe


def _ordered(moves):
    return sorted(moves, key=lambda c: abs(c - CENTER_COL))


class _Node:
    __slots__ = ("board", "player", "parent", "action", "children", "untried", "N", "W")

    def __init__(self, board, player, parent=None, action=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.children = {}
        self.untried = _ordered(_free_cols(board)) if not _is_final(board) else []
        self.N = 0
        self.W = 0.0

    @property
    def is_terminal(self):
        return _is_final(self.board)

    @property
    def is_fully_expanded(self):
        return len(self.untried) == 0

    @property
    def q_hat(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def ucb1_child(self, c):
        log_n = math.log(max(1, self.N))
        best = None
        best_score = -float("inf")
        for child in self.children.values():
            exploit = -child.q_hat
            explore = c * math.sqrt(log_n / child.N)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best


class _MCTS:
    def __init__(self, rng, c=UCB_C):
        self.rng = rng
        self.c = c

    def _simulate(self, board, player, root_player):
        b = board.copy()
        p = player
        while True:
            if _is_final(b):
                w = _winner(b)
                if w == 0:
                    return 0.0
                return 1.0 if w == root_player else -1.0

            wins = _winning_moves(b, p)
            if wins:
                a = _ordered(wins)[0]
            else:
                blocks = _winning_moves(b, -p)
                if blocks:
                    a = _ordered(blocks)[0]
                else:
                    legal = _ordered(_free_cols(b))
                    a = legal[int(self.rng.integers(len(legal)))]

            b, r = _drop(b, a, p)
            if _wins_at(b, r, a, p):
                return 1.0 if p == root_player else -1.0
            p = -p

    def _select_and_expand(self, root):
        node = root
        while not node.is_terminal:
            if not node.is_fully_expanded:
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

    def search(self, board, current_player, iterations, time_budget_s, allowed_actions=None):
        root = _Node(board, current_player)
        if allowed_actions is not None:
            root.untried = [a for a in root.untried if a in allowed_actions]

        deadline = time.monotonic() + time_budget_s
        root_player = current_player

        for _ in range(iterations):
            leaf = self._select_and_expand(root)
            if leaf.is_terminal:
                w = _winner(leaf.board)
                reward = 0.0 if w == 0 else (1.0 if w == root_player else -1.0)
            else:
                reward = self._simulate(leaf.board, leaf.player, root_player)
            self._backpropagate(leaf, reward, root_player)

            if time.monotonic() > deadline:
                break

        if not root.children:
            choices = allowed_actions if allowed_actions else _free_cols(board)
            return int(self.rng.choice(choices))

        return max(root.children.items(), key=lambda kv: kv[1].N)[0]


class AgenT1000(Policy):
    def __init__(self):
        self._rng = None
        self._mcts = None
        self._q_values = {}
        self._counts = {}

    def mount(self, time_budget=None) -> None:
        if self._mcts is None:
            self._rng = np.random.default_rng()
            self._mcts = _MCTS(self._rng, c=UCB_C)

            try:
                model_path = Path(__file__).resolve().parent / MODEL_FILE
                with open(model_path, "rb") as f:
                    payload = pickle.load(f)
                    self._q_values = payload.get("q_values", {})
                    self._counts = payload.get("returns_count", {})
            except Exception:
                self._q_values = {}
                self._counts = {}

    def act(self, s: np.ndarray) -> int:
        board = s.copy()
        n_red = int((board == -1).sum())
        n_yel = int((board == 1).sum())
        current_player = -1 if n_red == n_yel else 1

        legal = _free_cols(board)
        if not legal:
            return 0

        wins = _winning_moves(board, current_player)
        if wins:
            return int(_ordered(wins)[0])

        blocks = _winning_moves(board, -current_player)
        if blocks:
            return int(_ordered(blocks)[0])

        safe = _safe_moves(board, current_player)
        candidates = safe if safe else legal

        state_key = _board_key(board, current_player)
        scored = []
        for a in candidates:
            q = self._q_values.get((state_key, a), None)
            n = self._counts.get((state_key, a), 0)
            if q is not None and n >= 3:
                center_bonus = 0.03 * (3 - abs(a - 3))
                scored.append((q + center_bonus, n, a))

        if scored:
            scored.sort(reverse=True)
            return int(scored[0][2])

        action = self._mcts.search(
            board,
            current_player,
            iterations=ITERATION_BUDGET,
            time_budget_s=TIME_BUDGET_S,
            allowed_actions=candidates if len(candidates) < len(legal) else None,
        )
        return int(action)