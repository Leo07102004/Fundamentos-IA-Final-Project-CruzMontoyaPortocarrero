import pickle
import random
from collections import defaultdict
import numpy as np

ROWS, COLS = 6, 7
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]


def free_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def drop(board, col, player):
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            nb = board.copy()
            nb[r, col] = player
            return nb, r
    return None, -1


def wins_at(b, r, c, p):
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


def winner(board):
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


def is_final(board):
    return winner(board) != 0 or all(board[0, c] != 0 for c in range(COLS))


def board_key(board, player):
    return (tuple(board.flatten().tolist()), int(player))


def winning_moves(board, player):
    ans = []
    for a in free_cols(board):
        nb, r = drop(board, a, player)
        if nb is not None and wins_at(nb, r, a, player):
            ans.append(a)
    return ans


def choose_behavior_action(board, player, epsilon, q_values):
    legal = free_cols(board)

    wins = winning_moves(board, player)
    if wins:
        return sorted(wins, key=lambda c: abs(c - 3))[0]

    blocks = winning_moves(board, -player)
    if blocks:
        return sorted(blocks, key=lambda c: abs(c - 3))[0]

    if random.random() < epsilon:
        ordered = sorted(legal, key=lambda c: abs(c - 3))
        top = ordered[:min(4, len(ordered))]
        return random.choice(top)

    key = board_key(board, player)
    best_a = None
    best_q = -10**9
    for a in legal:
        q = q_values.get((key, a), 0.0)
        bonus = 0.05 * (3 - abs(a - 3))
        score = q + bonus
        if score > best_q:
            best_q = score
            best_a = a

    if best_a is None:
        return random.choice(legal)
    return best_a


def generate_episode(q_values, epsilon=0.20):
    board = np.zeros((ROWS, COLS), dtype=int)
    player = -1
    episode = []

    while not is_final(board):
        state = board.copy()
        action = choose_behavior_action(state, player, epsilon, q_values)
        episode.append((board_key(state, player), action, player))
        board, _ = drop(board, action, player)
        player = -player

    w = winner(board)
    return episode, w


def train_fvmc(num_episodes=20000, epsilon=0.20, out_file="q_values.pkl"):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    q_values = {}

    for ep in range(1, num_episodes + 1):
        episode, w = generate_episode(q_values, epsilon=epsilon)
        seen = set()

        for state_key, action, player in episode:
            sa = (state_key, action)
            if sa in seen:
                continue
            seen.add(sa)

            if w == 0:
                G = 0.0
            elif w == player:
                G = 1.0
            else:
                G = -1.0

            returns_sum[sa] += G
            returns_count[sa] += 1
            q_values[sa] = returns_sum[sa] / returns_count[sa]

        if ep % 2000 == 0:
            print(f"Episodios: {ep}, estados-accion aprendidos: {len(q_values)}")

    payload = {
        "q_values": dict(q_values),
        "returns_count": dict(returns_count),
        "episodes": num_episodes,
    }

    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    print(f"Entrenamiento terminado. Archivo guardado en: {out_file}")


if __name__ == "__main__":
    train_fvmc(num_episodes=20000, epsilon=0.20, out_file="q_values.pkl")