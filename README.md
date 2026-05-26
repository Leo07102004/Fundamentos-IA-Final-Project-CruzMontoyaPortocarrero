# MontoyaAgent — Agente Connect-4 (Versión Final: V3)

## Idea principal

**MontoyaAgent** implementa Monte-Carlo Tree Search (MCTS) con los cuatro
conceptos del curso integrados: GPI con Q-values estimados por Monte Carlo
(clase 11), Alternating Markov Game de suma cero con propagación bipolar
(clase 12), exploración UCB (clase 12 slide 22) y Trial-Based Online Policy
Improvement con default policy heurística (clase 13).

**Diferencial clave respecto a otros agentes del grupo:**

| Aspecto | Agentes típicos | MontoyaAgent V3 |
|---|---|---|
| Representación interna | `np.ndarray` (copias) | Bitboards enteros (sin copias) |
| Detección de victoria | `_winner` O(n²) numpy | `_has_won` O(8 ops de bit) |
| Default policy | Rollout aleatorio | Heurística: gana→bloquea→score posicional |
| Lookup entre turnos | BFS O(n) o nulo | Tabla hash `(red,yel)→_Node`, O(1) |
| Pruning en expansión | Ninguno | Descarta movimientos suicidas |
| Parámetros de análisis | Fijos | `ucb_c` numérico + `warmup` booleano |

---

## Estructura del repositorio

```
mi_agente/
├── policy.py      ← MontoyaAgent (V3 final)
└── README.md      ← este archivo
```

Versiones anteriores (para comparación):
```
policy_v1.py      ← MCTS básico con numpy
policy_v2.py      ← MCTS con bitboards (MCTSAgent)
policy_v3.py      ← MontoyaAgent con bitboards + fork detection + doc completo
```

---

## Guía de uso

### Requisitos

```bash
pip install numpy
```

### Integración en el torneo

```
tournament/groups/mi_agente/policy.py
```

El sistema detecta automáticamente `MontoyaAgent` (hereda de `Policy`).

### Uso directo

```python
import numpy as np
from mi_agente.policy import MontoyaAgent

# Versión completa (torneo)
agent = MontoyaAgent(ucb_c=1.41, warmup=True)
agent.mount(28.0)                          # precalentamiento ~24.6s
board = np.zeros((6, 7), dtype=int)
col = agent.act(board)
print(f"Columna: {col}")

# Versión sin warmup (análisis)
agent_nw = MontoyaAgent(ucb_c=1.41, warmup=False)
agent_nw.mount(28.0)

# Versión greedy (análisis)
agent_gr = MontoyaAgent(ucb_c=0.0, warmup=True)
agent_gr.mount(28.0)
```

---

## Parámetros configurables

| Parámetro | Default | Descripción |
|---|---|---|
| `ucb_c` | `1.41` | Constante UCB (√2). 0 = puro greedy, >2 = explorador |
| `warmup` | `True` | Pre-pobla árbol en `mount()` antes de la partida |
| `default_budget` | `28.0` | Segundos si no se pasa argumento a `mount()` |

---

## Descripción técnica detallada

### Representación: Bitboards

Cada estado se codifica en dos enteros Python (64+ bits):
```
bit_position = col * 7 + fila_desde_abajo   (fila 0 = base)
```
La detección de victoria usa 4 operaciones de desplazamiento:
```python
m = mask & (mask >> 7)   # horizontal: 4 fichas en la misma fila
if m & (m >> 14): return True
```
Esto reemplaza el loop O(ROWS×COLS) de numpy, logrando **8-17× más iteraciones**
MCTS en el mismo presupuesto de tiempo.

### `mount(time_budget)` — Fase offline

1. Normaliza el argumento (float, int, tuple o None → gradescope-compatible).
2. Si `warmup=True`: dedica el 88% del presupuesto a self-play MCTS desde
   tablero vacío (Exploring Starts, clase 11), construyendo el árbol con N y Q.
3. Guarda `_game_start` para distribuir el tiempo restante entre los `act()`.

### `act(board)` — Fase online

Orden de prioridades:
1. **Victoria inmediata** — O(7 × 8 ops de bit). Si existe, la toma sin gastar tiempo MCTS.
2. **Bloqueo inmediato** — ídem para el oponente.
3. **MCTS online** — busca desde el nodo del árbol precalentado (lookup O(1)),
   añade simulaciones con el tiempo restante proporcional a movidas estimadas,
   devuelve la columna con mayor N (criterio robusto, clase 13 slide 26).

### MCTS interno: las 4 fases

**Selección:** baja por UCB (`sign × Q/N + C × √(log(N_parent)/N)`) hasta
un nodo con hijos no explorados o terminal.

**Expansión:** elige aleatoriamente de `untried`, excluye movimientos que
regalan victoria al oponente (pruning).

**Simulación (default policy):** rollout heurístico — gana si puede,
bloquea si el oponente gana, elige entre top-2 por score posicional.

**Retropropagación bipolar:** `r → -r → r → ...` subiendo al padre,
consistente con Alternating Markov Game de suma cero (clase 12 slide 17).

---

## Enlace al código

[Branch del estudiante](https://github.com/Leo07102004/Fundamentos-IA-Final-Project-CruzMontoyaPortocarrero/tree/Montoya)
