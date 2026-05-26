# AgenT1000 — Agente híbrido para Connect-4

**Curso:** Fundamentos de Inteligencia Artificial  
**Reto:** Agente Connect-4  
**Agente final:** `AgenT1000` 
**Autor:** Carlos David Cruz 
**Branch:** <https://github.com/Leo07102004/Fundamentos-IA-Final-Project-CruzMontoyaPortocarrero/tree/Cruz>  
**Ruta del agente final:** `groups/Cruz/policy.py`

---

## 1. Descripción general

`AgenT1000` es un agente autónomo para jugar Connect-4. Su diseño combina tres componentes principales:

1. **Reglas tácticas inmediatas:** detectar jugadas ganadoras, bloquear amenazas directas del rival y evitar jugadas inseguras.
2. **Aprendizaje offline:** uso de una tabla `q_values.pkl` entrenada previamente mediante First-Visit Monte Carlo.
3. **Búsqueda online:** uso de Monte Carlo Tree Search (MCTS) como respaldo cuando el estado actual no tiene información confiable en la tabla de valores.

El objetivo del diseño es que el agente no dependa de una sola estrategia. Primero resuelve casos tácticos evidentes, después aprovecha conocimiento aprendido offline y, si no reconoce suficientemente el estado, realiza búsqueda en árbol durante la partida.

---

## 2. Archivos principales

La carpeta del agente debe contener, como mínimo:

```text
groups/
└── Cruz/
    ├── policy.py          # Código final del agente AgenT1000
    └── q_values.pkl       # Valores Q aprendidos offline

train_fvmc.py              # Script usado para entrenar q_values.pkl
CruzEntrega.ipynb              # Notebook de validación experimental y gráficas
```

Los archivos de la carpeta `connect4/`, `main.py`, `tournament.py` y demás archivos base no se modifican.

---

## 3. Requisitos

El proyecto usa Python y las siguientes librerías:

```bash
pip install numpy pandas matplotlib
```

El agente final usa únicamente:

```python
import math
import time
import pickle
import numpy as np
from pathlib import Path
from connect4.policy import Policy
```

---

## 4. Cómo ejecutar el agente

### 4.1. Uso normal dentro del entorno del reto

El runner del curso carga las clases que heredan de `connect4.policy.Policy`. El agente final está definido como:

```python
class AgenT1000(Policy):
    ...
```

Para usarlo en el torneo, debe estar ubicado en:

```text
groups/Group C/policy.py
```

Además, el archivo `q_values.pkl` debe estar en la misma carpeta que `policy.py`, porque el agente lo carga con:

```python
model_path = Path(__file__).resolve().parent / MODEL_FILE
```

Donde:

```python
MODEL_FILE = "q_values.pkl"
```

### 4.2. Reentrenar los Q-values

Si se desea regenerar la tabla aprendida, ejecutar:

```bash
python train_fvmc.py
```

Esto entrena el modelo con 20 000 episodios por defecto y genera:

```text
q_values.pkl
```

Ese archivo debe copiarse o mantenerse dentro de la carpeta donde está `policy.py`.

### 4.3. Ejecutar la validación experimental

Abrir y correr:

```text
CruzEntrega.ipynb
```

El notebook realiza pruebas contra:

- jugador aleatorio,
- AgentT600,
- AgenT1000 contra sí mismo,
- versión completa vs. versión sin Q-values.

También genera gráficas de resultados, duración promedio y tiempo promedio por jugada.

---

## 5. Entrenamiento offline: `train_fvmc.py`

El archivo `train_fvmc.py` entrena una tabla de valores Q usando **First-Visit Monte Carlo**.

### 5.1. Representación del tablero

El tablero se representa como una matriz de 6 filas por 7 columnas:

```python
ROWS, COLS = 6, 7
```

Los jugadores se codifican como:

```text
-1 = rojo
 1 = amarillo
 0 = casilla vacía
```

Cada estado se transforma en una llave hashable mediante:

```python
def board_key(board, player):
    return (tuple(board.flatten().tolist()), int(player))
```

Esto permite guardar valores para cada par `(estado, jugador_actual)`.

### 5.2. Política de comportamiento durante el entrenamiento

La función `choose_behavior_action` define cómo juega el agente durante la generación de episodios. Esta política no es completamente aleatoria; combina reglas y exploración:

1. Si existe una jugada ganadora inmediata, la juega.
2. Si el rival tiene una amenaza directa, la bloquea.
3. Con probabilidad `epsilon`, explora entre columnas centrales.
4. Si no explora, escoge la acción con mayor valor Q estimado más un pequeño bono por cercanía al centro.

El valor de exploración usado fue:

```python
epsilon = 0.20
```

Esto significa que aproximadamente 20% de las veces el entrenamiento intenta acciones exploratorias para descubrir estados nuevos, mientras que 80% de las veces explota lo que ya aprendió.

### 5.3. Generación de episodios

La función `generate_episode` simula una partida completa desde tablero vacío hasta victoria o empate. Durante la partida guarda una lista de transiciones:

```python
episode.append((board_key(state, player), action, player))
```

Cada elemento contiene:

- estado,
- acción tomada,
- jugador que tomó la acción.

Al final, se calcula el ganador con:

```python
w = winner(board)
```

### 5.4. Actualización First-Visit Monte Carlo

El entrenamiento recorre cada episodio y actualiza cada par estado-acción solo en su primera aparición dentro de la partida:

```python
if sa in seen:
    continue
seen.add(sa)
```

La recompensa final se asigna así:

```text
G =  1.0 si el jugador que tomó esa acción ganó la partida
G = -1.0 si ese jugador perdió
G =  0.0 si hubo empate
```

Después se actualiza el promedio acumulado:

```python
returns_sum[sa] += G
returns_count[sa] += 1
q_values[sa] = returns_sum[sa] / returns_count[sa]
```

Por eso `q_values[(estado, accion)]` representa el retorno promedio observado cuando se tomó esa acción desde ese estado.

### 5.5. Archivo generado

Al final se guarda un archivo pickle:

```python
payload = {
    "q_values": dict(q_values),
    "returns_count": dict(returns_count),
    "episodes": num_episodes,
}
```

Este archivo permite que el agente final use aprendizaje previo sin tener que entrenar durante el torneo.

---

## 6. Funcionamiento del agente final: `policy.py`

El agente final `AgenT1000` tiene el siguiente flujo de decisión:

```text
1. Determinar jugador actual
2. Obtener columnas legales
3. Buscar victoria inmediata
4. Bloquear victoria inmediata del rival
5. Filtrar jugadas inseguras
6. Consultar Q-values confiables
7. Si no hay Q-value confiable, usar MCTS
```

### 6.1. Determinar jugador actual

El agente no recibe explícitamente quién juega. Lo calcula contando fichas:

```python
n_red = int((board == -1).sum())
n_yel = int((board == 1).sum())
current_player = -1 if n_red == n_yel else 1
```

Como rojo empieza, si ambos tienen la misma cantidad de fichas, juega rojo. Si rojo tiene una ficha más, juega amarillo.

### 6.2. Jugadas legales

Una columna es legal si la celda superior está vacía:

```python
def _free_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]
```

### 6.3. Victoria inmediata

El agente revisa si alguna acción legal produce cuatro en línea. Si existe, la juega inmediatamente:

```python
wins = _winning_moves(board, current_player)
if wins:
    return int(_ordered(wins)[0])
```

Esto evita perder oportunidades obvias de ganar.

### 6.4. Bloqueo de amenaza rival

Si no puede ganar, revisa si el rival podría ganar en su próximo turno. Si existe esa amenaza, la bloquea:

```python
blocks = _winning_moves(board, -current_player)
if blocks:
    return int(_ordered(blocks)[0])
```

Esto reduce derrotas inmediatas contra jugadores tácticos.

### 6.5. Filtro de jugadas inseguras

Después calcula `safe_moves`, que son acciones que no dejan al rival con una victoria inmediata:

```python
safe = _safe_moves(board, current_player)
candidates = safe if safe else legal
```

Si hay jugadas seguras, limita la decisión solo a ellas. Si todas son riesgosas, usa todas las legales para no quedarse sin acción.

### 6.6. Uso de Q-values

El agente transforma el estado actual en una clave:

```python
state_key = _board_key(board, current_player)
```

Luego revisa si tiene valores Q para las acciones candidatas. Solo acepta un Q-value si fue observado al menos 3 veces durante el entrenamiento:

```python
if q is not None and n >= 3:
    center_bonus = 0.03 * (3 - abs(a - 3))
    scored.append((q + center_bonus, n, a))
```

La condición `n >= 3` evita confiar en acciones vistas muy pocas veces. El bono al centro favorece columnas centrales, que suelen ser estratégicamente más fuertes en Connect-4 porque participan en más líneas posibles.

Si existen acciones con Q-value confiable, el agente escoge la de mayor puntaje:

```python
scored.sort(reverse=True)
return int(scored[0][2])
```

### 6.7. Uso de MCTS como respaldo

Si no hay Q-values confiables, el agente usa Monte Carlo Tree Search:

```python
action = self._mcts.search(
    board,
    current_player,
    iterations=ITERATION_BUDGET,
    time_budget_s=TIME_BUDGET_S,
    allowed_actions=candidates if len(candidates) < len(legal) else None,
)
```

Los parámetros usados son:

```python
ITERATION_BUDGET = 350
TIME_BUDGET_S = 0.20
UCB_C = sqrt(2)
```

Esto significa que el agente busca hasta 350 iteraciones o hasta 0.20 segundos por jugada.

---

## 7. MCTS en el agente final

MCTS se usa para elegir una acción mediante simulaciones. Cada nodo contiene:

- `board`: estado del tablero,
- `player`: jugador que debe actuar en ese nodo,
- `children`: nodos hijos,
- `untried`: acciones no exploradas,
- `N`: número de visitas,
- `W`: recompensa acumulada.

El proceso de búsqueda tiene cuatro pasos:

### 7.1. Selección

Cuando un nodo ya está expandido, se escoge el hijo con UCB1:

```text
UCB = explotación + exploración
```

En el código:

```python
exploit = -child.q_hat
explore = c * sqrt(log_n / child.N)
score = exploit + explore
```

El término de exploración favorece nodos poco visitados. El término de explotación favorece nodos con buen valor estimado.

### 7.2. Expansión

Si el nodo tiene acciones no probadas, se crea un nuevo hijo aplicando una acción:

```python
action = node.untried.pop(0)
nb, _ = _drop(node.board, action, node.player)
child = _Node(nb, -node.player, parent=node, action=action)
```

Las acciones se ordenan por cercanía al centro.

### 7.3. Simulación

Desde el nuevo nodo se simula una partida hasta el final. La simulación no es totalmente aleatoria: también revisa victorias y bloqueos inmediatos antes de escoger al azar.

Esto hace que los rollouts sean más realistas que un juego completamente random.

### 7.4. Retropropagación

Cuando termina la simulación, la recompensa se propaga hacia arriba:

```python
node.N += 1
node.W += sign * reward_root
```

La recompensa vale:

```text
 1 si gana el jugador raíz
-1 si pierde el jugador raíz
 0 si empata
```

Al final de la búsqueda, el agente escoge la acción más visitada:

```python
return max(root.children.items(), key=lambda kv: kv[1].N)[0]
```

---

## 8. Validación experimental

La validación se hizo en `CruzEntrega.ipynb` con 100 partidas por experimento.

Resultados principales:

| Experimento | Resultado |
|---|---:|
| T1000 completo vs Random, T1000 rojo | 100% victorias |
| Random vs T1000 completo, T1000 amarillo | 100% victorias |
| T1000 completo vs T600, T1000 rojo | 85% victorias, 12% derrotas, 3% empates |
| T600 vs T1000 completo, T1000 amarillo | 88% victorias, 12% derrotas |
| Autojuego T1000 | amarillo 71%, rojo 24%, empate 5% |
| T600 vs T1000 sin Q-values, T1000 amarillo | 78% victorias |

En todos los experimentos se registraron 0 movimientos ilegales.

---

## 9. Interpretación

El agente cumple el requisito mínimo porque no pierde contra Random y gana el 100% de las partidas en ambos colores. También mantiene un desempeño alto frente a T600, que es un rival más fuerte por usar búsqueda MCTS.

La comparación con la versión sin Q-values muestra que el aprendizaje offline ayuda especialmente cuando el agente juega como amarillo contra T600, donde la tasa de victoria mejora de 78% a 88%. Sin embargo, el efecto no es uniforme en todos los escenarios, lo que sugiere que la tabla aprendida todavía puede mejorar.

---

## 10. Limitaciones y mejora futura

La principal limitación es la cobertura de la tabla `q_values.pkl`: si un estado no fue visto suficientes veces durante el entrenamiento, el agente no usa Q-values y depende de MCTS. Además, los resultados de autojuego muestran que existe asimetría entre colores.

Mejoras futuras propuestas:

- entrenar con más episodios,
- usar una estrategia epsilon decreciente,
- aumentar cobertura de estados mediante simetrías del tablero,
- ajustar los recursos de MCTS,
- entrenar contra oponentes más fuertes y no solo contra la política de comportamiento propia,
- calibrar mejor el peso del bono por columna central.

---

## 11. Resumen del flujo completo

```text
train_fvmc.py
    ↓
Genera partidas de autoentrenamiento
    ↓
Calcula retornos First-Visit Monte Carlo
    ↓
Guarda q_values.pkl
    ↓
AgenT1000 carga q_values.pkl en mount()
    ↓
En cada act(): reglas tácticas → Q-values confiables → MCTS fallback
    ↓
Devuelve la columna seleccionada
```
