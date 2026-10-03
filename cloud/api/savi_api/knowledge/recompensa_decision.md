# Función de recompensa y significado de las decisiones

## Las tres acciones del agente

El agente SAVI no decide el precio de una propiedad: decide **qué hacer** con la valuación que
produjo el AVM (XGBoost) para esa parcela.

- **APROBAR**: aceptar el valor del AVM tal cual, sin revisión humana adicional. Operativamente
  significa que la valuación sigue el flujo automático (p.ej. para una línea de crédito con
  garantía hipotecaria, o un avalúo masivo de cartera).
- **REVISAR**: enviar la parcela a un tasador humano antes de usar el valor del AVM. Tiene un
  costo (tiempo, dinero) pero evita aceptar o rechazar una valuación dudosa a ciegas.
- **RECHAZAR**: no usar el valor del AVM para esa parcela; se requiere información adicional o un
  proceso distinto (por ejemplo, demasiado atípica para el modelo entrenado).

## Función de recompensa

La recompensa es económica y se calcula a partir del **error relativo** del AVM respecto al
precio real de venta (fuera de muestra, out-of-fold, para no premiar sobreajuste):

- `APROBAR`: `+200` si error < 10% · `−500` si error < 25% · `−2000` si error ≥ 25%
  (aprobar una valuación muy errada es el peor desenlace posible).
- `REVISAR`: `−150` si error < 10% (se gastó tiempo de revisión sin necesitarla) ·
  `−50` en caso contrario (la revisión sí aportó valor, pero tiene un costo menor).
- `RECHAZAR`: `+50` si error > 20% (fue correcto pedir más información) ·
  `−200` en caso contrario (se rechazó una valuación que en realidad era buena).

Esta función de recompensa es la misma que usan Value Iteration, Q-Learning y el Double DQN
para aprender sus políticas; también es la que se usa para evaluar y comparar las reglas
candidatas de decisión en los conjuntos de validación (VAL) y prueba (TEST).

## Selección de la regla final y resultado en TEST

El motor GPU entrena el Double DQN sólo con el 70% de las ventas (TRAIN), elige entre varias
reglas candidatas la que da mejor recompensa media en el 15% de validación (VAL), y reporta el
resultado una única vez sobre el 15% restante (TEST) para que la métrica no esté sesgada por la
elección de la regla. El resultado concreto (recompensa media, intervalo de confianza bootstrap
al 95%, y comparación contra la política base de Value Iteration) se puede consultar con la
herramienta `model_card`, y depende de cada corrida del pipeline — no se debe asumir un número
fijo aquí; siempre se reporta el que viene en `serving/policy.json`.

## Consenso y "gating"

Varias de las reglas candidatas combinan los tres agentes (VI, QL, DQN) por votación mayoritaria
(si hay empate, gana el DQN, por ser el que ve el estado continuo más rico). Las reglas "gated"
(`gated_qXX`) sólo dejan decidir al DQN por parcela cuando está "seguro" — su margen entre la
mejor y la segunda mejor acción (Q1 − Q2) supera un umbral τ calculado como un percentil del
margen observado en validación — y si no, caen de vuelta al consenso por parcela, más
conservador. Esto evita que el agente actúe con confianza alta en estados poco frecuentes o
atípicos donde su estimación de Q es menos confiable.
