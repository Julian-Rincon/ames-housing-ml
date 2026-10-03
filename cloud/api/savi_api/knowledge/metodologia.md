# Metodología del pipeline SAVI

## Arquitectura general

SAVI integra dos motores que corren en momentos distintos del pipeline:

- **Motor CPU** (`savi_cpu_pipeline.py`, EC2): integra las fuentes reales por número de
  parcela (PID), entrena el modelo de valuación automática (AVM) y el agente de política
  tabular (Value Iteration y Q-Learning).
- **Motor GPU** (`savi_gpu_sagemaker.py`, SageMaker Training Job): entrena un agente Double
  DQN sobre los mismos estados y produce la decisión final por parcela (consenso).

El agente SAVI no predice el precio de una casa por sí mismo: usa el AVM (XGBoost) como
estimador de valor y un agente de refuerzo (RL) para decidir qué hacer con esa estimación:
**APROBAR**, **REVISAR** o **RECHAZAR** la solicitud de valuación.

## AVM (modelo de valuación automática)

- Se comparan XGBoost y LightGBM con validación cruzada de 5 particiones (K-Fold), usando
  errores **out-of-fold** (OOF) para no sesgar las métricas ni las recompensas del agente RL.
- El objetivo es `log1p(precio_hoy)`; la predicción final es `expm1(base_score + Σ hojas)`.
- Se queda el modelo con mejor R²(log) en CV. En la corrida de referencia: XGBoost
  R²(log)=0.921, R²($)=0.927, MAE=$27,110, MAPE=7.8%; LightGBM R²(log)=0.912, MAE=$29,053.
- Las 27 variables numéricas (`NUM_FEATURES`) más un one-hot del área geográfica
  (`map_area`, los 4 primeros dígitos del PID) y el cluster K-Means forman la matriz de diseño.

## Estados del MDP (K-Means)

- Se agrupa la cartera 2024 completa (no sólo las ventas) en 6 clusters con K-Means
  (`k=6`, `n_init=15`), tras estandarizar las 27 features (media/desviación).
- El cluster asignado a cada parcela es el estado discreto que usan Value Iteration y
  Q-Learning. En la corrida de referencia el silhouette score es 0.294.
- La distancia euclídea al centroide más cercano, en el espacio estandarizado, se usa luego
  como señal de "atipicidad" en el estado continuo del DQN.

## MDP tabular: Value Iteration y Q-Learning

- Recompensa `R[s][a]`: promedio de la recompensa económica (ver `recompensa_decision.md`)
  de las ventas de ENTRENAMIENTO que caen en el cluster `s`, usando el error OOF del AVM.
- Transición `P[s][s']`: frecuencia empírica entre clusters de ventas **consecutivas en el
  tiempo** (orden cronológico real de las solicitudes), no el orden arbitrario de filas.
- Value Iteration converge con `γ=0.95`, `θ=1e-4`. Q-Learning tabular corre 8,000 episodios
  con `ε` decreciente por episodio y tasa de aprendizaje `α` decreciente por visitas a (s,a).
- En la práctica, ambas políticas tabulares convergen a la misma decisión por estado.

## Double DQN (estado continuo)

- Estado continuo (23 dimensiones): las 20 variables más importantes del AVM
  (`feature_importances_`, incluye variables numéricas, algunas áreas y el cluster) más 3
  señales: `log1p(valor_avm)`, brecha `(avm − avalúo) / avalúo` recortada a [-1, 1], y la
  distancia K-Means. Todo escalado con MinMax y recortado a [0, 1].
- Arquitectura: `Linear(23,128) → BatchNorm1d → ReLU → Dropout(0.2) → Linear(128,64) → ReLU
  → Linear(64,3)`. En inferencia (API) el Dropout no se aplica y BatchNorm usa las
  estadísticas acumuladas (`running_mean`/`running_var`), no las del lote.
- Es un **Double DQN real** (van Hasselt 2016): la red online elige la acción `a*` en el
  estado siguiente y la red target evalúa `Q(s', a*)`, para no sobreestimar el valor.
- El agente se entrena únicamente con el subconjunto TRAIN de ventas (70%); la regla de
  decisión final se selecciona con el subconjunto VAL (15%) y se reporta una única vez sobre
  TEST (15%), para evitar sesgo de selección.

## Selección de la regla de decisión

El motor GPU evalúa varias reglas candidatas sobre el reward medio en VAL y elige la mejor
(empate → la más simple, en orden de definición): política por estado de VI, de QL, del DQN
agregado por estado, consenso por estado (la del enunciado original), DQN por parcela,
consenso por parcela, y variantes "gated" (DQN por parcela sólo si su margen Q1−Q2 supera un
umbral τ, si no usa el consenso por parcela). La regla ganadora se guarda en
`serving/policy.json["selected_rule"]` y la API la aplica igual para valuaciones "what-if".
