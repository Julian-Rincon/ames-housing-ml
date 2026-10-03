# SAVI Agent API (RAG) — contrato de integración

Fuente de verdad para `api/` (servicio + agente + UI). Lo decide el integrador (Opus).
Complementa `CONTRACT.md` (pipeline). Cualquier duda: documentarla en el reporte, no inventar.

## Objetivo
Una **URL estable** (Lambda Function URL) que sirve:
- `GET /` → interfaz web (un solo `index.html`, sin build) para probar el agente.
- `/api/*` → API JSON del agente SAVI: consulta de parcelas, comparables, valuación what-if,
  decisión del agente RL (APROBAR / REVISAR / RECHAZAR), mercado, zonas y **chat RAG**.

## Restricciones duras
- **Lambda python3.12, sin numpy/pandas/torch/xgboost.** Toda la inferencia en Python estándar.
  Única dependencia externa permitida: `anthropic` (SDK oficial) para el chat con LLM.
- **Bedrock está BLOQUEADO en esta cuenta (Learner Lab).** El LLM es Claude vía API de Anthropic:
  `anthropic.Anthropic(api_key=...)`, modelo por env `SAVI_LLM_MODEL` (default `claude-opus-5`).
  Clave en SSM Parameter Store SecureString `/savi/anthropic_api_key` (o env `ANTHROPIC_API_KEY`).
  **Si no hay clave → modo determinista**: el agente igual recupera evidencia con las herramientas
  y redacta la respuesta con plantillas en español. La UI muestra qué modo respondió.
- Respuestas en **español**. Cifras en USD de hoy (`hpi_latest`, p.ej. 2026Q2).
- Nunca inventar datos: toda cifra de una respuesta debe venir de una herramienta (RAG).

## Artefactos que carga la API (ya generados por el pipeline)
Puntero: `s3://savi-processed-<acct>/serving/latest.json`
`{"run_id", "rag_prefix": "runs/<id>/rag/", "serving_prefix": "runs/<id>/serving/", "updated_utc"}`
En local: `SAVI_DATA_DIR=<dir>` con subcarpetas `rag/` y `serving/` (fixtures reales:
`out/local_run/rag/` y `out/model/serving/`).

### rag/ (motor CPU, `savi_rag_export.py`)
| Archivo | Contenido |
|---|---|
| `parcels.json.gz` | lista de 18,078 parcelas (padrón 2024). Campos: `pid, subdivision, map_area, neighborhood, lat, lon, geo_precision (exact/subdivision/map_area/none), classification, style, grade, condition_label, basement_type, garage_type, year_built, living_area, rooms_above, beds_above, beds_below, n_plumbing, n_fireplaces, land_value_2024, assessed_2024, assessed_today, avm_value_today, gap_vs_assessed, cluster, price_per_sqft_today, features[27]` (orden = `kmeans.json.features`) |
| `sales.json.gz` | 2,930 ventas reales De Cock: `pid, yr_sold, mo_sold, sale_date (YYYYMM), price, price_today, price_per_sqft_today, gr_liv_area, year_built, year_remod, overall_qual, overall_cond, bedrooms, full_bath, half_bath, garage_cars, total_bsmt_sf, lot_area, house_style, bldg_type, kitchen_qual, exter_qual, fireplaces, sale_condition, sale_type, neighborhood, lat, lon, in_training, avm_oof, avm_error, cluster, split` |
| `market.json` | `fhfa_hpi{source, series[{period,hpi}], latest, yoy_pct, since_2010_pct}`, `zillow_zhvi{source, series[{month,zhvi}], latest, yoy_pct}` |
| `areas.json` | `neighborhoods[]` (28 barrios: n_sales, median_price_today, median_ppsf_today, median_year_built, median_living_area, lat, lon, n_parcels, median_avm_today, median_assessed_2024), `subdivisions[]` (672), `clusters[]` (6) |
| `avm_model.json` | XGBoost: `base_score` (margen), `columns` (orden de entrada), `feature_index`, `trees` (dump JSON). Predicción = `expm1(base_score + Σ hoja)`. **Comparar en float32**: `f32(x) < f32(split_condition)` → `yes`, si no `no`; valor ausente → `missing`. Columnas: las 27 de `kmeans.features` + one-hot `area_<map_area>` + `cluster` |
| `kmeans.json` | `features[27]`, `mean`, `scale`, `centroids[6][27]` → cluster = argmin distancia euclídea en espacio estandarizado; `kmeans_distance` = esa distancia mínima |
| `dqn_state.json` | `top_features[20]` (columnas del AVM), `extra_signals = [log_avm_value=log1p(avm), gap_vs_assessed=clip((avm−assessed_today)/assessed_today,−1,1), kmeans_distance]`, `min`, `range` (MinMax; estado = clip((x−min)/range, 0, 1); range 0 → 0) |
| `manifest.json` | versión, run_id, hpi_latest, conteos, métricas del AVM |

### serving/ (motor GPU, `savi_gpu_sagemaker.py`)
| Archivo | Contenido |
|---|---|
| `decisions.json.gz` | `{pid: {q[3], margin, dqn, consensus, final}}` para las 18,078 parcelas |
| `dqn_weights.json` | `state_dict` de la QNetwork: `net.0` Linear(23→128), `net.1` BatchNorm1d (eval: `(x−running_mean)/sqrt(running_var+bn_eps)·weight+bias`), ReLU, `net.4` Linear(128→64), ReLU, `net.6` Linear(64→3). `weight` es `[out][in]` |
| `policy.json` | `selected_rule` (vi/ql/dqn_state/consensus_state/dqn_property/consensus_property/gated_qXX), `gated_tau`, `policy_vi/ql/dqn/final` (claves str "0".."5"), `rules` (reward train/val/test por regla), `test`, `reward_function`, `portfolio_action_share` |

**Regla de decisión para casas nuevas (what-if)**: aplicar `selected_rule` igual que el motor GPU:
`vi`/`ql`/`dqn_state`/`consensus_state` → política por cluster; `dqn_property` → argmax Q;
`consensus_property` → votación VI[s]+QL[s]+DQN (empate → DQN); `gated_qXX` → DQN si margen ≥ τ, si no consenso_property.

## Módulos (`cloud/api/`)
```
api/
├── savi_api/
│   ├── store.py       # carga artefactos (S3 vía latest.json, o SAVI_DATA_DIR), cache en memoria + /tmp
│   ├── inference.py   # XGBoost JSON (float32), KMeans, estado DQN, MLP forward, regla de decisión
│   ├── retrieval.py   # parcelas, búsqueda con filtros, comparables k-NN, zonas, mercado, documentos (BM25)
│   ├── tools.py       # TOOLS (schemas Anthropic) + run_tool(name, input) → dict JSON
│   ├── agent.py       # loop agéntico Claude (manual) + modo determinista
│   └── knowledge/     # *.md: metodología, fuentes de datos, diccionario de variables (BM25)
├── handler.py         # router Lambda Function URL (event payload v2.0)
├── static/index.html  # UI
├── build_lambda.py    # empaqueta zip (anthropic + deps para manylinux x86_64 py3.12)
└── local_server.py    # servidor de desarrollo (http.server) que invoca handler.handler
```

### Interfaz `tools.py` (la usan agent.py y handler.py)
```python
TOOLS: list[dict]            # [{"name","description","input_schema"}] formato Anthropic (strict-friendly)
def run_tool(name: str, tool_input: dict) -> dict   # nunca lanza: errores → {"error": "..."}
```
Herramientas (nombres exactos):
| name | input | output (resumen) |
|---|---|---|
| `get_parcel` | `{pid}` | parcela + decisión del agente (final, dqn, consensus, q, margin, explicación de la regla) |
| `search_parcels` | `{neighborhood?, subdivision?, min_value?, max_value?, min_area?, max_area?, min_year?, max_year?, beds_min?, style_contains?, decision?, sort_by?(avm_value_today/gap_vs_assessed/year_built/living_area), descending?, limit≤25}` | `{total_matches, results[]}` (campos legibles, sin `features`) |
| `find_comparables` | `{pid}` o `{features…}`, `k≤10` | ventas reales más parecidas (distancia estandarizada + geográfica si hay lat/lon): precio nominal y de hoy, fecha, barrio, lat/lon, similitud |
| `value_property` | `{pid?}` + overrides `{living_area, year_built, beds_above, grade_base, condition, neighborhood/map_area, ...}` | valuación AVM what-if, cluster, estado DQN, Q-values, decisión con la regla seleccionada, rango ±MAPE |
| `area_stats` | `{neighborhood?}` o `{subdivision?}` o `{cluster?}` | estadísticas agregadas |
| `market_trend` | `{since?: "YYYY"}` | serie FHFA/Zillow resumida + variaciones |
| `search_knowledge` | `{query}` | fragmentos de documentación (metodología, fuentes, recompensa, limitaciones) |
| `model_card` | `{}` | métricas AVM, política seleccionada, rewards val/test, split, fuentes |

### API HTTP (`handler.py`, Function URL payload 2.0)
| Método y ruta | Descripción |
|---|---|
| `GET /` | `static/index.html` |
| `GET /api/health` | `{status, run_id, hpi_latest, llm_mode: "claude"/"deterministic", counts}` |
| `GET /api/parcel/{pid}` | `get_parcel` |
| `GET /api/search?…` | `search_parcels` (query params) |
| `GET /api/comps/{pid}?k=` | `find_comparables` |
| `POST /api/valuate` | `value_property` (body JSON) |
| `GET /api/areas` | lista de barrios con stats (para mapa/selector) |
| `GET /api/market` | `market_trend` completo |
| `GET /api/model` | `model_card` |
| `POST /api/chat` | body `{message, history?[{role,content}]}` → `{answer, mode, tool_calls[{name,input,summary}], evidence{parcels[], sales[], areas[]}, model?}` |
CORS abierto (`*`) para GET/POST. Errores: JSON `{error}` con 400/404/500. Límite body 16 KB.
Coste: concurrencia reservada baja, `SAVI_MAX_AGENT_TURNS` (default 6), `max_tokens` acotado.

### Despliegue (lo integra `infra/deploy.py`)
Lambda `savi-api` (python3.12, 1024 MB, timeout 60 s, rol LabRole), Function URL `AuthType=NONE`
con CORS, env `SAVI_PROCESSED_BUCKET`, `SAVI_LLM_MODEL`, `SAVI_MAX_AGENT_TURNS`,
`SAVI_SSM_KEY_PARAM=/savi/anthropic_api_key`.
