# SAVI: Sistema Autonomo de Valuacion Inmobiliaria

**End-to-end Machine Learning pipeline for Ames Housing: segmentation, price prediction and risk-aware decision making with Reinforcement Learning.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Reinforcement Learning](https://img.shields.io/badge/AI-Markov_Decision_Process-purple.svg)]()
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub_Pages-success.svg)](https://julian-rincon.github.io/ames-housing-ml/MDP_Ames_Presentacion.html)

> **Demo interactiva:** [ver el agente SAVI y la politica optima del MDP](https://julian-rincon.github.io/ames-housing-ml/MDP_Ames_Presentacion.html)

---

## Resumen Ejecutivo

La valuacion inmobiliaria tradicional depende de revision humana: es lenta, costosa y puede variar entre tasadores. Un modelo de regresion ayuda a estimar precios, pero no resuelve por si solo la pregunta de negocio mas importante: **cuando conviene confiar en la prediccion y cuando conviene escalar el caso a revision humana**.

SAVI aborda ese problema como un sistema de decision. El pipeline segmenta el mercado con clustering, estima precios con un modelo supervisado y usa un **Proceso de Decision de Markov (MDP)** para elegir entre tres acciones:

| Accion | Uso operativo |
|---|---|
| `APROBAR` | Automatizar la valuacion cuando el riesgo esperado es bajo. |
| `REVISAR` | Enviar a tasador humano cuando el costo esperado del error supera el costo de auditoria. |
| `RECHAZAR` | Solicitar mas informacion cuando el segmento tiene incertidumbre extrema o datos insuficientes. |

---

## Evolucion del Proyecto

Este repositorio conserva el trabajo original de analisis exploratorio, clustering y modelos supervisados. La actualizacion SAVI no reemplaza esa base: la extiende hacia una capa de decision autonoma.

| Etapa | Objetivo | Artefactos |
|---|---|---|
| Corte 1 / base ML | Segmentacion no supervisada y modelos predictivos sobre Ames Housing. | `notebooks/01` a `notebooks/08` |
| Corte 3 / SAVI | Agente MDP que decide aprobar, revisar o rechazar valuaciones segun riesgo. | `MDP_Ames_SAVI.py`, `MDP_Ames_Presentacion.html`, `Documentacion_SAVI.pdf` |

---

## Arquitectura Tecnica

### 1. Segmentacion del mercado

Se utiliza **K-Means** para convertir propiedades en estados del entorno. Cada estado representa un segmento del mercado con comportamiento economico similar.

En la version SAVI:

- `k = 6` estados.
- Variables numericas escaladas con `StandardScaler`.
- Validacion con `silhouette_score`.

### 2. Prediccion supervisada

El pipeline entrena un modelo **XGBoost Regressor** para estimar `SalePrice`. El cluster calculado por K-Means se incorpora como feature contextual, conectando la segmentacion con la prediccion.

Resultado reportado por el script SAVI:

| Modelo | R2 | MAE |
|---|---:|---:|
| XGBoost + cluster feature | 0.9606 | $26,254 |

### 3. Decision con Reinforcement Learning

El problema se formula como un **MDP**:

- **Estados:** clusters de propiedades.
- **Acciones:** `APROBAR`, `REVISAR`, `RECHAZAR`.
- **Recompensas:** costo economico esperado segun error relativo de prediccion y accion tomada.
- **Transiciones:** probabilidades historicas entre estados.
- **Algoritmo:** Value Iteration.

Parametros principales:

| Parametro | Valor |
|---|---:|
| `gamma` | 0.95 |
| `theta` | 0.0001 |
| Convergencia | 259 iteraciones |

---

## Resultados SAVI

El agente aprende una politica operativa que combina automatizacion y control de riesgo:

| Resultado | Interpretacion |
|---|---|
| 93.1% `APROBAR` | Automatizacion directa en segmentos donde el modelo es suficientemente estable. |
| 6.9% `REVISAR` | Revision humana selectiva en segmentos con mayor incertidumbre. |
| Casos `RECHAZAR` | Proteccion frente a segmentos atipicos o con informacion insuficiente. |

La decision no se basa solo en el precio estimado, sino en el costo esperado de equivocarse. Ese punto es clave para aplicaciones PropTech, credito hipotecario, underwriting inmobiliario y automatizacion de tasaciones.

---

## Resultados Previos del Repositorio

La base original del proyecto incluye analisis no supervisado y supervisado sobre Ames Housing extendido 2006-2024.

### Dataset

| Propiedad | Valor |
|---|---:|
| Filas | 20,203 |
| Columnas | 81 |
| Periodo | 2006-2024 |
| Variable objetivo | `SalePrice` |
| Fuentes | Kaggle + City of Ames Assessor 2024 |

> Nota: el dataset no se versiona por tamano. Las instrucciones estan en `data/README.md`.

### Modelos supervisados previos

| Modelo | Tarea | R2 / metrica principal | MAE |
|---|---|---:|---:|
| LightGBM | Regresion | 0.75 | ~$23,600 |
| Random Forest | Regresion | 0.3349 | $47,546 |
| Decision Tree | Regresion | 0.3276 | $48,084 |
| KNN | Regresion | 0.2509 | $51,562 |
| Linear Regression | Regresion | 0.0767 | $60,076 |
| SVM | Clasificacion binaria | Accuracy 95% | F1 0.9398 |

### Clustering previo

| Algoritmo | Configuracion / resultado |
|---|---|
| K-Means | Segmentacion con Elbow, Silhouette y Dunn Index. |
| Hierarchical Clustering | Comparacion Ward, Complete y Average. |
| DBSCAN | Deteccion de clusters, ruido y outliers de alto valor. |
| PCA | PC1=41.6%, PC2=13.5%, total=55.0%. |

---

## Estructura del Repositorio

```text
.
├── MDP_Ames_SAVI.py              # Pipeline SAVI: K-Means + XGBoost + MDP + Value Iteration
├── MDP_Ames_Presentacion.html    # Demo visual interactiva para GitHub Pages
├── Documentacion_SAVI.pdf        # Documento tecnico del agente SAVI
├── Taller1_Corte3_Final.docx     # Fuente editable de la documentacion
├── notebooks/                    # Trabajo original de clustering y modelos supervisados
├── data/                         # Instrucciones para ubicar el dataset local
├── requirements.txt              # Dependencias Python
└── README.md
```

---

## Como Ejecutar

```bash
git clone git@github.com:Julian-Rincon/ames-housing-ml.git
cd ames-housing-ml
pip install -r requirements.txt
```

Para ejecutar el pipeline SAVI, el script busca `ames_combined_2006_2024.csv` en este orden:

```text
AMES_DATASET_PATH
./ames_combined_2006_2024.csv
./data/ames_combined_2006_2024.csv
C:\Users\jrinc\Desktop\Aprendizaje de maquina\ames House Price\ames_combined_2006_2024.csv
```

Luego corre:

```bash
python MDP_Ames_SAVI.py
```

Para reproducir el trabajo previo, ubica `ames_combined_2006_2024.csv` en `data/` y abre los notebooks en orden:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08
```

---

## Autores

- **Julian Rincon** - [github.com/Julian-Rincon](https://github.com/Julian-Rincon)
- Valeria Larea
- Nicolas Garzon
- Juan Nino

*Universidad Sergio Arboleda - Machine Learning*
