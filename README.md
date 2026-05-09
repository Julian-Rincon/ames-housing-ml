# SAVI: Autonomous Real Estate Valuation System

**End-to-end Machine Learning pipeline for Ames Housing: market segmentation, price prediction, and risk-aware decision making with Reinforcement Learning.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Reinforcement Learning](https://img.shields.io/badge/AI-Markov_Decision_Process-purple.svg)]()
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub_Pages-success.svg)](https://julian-rincon.github.io/ames-housing-ml/MDP_Ames_Presentacion.html)

> **Interactive demo:** [explore the SAVI agent and its optimal MDP policy](https://julian-rincon.github.io/ames-housing-ml/MDP_Ames_Presentacion.html)

---

## Executive Summary

Traditional real estate valuation depends heavily on human appraisers. It is slow, expensive, and can vary across reviewers. A regression model can estimate a property price, but it does not answer the operational question that matters most: **when should the business trust the prediction, and when should it escalate the case to human review?**

SAVI frames valuation as a decision system. The pipeline segments the housing market with clustering, estimates prices with a supervised model, and uses a **Markov Decision Process (MDP)** to select one of three actions:

| Action | Operational Meaning |
|---|---|
| `APPROVE` | Automate the valuation when expected risk is low. |
| `REVIEW` | Send the case to a human appraiser when expected model error is more expensive than review. |
| `REJECT` | Request more information when the segment is too uncertain or underrepresented. |

---

## Project Evolution

This repository preserves the original exploratory analysis, clustering, and supervised learning work. The SAVI update does not replace that foundation; it extends it with an autonomous decision layer.

| Stage | Goal | Artifacts |
|---|---|---|
| ML foundation | Unsupervised segmentation and predictive modeling on Ames Housing. | `notebooks/01` to `notebooks/08` |
| SAVI decision layer | MDP agent that decides whether to approve, review, or reject valuations based on risk. | `MDP_Ames_SAVI.py`, `MDP_Ames_Presentacion.html`, `Documentacion_SAVI.pdf` |

---

## Technical Architecture

### 1. Market Segmentation

SAVI uses **K-Means** to convert properties into environment states. Each state represents a market segment with similar economic and structural behavior.

Current SAVI configuration:

- `k = 6` states.
- Numerical features scaled with `StandardScaler`.
- Cluster validation with `silhouette_score`.

### 2. Supervised Price Prediction

The pipeline trains an **XGBoost Regressor** to estimate `SalePrice`. The K-Means cluster is added as a contextual feature, connecting market segmentation with price prediction.

Latest run on the local Ames Housing dataset:

| Model | R2 | MAE |
|---|---:|---:|
| XGBoost + cluster feature | 0.9609 | $26,752 |

### 3. Reinforcement Learning Decision Layer

The business problem is modeled as an **MDP**:

- **States:** property clusters.
- **Actions:** `APPROVE`, `REVIEW`, `REJECT`.
- **Rewards:** expected financial outcome based on relative prediction error and action taken.
- **Transitions:** historical state transition probabilities.
- **Algorithm:** Value Iteration.

Core parameters:

| Parameter | Value |
|---|---:|
| `gamma` | 0.95 |
| `theta` | 0.0001 |
| Convergence | 259 iterations |

---

## SAVI Results

The agent learns a policy that balances automation and risk control:

| Result | Interpretation |
|---|---|
| 93.1% `APPROVE` | Direct automation in stable segments where the model is reliable. |
| 6.9% `REVIEW` | Selective human review in higher-uncertainty segments. |
| 0.03% `REJECT` | Additional information requested for rare or poorly represented cases. |

The decision is not based only on the predicted price. It is based on the expected cost of being wrong, which makes the approach relevant for PropTech, mortgage underwriting, real estate risk scoring, and automated appraisal workflows.

---

## Original Repository Results

The original project includes unsupervised and supervised analysis on the extended Ames Housing dataset.

### Dataset

| Property | Value |
|---|---:|
| Rows | 20,203 |
| Columns | 81 |
| Period | 2006-2024 |
| Target variable | `SalePrice` |
| Sources | Kaggle + City of Ames Assessor 2024 |

> The dataset is not committed because of size and reproducibility constraints. See `data/README.md`.

### Previous Supervised Models

| Model | Task | Main metric | MAE |
|---|---|---:|---:|
| LightGBM | Regression | R2 = 0.75 | ~$23,600 |
| Random Forest | Regression | R2 = 0.3349 | $47,546 |
| Decision Tree | Regression | R2 = 0.3276 | $48,084 |
| KNN | Regression | R2 = 0.2509 | $51,562 |
| Linear Regression | Regression | R2 = 0.0767 | $60,076 |
| SVM | Binary classification | Accuracy = 95% | F1 = 0.9398 |

### Previous Clustering Work

| Algorithm | Result |
|---|---|
| K-Means | Segmentation using Elbow, Silhouette, and Dunn Index. |
| Hierarchical Clustering | Comparison of Ward, Complete, and Average linkage. |
| DBSCAN | Detection of clusters, noise, and high-value outliers. |
| PCA | PC1 = 41.6%, PC2 = 13.5%, total = 55.0%. |

---

## Repository Structure

```text
.
├── MDP_Ames_SAVI.py              # SAVI pipeline: K-Means + XGBoost + MDP + Value Iteration
├── MDP_Ames_Presentacion.html    # Interactive visual demo for GitHub Pages
├── Documentacion_SAVI.pdf        # Technical paper for the SAVI agent
├── Taller1_Corte3_Final.docx     # Editable source document
├── notebooks/                    # Original clustering and supervised learning work
├── data/                         # Dataset placement instructions
├── requirements.txt              # Python dependencies
└── README.md
```

---

## How to Run

```bash
git clone git@github.com:Julian-Rincon/ames-housing-ml.git
cd ames-housing-ml
pip install -r requirements.txt
```

The SAVI script searches for `ames_combined_2006_2024.csv` in this order:

```text
AMES_DATASET_PATH
./ames_combined_2006_2024.csv
./data/ames_combined_2006_2024.csv
C:\Users\jrinc\Desktop\Aprendizaje de maquina\ames House Price\ames_combined_2006_2024.csv
```

Then run:

```bash
python MDP_Ames_SAVI.py
```

To reproduce the original notebook workflow, place `ames_combined_2006_2024.csv` in `data/` and open the notebooks in order:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08
```

---

## Authors

- **Julian Rincon** - [github.com/Julian-Rincon](https://github.com/Julian-Rincon)
- Valeria Larea
- Nicolas Garzon
- Juan Nino

*Universidad Sergio Arboleda - Machine Learning*
