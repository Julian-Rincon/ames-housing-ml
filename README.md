# SAVI: Autonomous Real Estate Valuation System

**End-to-end Machine Learning pipeline for Ames Housing: market segmentation, price prediction, and risk-aware decision making with Reinforcement Learning.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Reinforcement Learning](https://img.shields.io/badge/AI-Markov_Decision_Process-purple.svg)]()
[![PyTorch](https://img.shields.io/badge/RL-PyTorch_DQN-EE4C2C.svg)](https://pytorch.org/)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub_Pages-success.svg)](https://julian-rincon.github.io/ames-housing-ml/SAVI_v2_ParcialFinal.html)

> **Interactive demo:** [explore the SAVI v2 full RL pipeline](https://julian-rincon.github.io/ames-housing-ml/SAVI_v2_ParcialFinal.html)
>
> **v1 demo:** [MDP Value Iteration presentation](https://julian-rincon.github.io/ames-housing-ml/MDP_Ames_Presentacion.html)

---

## Executive Summary

SAVI v2 extends the original MDP framework with a complete Reinforcement Learning pipeline. Starting from market segmentation with K-Means and price prediction with XGBoost (R²=0.9609), the system trains three RL agents in sequence: Value Iteration (MDP, convergence in 259 iterations), Q-Learning tabular (8,000 episodes, ε-greedy exploration), and a Deep Q-Network (PyTorch, Double DQN implicit architecture, 150 epochs, Adam optimizer). The final decision policy is determined by consensus across all three agents, with DQN as tiebreaker.

| Action | Operational Meaning |
|---|---|
| `APPROVE` | Automate the valuation when expected risk is low. |
| `REVIEW` | Send the case to a human appraiser when expected model error is more expensive than review. |
| `REJECT` | Request more information when the segment is too uncertain or underrepresented. |

---

## Project Evolution

This repository preserves the original exploratory analysis, clustering, and supervised learning work. The SAVI updates do not replace that foundation; they extend it with increasingly complete autonomous decision layers.

| Stage | Goal | Artifacts |
|---|---|---|
| ML foundation | Unsupervised segmentation and predictive modeling | `notebooks/01` to `notebooks/08` |
| SAVI v1 decision layer | MDP Value Iteration agent | `MDP_Ames_SAVI.py`, `MDP_Ames_Presentacion.html` |
| SAVI v2 full RL pipeline | Q-Learning + DQN + consensus policy + IEEE paper | `SAVI_v2_ParcialFinal.py`, `SAVI_v2_ParcialFinal.html`, `SAVI_v2_ArticuloIEEE.docx` |

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

### 3. Reinforcement Learning Pipeline (v2)

SAVI v2 trains three RL agents in sequence and combines their policies:

**3a. Value Iteration (MDP)**
- States: K-Means clusters (k=6)
- Actions: APPROVE, REVIEW, REJECT
- Convergence: 259 iterations (θ=0.0001, γ=0.95)

**3b. Q-Learning Tabular**
- Episodes: 8,000
- Learning rate α=0.1, ε-greedy decay from 1.0 → 0.05
- State space: same 6 clusters

**3c. Deep Q-Network (PyTorch)**
- Architecture: fully connected network on top-20 XGBoost features (continuous state vector)
- Double DQN implicit (online vs target network)
- 150 epochs, Adam lr=3e-4, batch=128, replay buffer=20,000
- ε decay: 1.0 → 0.05 over training

**3d. Consensus Policy**
Final policy = majority vote across VI, QL, DQN. DQN breaks ties.

---

## SAVI Results

| Result | Interpretation |
|---|---|
| XGBoost R²=0.9609 | Price prediction baseline |
| 259 VI iterations | MDP policy convergence |
| 8,000 QL episodes | Tabular agent training |
| 150 DQN epochs | Neural agent training |
| Consensus policy | Final APPROVE/REVIEW/REJECT per cluster |

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
├── SAVI_v2_ParcialFinal.py        # SAVI v2: full RL pipeline (VI + Q-Learning + DQN)
├── SAVI_v2_ParcialFinal.html      # Interactive v2 presentation for GitHub Pages
├── SAVI_v2_ArticuloIEEE.docx      # IEEE-format technical paper
├── MDP_Ames_SAVI.py               # SAVI v1: MDP + Value Iteration
├── MDP_Ames_Presentacion.html     # SAVI v1 interactive demo
├── Documentacion_SAVI.pdf         # v1 technical documentation
├── notebooks/
├── data/
├── requirements.txt
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
```

Then run the v2 pipeline:

```bash
python SAVI_v2_ParcialFinal.py
```

To run the original v1 MDP pipeline:

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
