# Ames Housing ML — Price Prediction & Segmentation (2006–2024)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-FF6600?style=for-the-badge&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Real estate price prediction and unsupervised segmentation on an extended Ames Housing dataset covering 2006–2024. Combines the original Kaggle dataset with the City of Ames Assessor's 2024 report, resulting in 20,203 properties and 81 features.

---

## Dataset

| Property | Value |
|----------|-------|
| Rows | 20,203 |
| Columns | 81 |
| Years covered | 2006 – 2024 |
| Target variable | SalePrice |
| Sources | Kaggle (2006–2010) + City of Ames Assessor 2024 |

> **Note:** 43 synthetic records with SalePrice ≈ $42.4M were detected and filtered during preprocessing.

The dataset is not committed to this repo. Download instructions in `data/README.md`.

---

## Key Results

### Supervised — Price Prediction

| Model | R² | MAE |
|-------|----|-----|
| LightGBM (final model) | 0.75 | ~$23,600 |
| Random Forest | — | — |
| SVM | — | — |
| Decision Tree | — | — |
| KNN | — | — |

### Unsupervised — Clustering

Applied to 12 numerical features: `OverallQual`, `OverallCond`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `YearBuilt`, `LotArea`, `SalePrice`, `TotRmsAbvGrd`, `Fireplaces`, `1stFlrSF`, `GarageCars`.

| Algorithm | Method | Optimal Clusters |
|-----------|--------|-----------------|
| K-Means | Elbow + Silhouette Score | — |
| Hierarchical | Ward, Complete, Average linkage | — |
| DBSCAN | eps/min_samples tuning | — |

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | kmeans_clustering | K-Means segmentation with Elbow and Silhouette analysis |
| 02 | hierarchical_clustering | Hierarchical clustering with multiple linkage methods |
| 03 | dbscan_clustering | DBSCAN with parameter tuning and noise analysis |
| 04 | clustering_comparison | Full side-by-side comparison of all 3 clustering methods |
| 05 | decision_tree_knn | Decision Tree and KNN classifiers with cross-validation |
| 06 | random_forest_svm | Random Forest and SVM with hyperparameter tuning |
| 07 | boosting_final_model | LightGBM final model — best results (R²=0.75, MAE≈$23K) |
| 08 | parcial1_complete_analysis | Complete end-to-end analysis |

---

## How to Run

```bash
git clone git@github.com:Julian-Rincon/ames-housing-ml.git
cd ames-housing-ml
pip install -r requirements.txt
```

Place `ames_combined_2006_2024.csv` inside the `data/` folder, then open any notebook in Jupyter.

Recommended order: 01 → 02 → 03 → 04 → 05 → 06 → 07

---

## Authors

- **Julian Rincón** — [github.com/Julian-Rincon](https://github.com/Julian-Rincon)
- Valeria Larea
- Nicolás Garzón
- Juan Niño

*Universidad Sergio Arboleda — Machine Learning Course, 2024*
