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

> Note: Models were trained on 6 features (YrSold, Utilities, LandContour, LotArea, Alley, OverallCond)
> using the original Kaggle dataset (1,460 rows). Train/Test split: 80/20 (1,168 train / 292 test).

| Model | Task | R² | MAE | RMSE |
|-------|------|----|-----|------|
| LightGBM (final model) | Regression | 0.75 | ~$23,600 | — |
| Random Forest (n=70, depth=10) | Regression | 0.3349 | $47,546 | $71,423 |
| Decision Tree (depth=5) | Regression | 0.3276 | $48,084 | $71,815 |
| KNN (k=15, Manhattan) | Regression | 0.2509 | $51,562 | $75,802 |
| Linear Regression | Regression | 0.0767 | $60,076 | $84,155 |
| SVM (RBF kernel, C=1) | Classification* | Acc: 95% | F1: 0.9398 | AUC-ROC: 0.9830 |

> *SVM was applied as a binary classifier (price above/below median). Best configuration: kernel=RBF, C=1.
> Random Forest best configuration: n_estimators=70, max_depth=10, min_samples_leaf=2.

### Unsupervised — Clustering

Applied to 12 numerical features on 20,043 clean records (160 outliers with SalePrice ≈ $42.4M filtered out).
Features: OverallQual, OverallCond, GrLivArea, TotalBsmtSF, GarageArea, YearBuilt, LotArea, SalePrice,
TotRmsAbvGrd, Fireplaces, 1stFlrSF, GarageCars.

| Algorithm | Optimal k | Best Silhouette | Method used |
|-----------|-----------|-----------------|-------------|
| K-Means | 4 | 0.3393 (at k=3) | Elbow + Silhouette + Dunn Index |
| Hierarchical — Ward | 3 | 0.3442 | Silhouette Score across k=2–10 |
| Hierarchical — Complete | 2 | 0.9413 | Silhouette Score across k=2–10 |
| Hierarchical — Average | 2 | 0.9413 | Silhouette Score across k=2–10 |
| DBSCAN | 3 clusters + noise | — | k-NN distance (elbow), eps=1.872, min_samples=10 |

**DBSCAN results:** 3 clusters formed, 111 noise points (0.5%). Cluster 0: 20,029 standard properties
(mean price $255K). Cluster 1: 42 synthetic records (SalePrice = $42.4M). Cluster 2: 21 large outlier
properties (mean area 12,557 sqft, mean price $1.44M). Noise points: 111 high-value irregular properties
(mean price $1.07M, mean area 4,555 sqft).

**K-Means cluster interpretation (k=4):**
- Cluster 4 — "Vivienda Económica": 6,460 properties (32.2%), mean price $126,209
- Cluster 1 — "Vivienda Estándar": 1,202 properties (6.0%), mean price $195,306
- Cluster 2 — "Vivienda de Alta Gama": 4,234 properties (21.1%), mean price $197,013
- Cluster 3 — "Vivienda Premium": remaining properties, mean price $368,043

**Hierarchical Ward cluster interpretation (k=3):**
- Cluster 1: 5,031 properties (25.1%), mean price $193,440, mean area 1,208 sqft, built ~1964
- Cluster 2: 6,768 properties (33.8%), mean price $134,513, mean area 1,028 sqft, built ~1999, no garage
- Cluster 3: 8,244 properties (41.1%), mean price $363,790, mean area 1,838 sqft, built ~1979

**PCA variance explained:** PC1=41.6%, PC2=13.5% (total 55.0%)

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
