# Dataset

## Sources

### Original — Kaggle (2006–2010)
- Name: House Prices: Advanced Regression Techniques
- URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- Records: ~2,900 rows

### Extended — City of Ames Assessor 2024
- Source: City of Ames Assessor's Office official report
- Records added: ~17,300 rows (2011–2024)

## Combined Dataset
Both sources were merged into a single file: `ames_combined_2006_2024.csv`
- Total: 20,203 rows × 81 columns
- Years: 2006 to 2024

## How to get the data
1. Download the original from Kaggle (link above)
2. Place `ames_combined_2006_2024.csv` in this `data/` folder
3. Run notebooks in order starting from 01

## SAVI / MDP pipeline
The reinforcement learning pipeline in `MDP_Ames_SAVI.py` expects the CSV version of the combined
dataset at the repository root:

```text
ames_combined_2006_2024.csv
```

This file is intentionally not committed because datasets are ignored by `.gitignore`.

> Note: 43 synthetic records with SalePrice ≈ $42.4M were filtered during preprocessing.
