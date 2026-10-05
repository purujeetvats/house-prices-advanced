# Ames House Prices — Advanced Regression

Phase 3 of my ML learning roadmap: pipelines, hyperparameter tuning, gradient boosting, feature engineering.

## Problem
Given 79 features describing a house in Ames, Iowa (size, quality, year built, neighborhood, garage, basement...), predict its sale price.

## Data
Kaggle "House Prices - Advanced Regression Techniques": https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- `train.csv` — 1460 houses, with `SalePrice`
- `test.csv` — 1459 houses, no price (Kaggle scores these)
- `data_description.txt` — what every column means

Put the files in `data/` (they're not committed — download from Kaggle).

## Approach
_(fill in as you go)_

## Results
Metric = RMSE on log(SalePrice) (what Kaggle uses).

5-fold CV (`KFold(shuffle=True, random_state=42)`). All models run inside a Pipeline (median impute numerics, "None" + one-hot for categoricals → 303 columns).

| Model | Data | CV RMSE (mean ± std) |
|---|---|---|
| Baseline (predict median) | all 1460 | 0.3997 ± 0.0240 |
| Linear Regression | all 1460 | 0.1528 ± 0.0465 |
| Linear Regression | 2 outliers removed | 0.1301 ± 0.0135 |
| Ridge (alpha=10, GridSearchCV) | 2 outliers removed | 0.1149 ± 0.0083 |
| XGBoost (defaults) | 2 outliers removed | 0.1350 ± 0.0095 |
| XGBoost (tuned: 500 trees, lr 0.05, depth 3) | 2 outliers removed | 0.1197 |
| 50/50 blend Ridge + XGBoost | 2 outliers removed | 0.1114 (out-of-fold) → Kaggle 0.12689 |
| Ridge + `TotalSF` feature | 2 outliers removed | 0.1149 ± 0.0083 (no gain) |
| Ridge + log of GrLivArea, LotArea, 1stFlrSF | 2 outliers removed | 0.1119 ± 0.0076 |
| **50/50 blend log-Ridge + XGBoost** (final) | 2 outliers removed | **0.1107** (out-of-fold) → Kaggle **0.12473** |

- Outliers = houses with GrLivArea > 4000 sq ft that sold under $300k (Ids 524, 1299). Both landed in one CV fold and doubled its error.
- Ridge beat plain Linear Regression because the overlapping columns (e.g. GarageCars / GarageArea) gave Linear Regression huge weights that cancelled each other out.
- Tuned XGBoost alone did not beat Ridge, but blending the two beat both.
- `TotalSF` (a sum of existing columns) added nothing, because Ridge can already compute that sum with its own weights.
- Logging the size columns helped, because size has diminishing returns (a curve) and Ridge can only fit straight lines. Logging all 20 skewed columns blindly did worse (0.1134).

Kaggle leaderboard score: **0.12473** (first submission 0.12689)

## What I learned / what failed
log1p(x)= log(1+x)

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
