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
| LightGBM (defaults) | 2 outliers removed | 0.1281 ± 0.0094 |
| LightGBM (RandomizedSearchCV, 20 of 256 combos: 500 trees, lr 0.1, 4 leaves, min_child_samples 20) | 2 outliers removed | 0.1186 |
| **50/50 blend log-Ridge + XGBoost** (final) | 2 outliers removed | **0.1107** (out-of-fold) → Kaggle **0.12473** |

- Outliers = houses with GrLivArea > 4000 sq ft that sold under $300k (Ids 524, 1299). Both landed in one CV fold and doubled its error.
- Ridge beat plain Linear Regression because the overlapping columns (e.g. GarageCars / GarageArea) gave Linear Regression huge weights that cancelled each other out.
- Tuned XGBoost alone did not beat Ridge, but blending the two beat both.
- `TotalSF` (a sum of existing columns) added nothing, because Ridge can already compute that sum with its own weights.
- Logging the size columns helped, because size has diminishing returns (a curve) and Ridge can only fit straight lines. Logging all 20 skewed columns blindly did worse (0.1134).

Kaggle leaderboard score: **0.12473** (first submission 0.12689)

## What I learned / what failed
log1p(x)= log(1+x)

**Why does test have 80 columns?** It's missing `SalePrice`, the answer. Kaggle hides it and scores my predictions against it. (Titanic was the same: test had no `Survived`.)

**Why predict log(SalePrice)?**
- Errors become percentages: a $20k miss on a $60k house counts 10× more than on a $700k house.
- A few mansions stop dominating the model.
- A linear model gets percentage effects (a bathroom adds +8%, not a fixed $).
- No negative prices.
- Kaggle scores on log prices anyway.

**When should I use a log?** When:
- all values are positive
- the data is right-skewed (skew > 1)
- the values span 10× or more
- "% error" is what matters

Don't use it for negative values, symmetric data, fixed ranges (ratings 1–10), categories, left-skewed data, or when absolute $ matters. If unsure, try both and compare CV.

**Undo the log at the end:** `log1p` ↔ `expm1`. Kaggle wants dollars.

**What does one-hot encoding do?** It turns a text column into one 0/1 column per category. Label encoding (RL=0, RM=1, FV=2...) invents a fake order ("FV is twice RM"). One-hot has no order. `handle_unknown="ignore"` turns new test categories into all zeros instead of crashing.

**Why 303 columns?** 36 numeric columns stay as they are, and 43 text columns become 267 one-hot columns (one per category, with "None" counting as a category). 36 + 267 = 303. Test also gets 303 because the encoder reuses train's category list.

**NaN often means "doesn't have one":** PoolQC NaN = no pool. Fill it with "None", not the median, or you'd invent a pool.

**Pipeline format:** `Pipeline([("name", tool), ("name", tool)])`. It's a list of (sticker, tool) pairs, run top to bottom, and each step's output feeds the next. Fitting inside CV means each fold learns its own medians, so there's no leakage.

**ColumnTransformer:** sends numeric columns to one recipe and text columns to another. Any column it isn't told about is silently dropped.

**Outliers:** 2 huge houses (>4000 sq ft) sold cheap. Both were in fold 3 and doubled its error. Removing them took CV from 0.153 to 0.130, and the std from 0.047 to 0.014.

**Why is Ridge better than Linear Regression?** With overlapping columns (GarageCars/GarageArea), Linear Regression picks crazy weights like +25 and −15 that cancel out on training data but blow up on new data. Ridge doesn't allow big weights, so it splits the credit sensibly (+5 / +4) and holds up on new houses. `alpha` is the brake strength: too weak overfits, too strong underfits, and 10 was best.

**GridSearchCV:** tries every combination of the knobs, cross-validates each one, and keeps the winner retrained on all the data. `model__alpha` means "alpha inside the step called model" (two underscores). `best_params_` always names a winner, even when it's a coin flip, so check the full table.

**`cross_val_score` vs `.fit`:** `cross_val_score` = practice exams. It trains temporary models only to get a grade, then throws them away. `.fit` = studying everything for the real exam, which trains the one model you use to predict. GridSearch does both.

**XGBoost:** a relay team of trees, each fixing the previous trees' leftover mistakes (boosting). Random forest is a crowd voting independently. Tuned: 500 trees, learning_rate 0.05, depth 3. Simple and careful won.

**LightGBM:** the same relay idea, but it grows trees where the error is worst first, so it's faster. RandomizedSearchCV tries random combinations (20 of 256) instead of all of them.

**The fancy model doesn't automatically win:** Ridge beat tuned XGBoost and LightGBM on this data.

**Blending:** averaging Ridge and XGBoost beat both, because they make different kinds of mistakes. Adding LightGBM didn't help, because it thinks like XGBoost.

**Feature engineering:** `TotalSF` (a sum of existing columns) added nothing, because Ridge can already add columns itself. Logging the size columns helped, because size has diminishing returns (a curve) and Ridge can only draw straight lines. Logging all 20 skewed columns blindly did worse. Test must get the same transformation as train.

**Errors I made:**
- `np.expm1(a, b)` instead of `np.expm1(a + b)`: the comma made b the output storage, so prices came out around $400 and the Kaggle score was 6.01. Lesson: check that predictions look like real life before submitting.
- `strategy=` vs `scoring=`: "unexpected keyword argument" means the word before `=` is wrong.
- A loop body that wasn't indented ran only once, after the loop ended.
- `axes[0]` used twice drew both charts on top of each other.
- `np.log1p` without `(...)` stored the function itself instead of calling it.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
