# Experiment Report: Log Features + Ridge Regression
**Notebook:** `04_experiment_log_feats.ipynb`  
**Task:** Predicting house sale prices (regression on `SalePrice`)

---

## What Was Done

### 1. Starting Point — Baseline
Before the experiment, the metrics from the previous model version were recorded:

| Metric | Baseline |
|---|---|
| RMSE CV (mean) | 34 211 |
| RMSE CV (std) | 7 243 |
| RMSE train | 23 516 |
| RMSE valid | 30 872 |
| MAE valid | 19 423 |
| R² valid | 0.876 |

---

### 2. Feature Engineering — Splitting Numeric Features into Two Pipelines

Numeric features were divided into two groups:
- **`log_features` (18 features)** — features with a right-skewed distribution, transformed using `log1p` before scaling.
- **`numeric_features` (15 features)** — other numeric features, scaled with `StandardScaler` without log transformation.

Total features by type:
- 18 log-transformed numeric
- 15 regular numeric
- 21 ordinal categorical
- 25 nominal categorical (one-hot)

### 3. Preprocessing

A `ColumnTransformer` was built with separate branches:

| Branch | Transformation |
|---|---|
| Numeric (skewed) | `MedianImputer → log1p → StandardScaler` |
| Numeric (other) | `MedianImputer → StandardScaler` |
| Ordinal categorical | `OrdinalEncoder` with defined levels |
| Nominal categorical | `OneHotEncoder` |

### 4. Model

- **Ridge Regression** — linear regression with L2 regularization.
- The target variable `SalePrice` was also log-transformed using `TransformedTargetRegressor(func=log1p, inverse_func=expm1)`.

### 5. Validation

- Train/valid split: 80/20, `random_state=42`.
- Cross-validation: `StratifiedKFold(n_splits=5)`, stratified by deciles of `y_train` to preserve the price distribution across folds.

---

## Results

| Metric | Experiment | Baseline | Δ |
|---|---|---|---|
| RMSE CV (mean) | 33 524 | 34 211 | **−687** |
| RMSE CV (std) | 15 207 | 7 243 | +7 964 ⚠️ |
| RMSE train | 20 429 | 23 516 | **−3 087** |
| RMSE valid | **24 921** | 30 872 | **−5 951** |
| MAE valid | **15 164** | 19 423 | **−4 259** |
| R² valid | **0.919** | 0.876 | **+0.043** |

Validation RMSE dropped from **30 872 → 24 921 (~−19%)**, and R² improved from **0.876 → 0.919**.

---

## Error Analysis

### True vs Predicted
The model works well for mid-range prices, but **consistently underestimates expensive houses** (above $400k). A linear model cannot properly capture the tail of the distribution.

### Residuals vs Predictions
There is clear **heteroscedasticity**: the more expensive the house, the larger the error variance. A few outliers with residuals above $150k significantly increase the RMSE.

### Residual Distribution
Residuals are centered around zero, but there is a **right tail up to +$200k** — the model underestimates prices more often than it overestimates them.

---

## Feature Importance (Coefficients and SHAP)

Based on standardized Ridge coefficients and SHAP values, the most important features are:
- Quality ratings (`OverallQual`, `ExterQual`, `KitchenQual`)
- Area-related features (`GrLivArea`, `TotalBsmtSF`, `GarageArea`)
- Year built / remodeled

---

## Conclusions and Next Steps

**What worked:** Applying log transformation to skewed numeric features and the target variable gave a clear improvement on the validation set.

**Remaining issues:**
- High and unstable CV RMSE (std = 15k) — likely caused by price outliers in some folds.
- Systematic underestimation of expensive houses — a known limitation of linear models.

**What to try next:**
- Handling or removing price outliers to stabilize cross-validation.
- Additional feature engineering (feature interactions, polynomial features).