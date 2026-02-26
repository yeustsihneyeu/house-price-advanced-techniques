# Experiment Report: Log Transformation of the Target Variable

## Goal

Test the hypothesis that applying a log transformation to the target variable `SalePrice` (house price) will improve the performance of a Ridge regression model compared to the baseline.

---

## Data

- **Dataset:** `house_train.csv` (House Prices prediction task)
- **Target variable:** `SalePrice`
- **Split:** 80% train / 20% validation (stratified split by deciles of the target variable)
- **Cross-validation:** StratifiedKFold, 5 folds, stratified by quantiles of `SalePrice`

---

## Model Architecture

An `sklearn` pipeline with three preprocessing branches:

| Feature type | Features | Processing |
|---|---|---|
| **Numerical** | `LotFrontage`, `LotArea`, `YearBuilt`, `BsmtFinSF1`, etc. | SimpleImputer (median) → StandardScaler |
| **Ordinal categorical** | `ExterQual`, `KitchenQual`, `OverallQual`, `BsmtQual`, etc. (21 features) | SimpleImputer (constant="NA") → OrdinalEncoder (with explicit category order) |
| **Nominal categorical** | `MSZoning`, `Neighborhood`, `SaleType`, etc. (25 features) | SimpleImputer (most_frequent) → OneHotEncoder |

**Model:** `Ridge` regression

---

## Key Change: Target Variable Transformation

The model is wrapped in `TransformedTargetRegressor`:
- **func:** `np.log1p` — applies a log transformation to `SalePrice` before training
- **inverse_func:** `np.expm1` — converts predictions back to the original scale

This corrects the right-skewed distribution of house prices, which is especially important for linear models.

---

## Results

### Comparison with Baseline

| Metric | Baseline (no transformation) | Experiment (log transformation) | Change |
|---|---|---|---|
| **RMSE_train** | 23,515 | 20,834 | −11.4% ✅ |
| **RMSE_valid** | 30,871 | 23,311 | −24.5% ✅ |
| **MAE_valid** | 19,423 | 15,411 | −20.6% ✅ |
| **R² (valid)** | 0.876 | 0.929 | +0.053 ✅ |
| **RMSE_cv mean** | 34,211 | 58,105 | +69.8% ⚠️ |
| **RMSE_cv std** | 7,243 | 64,223 | very high ⚠️ |

### Cross-Validation Scores by Fold

| Fold | RMSE |
|---|---|
| 1 | 22,442 |
| 2 | 35,677 |
| 3 | 23,719 |
| 4 | 22,521 |
| 5 | **186,165** ⚠️ |

---

## Conclusions

**Positive results:**
- The log transformation significantly improved performance on the hold-out validation set: RMSE dropped from ~30,871 to ~23,311 (−24.5%), and R² increased from 0.876 to 0.929.
- The gap between train and validation RMSE became smaller (from ~7,356 to ~2,477), which suggests reduced overfitting.

**Problem:**
- Fold 5 of cross-validation produced an abnormal RMSE of 186,165, which pulled the mean (58,105) and standard deviation (64,223) of the CV score up sharply. This signals instability: stratification did not prevent a group of houses with unusually high prices (outliers) from ending up in the same fold.

**Summary:** The `log1p(SalePrice)` transformation gives a clear improvement in model quality on the real validation set. However, the high instability in cross-validation needs further investigation — most likely, better handling of outliers in the target variable is needed, or a different cross-validation strategy (for example, removing extreme values before splitting).