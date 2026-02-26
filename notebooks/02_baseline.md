# Baseline Report — House Price Prediction

## Goal

The task is a regression problem: predict the sale price of a house (`SalePrice`) using the `house_train.csv` dataset.

---

## What Was Done

### 1. Loading Data and Splitting

The dataset was loaded from `house_train.csv`. The target variable is `SalePrice`, and all other columns are used as features. The data was split into train and validation sets: **80% / 20%**, `random_state=42`.

### 2. Feature Engineering and Preprocessing

Features were divided into three groups:

**Numeric (`numeric`)** — all numeric columns that were not placed in categorical groups:
- Imputation: median value (`SimpleImputer(strategy='median')`)
- Scaling: `StandardScaler`

**Ordinal categorical (`ordinal`, 20 features)** — for example `OverallQual`, `ExterQual`, `BsmtQual`, `KitchenQual`, `GarageFinish`, `Fence`, and others:
- Imputation: constant value `"NA"`
- Encoding: `OrdinalEncoder` with a clearly defined category order

**Nominal categorical (`nominal`, 25 features)** — for example `Neighborhood`, `MSZoning`, `GarageType`, `SaleCondition`, and others:
- Imputation: most frequent value
- Encoding: `OneHotEncoder(handle_unknown='ignore')`

The `Id` column was explicitly dropped. All transformations were combined in a `ColumnTransformer` with `remainder='drop'`.

### 3. Model Comparison (5-Fold Cross-Validation)

Five models were trained and evaluated on the train set using RMSE:

| Model | RMSE mean | RMSE std |
|---|---|---|
| **Ridge** | **33,844** | 7,056 |
| ElasticNet | 34,211 | 7,243 |
| Lasso | 35,945 | 7,191 |
| LinearRegression | 36,061 | 7,366 |
| DummyRegressor (baseline) | 78,860 | 4,112 |

> Lasso and ElasticNet produced `ConvergenceWarning` with the given parameters, meaning the models did not fully converge during training.

### 4. Selecting the Best Model

**Ridge Regression** with default parameters (`alpha=1.0`) showed the best result in cross-validation.

### 5. Evaluation on the Validation Set

The best model was trained on the full training set and then evaluated on the hold-out validation set:

| Set | RMSE | MAE | R² |
|---|---|---|---|
| Train | 23,516 | 15,666 | 0.907 |
| Valid | 30,872 | 19,423 | 0.876 |

### 6. Model Analysis

- A **SHAP summary plot** was built to understand which features had the most impact on predictions.
- A **residuals histogram** was plotted to check how the prediction errors are distributed.

---

## Summary

- The best baseline model is **Ridge Regression** with a standard preprocessing pipeline.
- Validation RMSE: **~30,872** ($), R²: **0.876**.
- The gap between train and validation RMSE (~7,000) is moderate — there is some overfitting, but it is within the expected range for a linear model.
- The DummyRegressor (predicts the median) gives an RMSE of ~78,860 — the real models are more than **2x better**.

### Ideas for Further Improvement

- Tune the Ridge hyperparameter (`alpha`) using GridSearch or RandomizedSearch
- Apply a log-transformation to the target variable, since house prices are usually skewed
- Create new features through feature engineering (e.g. house age, total area, etc.)