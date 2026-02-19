# Baseline ML Pipeline Report

> **Notebook:** `02_baseline.ipynb`  
> **Task:** House price prediction

---

## Overview

This report describes the baseline machine learning pipeline for predicting house prices. The goal was to build the first working end-to-end pipeline, measure its quality, and create a starting point for future improvements.

| Property | Value |
|---|---|
| **Task type** | Regression (predicting a numeric value — price) |
| **Target variable** | `SalePrice` — the sale price of a house |
| **Quality metric** | RMSE (Root Mean Squared Error) |

---

## 1. Loading and Splitting Data

### 1.1 Dataset

The dataset is loaded from `house_train.csv` using pandas. The dataset contains information about **1400+ houses** with **79 features** describing property characteristics.

### 1.2 Features and Target

- **X** — all features (79 columns without `SalePrice`)
- **y** — target variable `SalePrice`

### 1.3 Train/Validation Split

The data is split into training and validation sets in an **80/20 ratio** with `random_state=42` for reproducibility.

> **Why 80/20?**  
> This is the standard ratio for medium-sized datasets. 80% of data is used for training, and 20% is used for final evaluation on "unseen" data — which simulates real-world usage.

---

## 2. Feature Categories

A key step is classifying features by the type of processing needed, based on EDA (Exploratory Data Analysis) results.

### 2.1 Log-Transformed Features (`log_features`)

8 numeric features have a right-skewed distribution — log transformation normalizes them:

- `LotArea` — lot area
- `GrLivArea` — above-ground living area
- `TotalBsmtSF` — total basement area
- `MasVnrArea` — masonry veneer area
- `OpenPorchSF`, `WoodDeckSF`, `EnclosedPorch`, `ScreenPorch` — porch and deck areas

> **Why log transformation?**  
> Skewed features cause problems for linear regression. `log1p(x) = log(x+1)` transforms them closer to a normal distribution, which improves training and model interpretability.

### 2.2 Zero-Inflated Features (`zero_inflated_features`)

9 features contain a large share of zeros — not every house has these amenities (e.g., `WoodDeckSF`, `PoolArea`, `3SsnPorch`).

**Strategy:** for each such feature, a binary indicator `has_FeatureName` is created — `1` if the feature exists (value > 0), `0` if not. This lets the model distinguish between "no feature" and "small feature."

### 2.3 Ordinal Categorical Features (`ordinal_mapping`)

18 features with ordered categories (quality, condition) are encoded as numbers while preserving order. Examples:

| Feature | Categories (worst → best) |
|---|---|
| `ExterQual`, `ExterCond`, `HeatingQC`, `KitchenQual` | Po → Fa → TA → Gd → Ex (0–4) |
| `BsmtQual`, `BsmtCond` | NA → Po → Fa → TA → Gd → Ex (0–5) |
| `BsmtExposure` | NA → No → Mn → Av → Gd (0–4) |
| `BsmtFinType1`, `BsmtFinType2` | NA → Unf → LwQ → Rec → BLQ → ALQ → GLQ (0–6) |
| `GarageQual`, `GarageCond` | NA → Po → Fa → TA → Gd → Ex (0–5) |
| `GarageFinish` | NA → Unf → RFn → Fin (0–3) |
| `FireplaceQu` | NA → Po → Fa → TA → Gd → Ex (0–5) |
| `Functional` | Sal → Sev → Maj2 → Maj1 → Mod → Min2 → Min1 → Typ (0–7) |
| `LotShape` | IR3 → IR2 → IR1 → Reg (0–3) |
| `LandSlope` | Sev → Mod → Gtl (0–2) |
| `PavedDrive` | N → P → Y (0–2) |
| `OverallQual`, `OverallCond` | 1–10 (already numeric) |

### 2.4 Nominal Categorical Features (`nominal_features`)

18 features without natural order are processed with One-Hot Encoding (OHE). Examples: `Neighborhood`, `GarageType`, `SaleCondition`, `CentralAir`, and others.

### 2.5 Numeric Features Without Log (`numeric_no_log_features`)

6 numeric features with a near-normal distribution — processed with standard normalization only:

- `GarageCars` — garage capacity (number of cars)
- `Fireplaces` — number of fireplaces
- `HalfBath` — number of half bathrooms
- `BsmtFullBath`, `BsmtHalfBath` — basement bathrooms
- `MoSold` — month of sale

---

## 3. Custom Transformers

### 3.1 `ZeroInflationTransformer`

A custom transformer that inherits from `BaseEstimator` and `TransformerMixin` (scikit-learn standard). It creates binary presence indicators for zero-inflated features.

**How it works:**
1. Takes a matrix `X` with zero-inflated features as input
2. Creates a binary matrix: `1` where value ≠ 0, otherwise `0`
3. Combines (hstack) the original values and the indicators
4. Result: twice as many features (value + presence flag)

### 3.2 `DiffTransformer`

A custom transformer for creating "object age" features. It takes two columns (`x1`, `x2`) and computes their difference.

Creates three new features:
- `AgeHouse = YrSold - YearBuilt` — house age at time of sale
- `AgeRemodel = YrSold - YearRemodAdd` — time since last renovation
- `AgeGarage = YrSold - GarageYrBlt` — garage age

> **Why age instead of year built?**  
> The year of construction is tied to a specific era and has a non-linear relationship with price. House age is a more universal and interpretable feature: a new house is more expensive than an old one, regardless of the year of sale.

---

## 4. Preprocessing Pipelines

A separate `Pipeline` is built for each feature group. This ensures reproducibility, prevents data leakage, and makes it easy to apply transformations to new data.

| Pipeline | Steps | Applied to |
|---|---|---|
| `numeric_pipeline` | `SimpleImputer(median)` → `StandardScaler` | `GarageCars`, `Fireplaces`, `HalfBath`, `BsmtFullBath`, `BsmtHalfBath`, `MoSold` |
| `skewed_numeric_pipeline` | `SimpleImputer(median)` → `log1p` → `StandardScaler` | `LotArea`, `GrLivArea`, `TotalBsmtSF`, `MasVnrArea`, `OpenPorchSF`, `WoodDeckSF`, `EnclosedPorch`, `ScreenPorch` |
| `ordinal_categorical_pipeline` | `SimpleImputer(most_frequent)` → `OrdinalEncoder` (manual order) | 18 ordinal features |
| `nominal_categorical_pipeline` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` | 18 nominal features |
| `zero_inflated_pipeline` | `SimpleImputer(median)` → `ZeroInflationTransformer` | 9 zero-inflated features |
| `ages_pipeline` | `DiffTransformer` × 3 → `SimpleImputer(median)` → `StandardScaler` | `YrSold`, `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` |

---

## 5. Final Preprocessor

All 6 pipelines are combined into a single `ColumnTransformer`. The `remainder='drop'` parameter ensures that features from `drop_features` do not get into the model.

| Transformer | Input Features | Output Features |
|---|---|---|
| `numeric_pipeline` | 6 numeric features | 6 |
| `skewed_numeric_pipeline` | 8 skewed features | 8 |
| `ages_pipeline` | 4 year columns → 3 age features | 3 |
| `zero_inflated_pipeline` | 9 zero-inflated features | 18 (value + indicator) |
| `ordinal_categorical_pipeline` | 18 ordinal features | 18 |
| `nominal_categorical_pipeline` | 18 nominal features | ~80+ (depends on unique values) |

---

## 6. Model

### 6.1 Base Model: Linear Regression Pipeline

The base model is a `Pipeline` with two steps:
1. `preprocess` — `ColumnTransformer` with the preprocessor described above
2. `model` — `LinearRegression()`

Using `Pipeline` ensures that the preprocessor is trained only on train data, and the already-trained transformer is applied to validation/test data. This prevents data leakage.

### 6.2 Target Variable Transformation: `TransformedTargetRegressor`

`SalePrice` has a right-skewed distribution. To improve model quality, the target variable is log-transformed:

- `func=np.log1p` — before training: `y_train → log(y_train + 1)`
- `inverse_func=np.expm1` — during prediction: `log_pred → exp(pred) - 1`

This lets the model train in log space, where relationships are more linear, and automatically returns predictions to the original price scale.

---

## 7. Cross-Validation Results

5-Fold cross-validation is used for a reliable quality estimate:

- `n_splits=5` — data split into 5 parts
- `shuffle=True` — data is shuffled before splitting
- `random_state=42` — reproducibility
- Metric: `neg_root_mean_squared_error`

| Fold | RMSE |
|---|---|
| Fold 1 | 27,298.90 |
| Fold 2 | 32,333.56 |
| Fold 3 | 39,819.19 |
| Fold 4 | 23,754.97 |
| Fold 5 | 22,807.79 |
| **Mean** | **29,202.88** |

The spread between folds (22k–39k) shows that model quality depends on data composition. Fold 3 is the worst, which may indicate unusual objects in that subset.

---

## 8. Final Results

After cross-validation, the model is trained on the full train set and evaluated on the hold-out validation set (20%):

**Validation RMSE: 26,611.98**

Comparison with the naive `DummyRegressor` (strategy: `median` — always predicts the median price):

| Model | RMSE | Improvement |
|---|---|---|
| DummyRegressor (median) | 88,667.17 | — |
| **Linear Regression (our model)** | **26,611.98** | **−70.0%** |

> **RMSE interpretation:**  
> RMSE ~26,600 means the average prediction error is about $26,000. With a median house price of ~$163,000, this is roughly a **16% relative error**. For a linear model as a baseline — this is a good result.

---

## 9. Residual Analysis

### 9.1 True vs Predicted

- In the mid-price segment ($100k–$300k), the model works well — points are close to the diagonal
- For expensive houses (>$400k), the model systematically underestimates the price — points fall below the diagonal
- The linear model cannot capture the tail of the price distribution

### 9.2 Residuals vs Predictions

- Clear **heteroscedasticity**: error variance grows with price
- A few outliers with residuals >$150k significantly increase RMSE
- The model more often underestimates price than overestimates it (positive residuals are more common for expensive houses)

### 9.3 Residual Distribution

- Distribution is centered around 0 — no global systematic bias
- Right tail up to +$200k — the model significantly underestimates some expensive houses
- A slight right skew confirms the tendency to underestimate

---

## 10. Summary

### ✅ What Was Done

- Built a complete end-to-end ML pipeline from raw data to price prediction
- Implemented 6 different processing pipelines for different feature types
- Written 2 custom transformers (`ZeroInflationTransformer`, `DiffTransformer`)
- Applied log transformation to the target variable to improve training
- Completed 5-fold cross-validation: **mean RMSE = 29,202**
- **Validation RMSE = 26,611** — a **70% improvement** over the naive baseline (88,667)

### ❌ Limitations

- The linear model fails for expensive houses of more than 400k - systematic underestimation
- Heteroscedasticity in residuals
- A few outliers with errors more than 150k significantly hurt RMSE
- High variance across folds (22k–39k) indicates model instability

### 🔜 Next Steps

- Try non-linear models: Ridge, Lasso, ElasticNet
- Handle outliers (outlier detection and/or winsorization)
- Additional feature engineering based on residual analysis

---