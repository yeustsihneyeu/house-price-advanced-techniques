# House Prices

## Overview

This project focuses on predicting residential property prices using the **House Prices: Advanced Regression Techniques** dataset from Kaggle.
The objective is to build a machine learning pipeline that performs data preprocessing, feature engineering, model training, and evaluation to accurately estimate house sale prices.

The task is a **supervised regression problem**, where the target variable is:

* **SalePrice** — the final sale price of each property.

## Dataset

Dataset: *House Prices: Advanced Regression Techniques* (Kaggle)

The dataset contains **80 explanatory variables** describing different aspects of residential homes, including:

* Property size and layout (LotArea, GrLivArea, TotalBsmtSF)
* Construction and materials (YearBuilt, OverallQual)
* Neighborhood information
* Garage, basement, and exterior features
* Sale conditions

Data files:

* `train.csv` – training data with target variable `SalePrice`
* `test.csv` – test data for predictions
* `data_description.txt` – full attribute description

## Project Structure

```
house-price-advanced-techniques/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
|   ├── 02_baseline.ipynb
|   ├── 03_experiment_log_target.ipynb
|   ├── 04_experiment_log_feats.ipynb
|   ├── 01_eda.md
|   ├── 02_baseline.md
|   ├── 03_experiment_log_target.md
│   └── 04_experiment_log_feats.md
│
├── models/
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis (EDA)

* Dataset overview
* Missing value analysis
* Detection of outliers
* Distribution analysis
* Correlation analysis
* Feature selections

### 2. Baseline

- The best baseline model is **Ridge Regression** with a standard preprocessing pipeline.
- Validation RMSE: **~30,872** ($), R²: **0.876**.
- The gap between train and validation RMSE (~7,000) is moderate — there is some overfitting, but it is within the expected range for a linear model.
- The DummyRegressor (predicts the median) gives an RMSE of ~78,860 — the real models are more than **2x better**.

### 3. Experiment (log transformation for target)

The `log1p(SalePrice)` transformation gives a clear improvement in model quality on the real validation set. However, the high instability in cross-validation needs further investigation — most likely, better handling of outliers in the target variable is needed, or a different cross-validation strategy (for example, removing extreme values before splitting).

### 4. Experiment (log transformation for features)

| Metric | Experiment | Baseline | Δ |
|---|---|---|---|
| RMSE | **24 921** | 30 872 | **−5 951** |
| MAE | **15 164** | 19 423 | **−4 259** |
| R² | **0.919** | 0.876 | **+0.043** |

- High and unstable CV RMSE (std = 15k) — likely caused by price outliers in some folds.
- Systematic underestimation of expensive houses — a known limitation of linear models.

### Modeling

Models evaluated:

* Linear Regression / Ridge / Lasso / ElasticNet

Evaluation metric:

* **Root Mean Squared Log Error (RMSLE)**
