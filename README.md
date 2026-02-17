# House Prices (Work in Progress)

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
│   ├── EDA.ipynb
│   └── EDA.md
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

### 2. Data Preprocessing



### 3. Feature Engineering



### 4. Modeling

Models evaluated:

* Linear Regression / Ridge / Lasso
* Random Forest Regressor
* Gradient Boosting (XGBoost / LightGBM)

Evaluation metric:

* **Root Mean Squared Log Error (RMSLE)**
