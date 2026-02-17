# Exploratory Data Analysis (EDA) Report

## 1. Problem formulation

**What was done**

* Defined the ML task as a regression problem: predict **SalePrice** for a single house listing.

**Why**

* Clear problem definition determines the correct evaluation metric, modeling approach, and preprocessing strategy.

**Result**

* **Target:** `SalePrice`
* **Recommended metric:** RMSE (standard for pricing problems)

---

## 2. Data loading and split

**What was done**

* Loaded dataset (`house_train.csv`)
* Split:

  * `y = SalePrice`
  * `X = all other features`

**Why**

* Separating target from predictors prevents accidental target leakage during preprocessing.

**Result**

* `X.shape = (1460, 80)`
* Dataset contains both **numeric** and **categorical** features.

---

## 3. Dataset overview (types, missingness, uniqueness)

**What was done**

* Checked schema (`info()`)
* Calculated per-feature:

  * dtype
  * % missing
  * number of unique values
* Checked rows with extremely high missing ratio (>80%)

**Why**

* Understanding feature structure allows correct preprocessing decisions:

  * which columns to drop
  * which columns require imputation
  * which columns behave like categorical variables

**Result**

* No rows with >80% missing values
* High-missing columns include:

  * `PoolQC` ~99.5%
  * `MiscFeature` ~96%
  * `Alley` ~93.8%
  * `Fence` ~80.8%
  * `MasVnrType` ~59.7%
* Dropped `Id` column as non-predictive.

---

## 4. Feature type split

**What was done**

* Split features into:

  * Numeric
  * Categorical
  * Numeric treated as categorical (low unique count, including `MSSubClass`)

**Why**

* Different feature types require different preprocessing:

  * scaling for numeric
  * encoding for categorical
  * special handling for discrete numeric categories

**Result**

* Enabled separate preprocessing strategies for each feature group.

---

## 5. Target analysis (SalePrice)

**What was done**

* Computed descriptive statistics
* Checked skewness
* Visualized distribution
* Applied `log1p(SalePrice)` and IQR outlier detection

**Why**

* Target distribution strongly affects model performance.
* Log transformation stabilizes variance and improves linear model behavior.

**Result**

* Mean ~180,921
* Median 163,000
* Skewness ≈ 1.88 (strong right skew)
* Outliers:

  * Low: ~1.03%
  * High: ~0.89%
* Recommendation:

  * Use `log1p(SalePrice)`
  * Do not remove outliers (valid high-value houses)

---

## 6. Feature-level EDA

### 6.1 Categorical features

**What was done**

* Detected features with:

  * High missingness
  * Extreme dominance
  * Moderate dominance

**Why**

* High-missing features may carry little predictive value.
* Highly dominant categories provide minimal information.
* Missing values sometimes represent “absence” rather than unknown values.

**Result**

* High missing (drop or special handling):

  * `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`
* Highly imbalanced (drop candidates):

  * `Street`, `Utilities`, `Condition2`, `RoofMatl`, `Heating`
* Moderately dominant (rare-category flags):

  * `Functional`, `Electrical`, `CentralAir`, `PavedDrive`, `LandSlope`
* Basement/Garage missing values (~2.5–5.5%) → impute `"None"`:

  * `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`,
    `GarageQual`, `GarageCond`

---

### 6.2 Numeric features

**What was done**

* Identified:

  * Skewed features (log-transform)
  * Outlier-heavy features
  * Zero-inflated features (create flags)

**Why**

* Skewed distributions reduce model stability.
* Outliers may distort model coefficients.
* Zero-inflated variables often contain useful binary signals (presence/absence).

**Result**

* Log-transform candidates:

  * `LotFrontage`, `LotArea`, `MasVnrArea`, `GrLivArea`,
  * `OpenPorchSF`, `EnclosedPorch`, `ScreenPorch`, `MiscVal`,
  * `BsmtFinSF1`, `BsmtFinSF2`, `LowQualFinSF`
* Outlier-heavy:

  * `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`,
  * `OpenPorchSF`, `WoodDeckSF`
* Zero-inflated:

  * `MiscVal`, `ScreenPorch`, `3SsnPorch`, `EnclosedPorch`, `BsmtFinSF2`

---

## 7. Initial preprocessing prototype

**What was done**

* Built temporary preprocessing:

  * Dropped selected features
  * Numeric imputation → median
  * Categorical imputation → mode or `"None"`
  * Log transforms
  * Binary flags for sparse features

**Why**

* Early preprocessing prototypes help validate assumptions before final pipeline construction.

**Result**

* Generated candidate lists for transformations and imputations.

---

## 8. Feature ↔ Target interaction

### 8.1 Numeric predictors

**What was done**

* Calculated Pearson, Spearman correlations and Mutual Information.

**Why**

* Identifies which predictors provide the strongest predictive signal.

**Result**
Strong predictors:

* `GrLivArea`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`,
* `YearBuilt`, `YearRemodAdd`, `TotRmsAbvGrd`, `GarageYrBlt`

Supporting predictors:

* `LotArea`, `LotFrontage`, `MasVnrArea`, `BsmtFinSF1`,
* `OpenPorchSF`, `WoodDeckSF`

---

### 8.2 Categorical predictors

**What was done**

* Calculated mutual information for categorical features.

**Why**

* Captures non-linear relationships between categorical variables and the target.

**Result**
Strong predictors:

* `OverallQual`, `Neighborhood`, `GarageCars`, `ExterQual`, `KitchenQual`,
* `BsmtQual`, `MSSubClass`, `FullBath`, `GarageFinish`, `Foundation`, `HeatingQC`

---

### 8.3 Low-signal features

**What was done**

* Identified features with consistently low signal across metrics.

**Why**

* Removing weak predictors reduces model complexity and noise.

**Result**
Numeric:

* `BsmtFinSF2`, `LowQualFinSF`, `3SsnPorch`, `MiscVal`, `MoSold`, `YrSold`

Categorical:

* `BsmtFinType2`, `BsmtHalfBath`, `Functional`, `LandSlope`

---

## 9. Feature ↔ Feature interaction (multicollinearity)

**What was done**

* Built correlation matrix
* Selected features with correlation > 0.5
* Calculated VIF
* Created age-based features

**Why**

* Multicollinearity inflates variance of model coefficients and reduces interpretability.

**Result**

* Created:

  * `AgeHouse = YrSold - YearBuilt`
  * `AgeRemodeHouse = YrSold - YearRemodAdd`
  * `AgeGarage = YrSold - GarageYrBlt`
* VIF significantly reduced

---

# Final EDA Summary

## Key findings

1. Target is strongly right-skewed → log transformation improves modeling stability.
2. Dataset contains structured missingness patterns → missing often indicates absence.
3. Several numeric variables are skewed or zero-inflated → require log transforms and binary indicators.
4. Strong drivers of price:

   * House size
   * Quality indicators
   * Garage features
   * Neighborhood
5. Multicollinearity must be handled using age-based features to stabilize models.

---

# Preprocessing Plan

## 1. Target

* Apply `y = log1p(SalePrice)`
* Use `expm1()` during inference
  **Why:** reduces skew and improves regression performance

## 2. Drop columns

* Drop `Id`
* Drop:

  * `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `RoofMatl`,
    `Condition2`, `Street`, `Utilities`, `PoolArea`
    **Why:** extremely high missingness or near-zero predictive value

## 3. Missing value handling

* Numeric → median
* Categorical → `"None"` where feature absence is meaningful
* Other categorical → mode
  **Why:** prevents information loss and preserves structural meaning

## 4. Skewed numeric transforms

Apply `log1p()` to:

* `LotFrontage`, `LotArea`, `MasVnrArea`, `OpenPorchSF`,
* `EnclosedPorch`, `ScreenPorch`, `3SsnPorch`,
* `MiscVal`, `LowQualFinSF`, `BsmtFinSF2`
  **Why:** reduces skewness and improves model stability

## 5. Zero-inflated features

Create binary flags:

* `has_MiscVal`
* `has_ScreenPorch`
* `has_3SsnPorch`
* `has_EnclosedPorch`
* `has_BsmtFinSF2`
  **Why:** presence/absence often carries predictive signal

## 6. Multicollinearity fixes

Create:

* `AgeHouse = YrSold - YearBuilt`
* `AgeRemodel = YrSold - YearRemodAdd`
* `AgeGarage = YrSold - GarageYrBlt`
  **Why:** reduces multicollinearity and improves model interpretability

