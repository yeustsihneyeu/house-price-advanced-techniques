# 🏠 House Prices — Exploratory Data Analysis Report

## Project Overview

**Goal:** Predict the final house sale price to support property valuation, pricing strategy, and investment analysis.  
**Dataset:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)  
**Target variable:** `SalePrice`  
**Evaluation metric:** RMSE

---

## Steps

### Step 1 — Dataset Overview

**What was done:** Loaded the dataset and examined its basic properties — shape, data types, missing values per column, and rows with too many missing values (threshold: 80%).

**Why:** To understand the scale of the data, identify obvious quality issues, and plan further analysis.

**Result:**
- Dataset contains **1,460 rows** and **80 features** (after removing `Id`).
- ⚠️ Small sample size — **risk of overfitting** should be considered during modeling.
- Features were split into three groups: **numeric**, **ordinal categorical**, and **nominal categorical**.
- The `Id` column was dropped as it carries no business information.
- A custom `dataset_overview_report()` function was built to show dtype, missing %, and unique counts per feature.

---

### Step 2 — Target Analysis

**What was done:** Explored the distribution of `SalePrice`: descriptive statistics, histogram, skewness check, log transformation, and outlier detection via IQR.

**Why:** Understanding the target distribution is critical for choosing the right model and loss function. Skewed targets can mislead linear models. Log-transforming the target often improves RMSE-based models.

**Result:**
- `SalePrice` ranges from **$34,900 to $755,000**, mean ≈ **$181,000**.
- The distribution is **right-skewed**.
- After applying `log1p` transformation, the distribution became approximately normal.
- Outliers in log-transformed target: ~**1.44% on the high end**, very few on the low end — acceptable, no immediate removal needed.
- **Recommendation:** Use `log1p(SalePrice)` as the modeling target.

---

### Step 3 — Feature Analysis (Quality Flags)

**What was done:** Built automated reports for numeric and categorical features. Each feature was analyzed for: skewness, outlier rate, missing rate, dominant category, rare categories, and constant values.

**Why:** To systematically detect data quality issues and prepare a list of preprocessing recommendations before modeling.

**Thresholds used:**
- Skew threshold: **2**
- Outlier threshold: **2%**
- Missing threshold: **30%**
- Rare category threshold: **1%**
- Dominant category threshold: **90%**

**Result:**  
Each feature received a set of **flags** (problems) and **recommendations** (actions). Features were grouped into actionable lists:
- Features needing **log transformation** (right skew)
- Features needing **median/mode imputation** (low missing %)
- Features needing **high-missing imputation or special treatment** (>30% missing)
- Features needing a **binary `Has_X` flag** (median = 0, mostly zeros)
- Features that are **drop candidates** (constant values)

---

### Step 4 — Feature–Target Interaction

**What was done:** Measured how much each feature relates to the target (`log(SalePrice)`) using **Pearson correlation**, **Spearman correlation**, and **Mutual Information (MI)**. For categorical features, an **ANOVA F-test** was also computed.

**Why:** To identify features that are actually informative for predicting the target and flag those that add no value.

**Result:**
- Numeric features with the strongest linear relationship: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`.
- Regression plots confirmed clear positive trends for top features.
- Categorical features with the highest MI: `Neighborhood`, `OverallQual`, `KitchenQual`, `ExterQual`, `GarageType`.
- Features with low MI across all metrics were flagged as **drop candidates**.

---

### Step 5 — Feature–Feature Interaction (Multicollinearity)

**What was done:** Computed a **Pearson correlation matrix** between numeric features and calculated the **Variance Inflation Factor (VIF)** for each.

**Why:** Multicollinearity inflates model variance and makes interpretation harder. Correlated features should be reduced or replaced.

**Result:**
- Several feature groups had high internal correlations (>0.5): area-related features (`GrLivArea`, `1stFlrSF`, `2ndFlrSF`), basement sub-features (`TotalBsmtSF`, `BsmtFinSF1`, `BsmtUnfSF`), year-related features.
- VIF analysis confirmed high multicollinearity in these groups.

**Feature engineering applied to reduce multicollinearity:**
- `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` → replaced by `AgeHouse`, `AgeRemodeHouse`, `AgeGarage` (age relative to `YrSold`).
- `1stFlrSF`, `2ndFlrSF` → dropped (covered by `GrLivArea`).
- `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF` → dropped (covered by `TotalBsmtSF`).
- `LotFrontage` → dropped (covered by `LotArea`).
- `TotRmsAbvGrd`, `GarageArea`, `KitchenAbvGr`, `BedroomAbvGr`, `FullBath` → dropped (redundant or covered by composite features).
- After dropping, VIF values improved significantly.

---

## Final Summary Table

# Preprocessing Plan

## Legend

| Action | Description |
|--------|-------------|
| **DROP** | Feature removed from final dataset |
| **ENGINEER** | New feature created; originals dropped |
| **IMPUTE+KEEP** | Fill missing values then use in model |
| **KEEP** | No transformation needed |

---

## DROP — Features Removed

| # | Feature | Type | Missing % | Skew / Dom% | Corr Target (Pearson/MI) | VIF / Multicollinearity | Preprocessing Step | Rationale |
|---|---------|------|-----------|-------------|--------------------------|-------------------------|-------------------|-----------|
| 1 | Id | Numeric/ID | 0% | — | — | — | Remove identifier | Pure row index — zero predictive value, causes data leakage if used |
| 2 | Utilities | Cat (nominal) | 0% | 99.9% same | MI≈0, F≈0 | — | Remove quasi-constant | Single-value dominant category (>99% 'AllPub'); no variance → zero information for model |
| 3 | Street | Cat (nominal) | 0% | 99.6% same | MI≈0 | — | Remove quasi-constant | 99.6% 'Pave'; essentially constant → adds noise, not signal |
| 4 | PoolQC | Cat (ordinal) | 99.5% | — | MI≈0 | — | Remove — extreme missings + low MI | 99.5% missing; even if imputed as 'None', near-zero target correlation — insufficient data to generalise |
| 5 | MiscFeature | Cat (nominal) | 96.3% | — | MI≈0 | — | Remove — extreme missings + low MI | 96.3% missing; miscellaneous features with no significant target correlation |
| 6 | Alley | Cat (nominal) | 93.8% | — | MI≈0 | — | Remove — extreme missings + low MI | 93.8% missing; presence of alley access has negligible effect on SalePrice |
| 7 | Fence | Cat (ordinal) | 80.8% | — | MI≈0 | — | Remove — extreme missings + low MI | 80.8% missing; fence type shows weak association with target |
| 8 | MiscVal | Numeric | 0% | skew≫2 | Pearson≈0.02, MI<0.01 | — | Remove — near-zero target correlation | Essentially zero Pearson & MI against log(SalePrice); mostly zeros; highly skewed |
| 9 | LotFrontage | Numeric | 17.7% | skew≈2.2 | Corr moderate but redundant | HIGH VIF with LotArea | Remove — multicollinearity with LotArea | Part of the lot dimension captured fully by LotArea; VIF>10; removing reduces collinearity without info loss |
| 10 | 1stFlrSF | Numeric | 0% | skew>2 | High | VIF>10 with GrLivArea | Remove — constituent of GrLivArea | GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF; component variables cause severe multicollinearity |
| 11 | 2ndFlrSF | Numeric | 0% | skew>2 | Moderate | VIF>10 with GrLivArea | Remove — constituent of GrLivArea | Same reason as 1stFlrSF; redundant given GrLivArea |
| 12 | BsmtFinSF1 | Numeric | 0% | skew>2 | Moderate | HIGH VIF with TotalBsmtSF | Remove — constituent of TotalBsmtSF | BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF = TotalBsmtSF; keeping parts alongside total inflates VIF |
| 13 | BsmtFinSF2 | Numeric | 0% | skew≫2 | Low | HIGH VIF with TotalBsmtSF | Remove — constituent + low MI | Redundant with TotalBsmtSF; additionally low target correlation |
| 14 | BsmtUnfSF | Numeric | 0% | skew>2 | Moderate | HIGH VIF with TotalBsmtSF | Remove — constituent of TotalBsmtSF | Same decomposition issue as BsmtFinSF1/2 |
| 15 | GarageArea | Numeric | 0% | skew≈1 | High (0.62) | VIF>10 with GarageCars | Remove — collinear with GarageCars | GarageCars and GarageArea are near-linear (r≈0.88); GarageCars carries slightly more MI; keep one |
| 16 | YearBuilt | Numeric | 0% | left skew | Moderate | HIGH VIF | Engineer → AgeHouse = YrSold − YearBuilt | Raw year is less interpretable and collinear with YearRemodAdd; age is stationary and more meaningful |
| 17 | YearRemodAdd | Numeric | 0% | left skew | Moderate | HIGH VIF with YearBuilt | Engineer → AgeRemodel = YrSold − YearRemodAdd | Correlated with YearBuilt; relative age better captures depreciation effect |
| 18 | GarageYrBlt | Numeric | 5.5% | left skew | Moderate | HIGH VIF | Engineer → AgeGarage = YrSold − GarageYrBlt | Same year-collinearity issue; age metric more informative and stationary |
| 19 | YrSold | Numeric | 0% | — | Low | Used only for engineering | Remove after engineering age features | Used solely as reference year to compute AgeHouse/AgeRemodel/AgeGarage; no standalone predictive value |
| 20 | TotRmsAbvGrd | Numeric | 0% | skew≈0.7 | High | HIGH VIF with GrLivArea | Remove — proxied by GrLivArea | Room count is strongly correlated with living area (r≈0.83); GrLivArea more precise; VIF>10 |
| 21 | KitchenAbvGr | Numeric | 0% | skew≈2.5 | Low | — | Remove — low MI + quasi-constant | 94% of houses have exactly 1 kitchen; near-constant; MI<0.01 |
| 22 | BedroomAbvGr | Numeric | 0% | skew≈0.2 | Low-Mod | VIF moderate with GrLivArea | Remove — weak MI, proxied by GrLivArea | GrLivArea already captures the effect of room count; BedroomAbvGr adds marginal signal |
| 23 | FullBath | Numeric | 0% | skew≈0.4 | Moderate | VIF with GrLivArea/BsmtFullBath | Remove — collinear with BsmtFullBath + GrLivArea | Bath count captured across basement and above-grade bath features; multicollinearity risk |
| 24 | Condition2 | Cat (nominal) | 0% | 97%+ same | MI≈0 | — | Remove — dominant + low MI | 97%+ 'Norm'; near-constant; ANOVA F-test not significant |
| 25 | RoofMatl | Cat (nominal) | 0% | 98%+ same | MI≈0 | — | Remove — dominant + low MI | >98% 'CompShg'; quasi-constant feature |
| 26 | Heating | Cat (nominal) | 0% | 97%+ same | MI≈0 | — | Remove — dominant + low MI | >97% 'GasA'; near-constant; negligible effect on target |

---

## ENGINEER — New Features Created

| # | Feature | Type | Missing % | Corr Target | Action | Preprocessing Step | Rationale |
|---|---------|------|-----------|-------------|--------|--------------------|-----------|
| 1 | AgeHouse *(new)* | Numeric (eng.) | 0% | Moderate-High | ENGINEER | Create: YrSold − YearBuilt | Age of house at time of sale is intuitive, stationary, and avoids raw-year multicollinearity |
| 2 | AgeRemodel *(new)* | Numeric (eng.) | 0% | Moderate | ENGINEER | Create: YrSold − YearRemodAdd | Time since last remodel captures renovation depreciation independently from original build year |
| 3 | AgeGarage *(new)* | Numeric (eng.) | 5.5% miss | Moderate | ENGINEER | Create: YrSold − GarageYrBlt; impute missing with AgeHouse | Captures garage age effect; houses without garages imputed logically from house age |
| 4 | Has_BsmtFinSF1 *(new)* | Binary (eng.) | 0% | Low-Mod | ENGINEER | Create binary: BsmtFinSF1 > 0 → 1 else 0 | Median=0; zero-inflation means presence/absence signal is more informative than raw area for sparse features |
| 5 | Has_2ndFlrSF *(new)* | Binary (eng.) | 0% | Moderate | ENGINEER | Create binary: 2ndFlrSF > 0 → 1 else 0 | Many single-storey homes; binary flag captures 'has 2nd floor' effect cleanly |
| 6 | Has_MasVnrArea *(new)* | Binary (eng.) | 0% | Moderate | ENGINEER | Create binary: MasVnrArea > 0 → 1 else 0 | Significant share of zero values; presence of masonry veneer is a quality signal |
| 7 | Has_WoodDeckSF *(new)* | Binary (eng.) | 0% | Low-Mod | ENGINEER | Create binary: WoodDeckSF > 0 → 1 else 0 | Majority zero; existence of deck adds value independently of its size |
| 8 | Has_OpenPorchSF *(new)* | Binary (eng.) | 0% | Low | ENGINEER | Create binary: OpenPorchSF > 0 → 1 else 0 | Zero-dominant; binary is more stable than raw area for sparse porch features |

---

## IMPUTE+KEEP — Retained with Preprocessing

| # | Feature | Type | Missing % | Skew / Dom% | Corr Target | VIF | Preprocessing Step | Rationale |
|---|---------|------|-----------|-------------|-------------|-----|--------------------|-----------|
| 1 | BsmtQual | Cat (ordinal) | 2.5% | — | High MI | — | Fill NA → 'None' (no basement) | NA means 'no basement', not actually missing; ordinal encode after imputation |
| 2 | BsmtCond | Cat (ordinal) | 2.5% | — | Moderate MI | — | Fill NA → 'None' | Same semantics as BsmtQual; absence should be encoded explicitly |
| 3 | BsmtExposure | Cat (ordinal) | 2.6% | — | Moderate MI | — | Fill NA → 'None' | No basement = no exposure; explicit 'None' level preserves information |
| 4 | BsmtFinType1 | Cat (ordinal) | 2.5% | — | Moderate MI | — | Fill NA → 'None' | Finish type of basement — NA is meaningful absence |
| 5 | BsmtFinType2 | Cat (ordinal) | 2.5% | — | Low-Mod MI | — | Fill NA → 'None' | Second basement finish type — same logic |
| 6 | FireplaceQu | Cat (ordinal) | 47.3% | — | High MI | — | Fill NA → 'None' (no fireplace) | High missing rate but NA = no fireplace; highly correlated with SalePrice via fireplaces count |
| 7 | GarageType | Cat (nominal) | 5.5% | — | Moderate MI | — | Fill NA → 'None' | No garage = 'None' type; important quality/location signal |
| 8 | GarageFinish | Cat (ordinal) | 5.5% | — | Moderate MI | — | Fill NA → 'None' | Finish quality only relevant if garage exists |
| 9 | GarageQual | Cat (ordinal) | 5.5% | — | Moderate MI | — | Fill NA → 'None' | NA = no garage; ordinal quality scale should include 'None' level |
| 10 | GarageCond | Cat (ordinal) | 5.5% | — | Moderate MI | — | Fill NA → 'None' | Same garage group imputation logic |
| 11 | MasVnrType | Cat (nominal) | 0.5% | — | Moderate MI | — | Fill NA → mode ('None') | Very few missings; mode imputation safe; type of masonry veneer is quality signal |
| 12 | MasVnrArea | Numeric | 0.5% | skew>2 | Moderate | — | Fill NA → 0; log1p transform | NA likely means no veneer (area = 0); right-skewed → log1p normalises distribution |
| 13 | Electrical | Cat (nominal) | 0.07% | — | Moderate MI | — | Fill NA → mode ('SBrkr') | Single missing value; mode imputation trivially safe |
| 14 | LotArea | Numeric | 0% | skew≫2 | Moderate | Low VIF after drops | log1p transform | No missings; heavy right skew → log1p stabilises variance and improves linearity with target |
| 15 | GrLivArea | Numeric | 0% | skew≈1.4 | High (0.73) | Moderate VIF | log1p transform | Strongest numeric predictor; mild skew → log1p recommended for RMSE-based models |
| 16 | TotalBsmtSF | Numeric | 0% | skew>2 | High (0.61) | — | log1p transform (after adding Has_ flag) | Strong predictor; right-skewed; log1p after zero-flag creation avoids log(0) |
| 17 | GarageCars | Numeric | 0.07% | skew≈0.2 | High (0.64) | — | Fill NA → 0 (no garage); keep as-is | Best garage-related predictor; low skew; strong MI |
| 18 | OverallQual | Cat→Ordinal | 0% | — | Highest (0.82) | Low | Treat as ordinal integer (already numeric 1–10) | Single most predictive feature; perfectly encoded as ordered integer |
| 19 | OverallCond | Cat→Ordinal | 0% | — | Low-Mod | — | Treat as ordinal integer (1–10) | Condition rating; lower target correlation than OverallQual but still meaningful |
| 20 | ExterQual | Cat (ordinal) | 0% | — | High MI | — | Ordinal encode: Po<Fa<TA<Gd<Ex | Strong quality signal for exterior material; explicit ordinal order preserves metric meaning |
| 21 | ExterCond | Cat (ordinal) | 0% | — | Low-Mod MI | — | Ordinal encode: Po<Fa<TA<Gd<Ex | Exterior condition; weaker than ExterQual but retains ordering information |
| 22 | HeatingQC | Cat (ordinal) | 0% | — | Moderate MI | — | Ordinal encode: Po<Fa<TA<Gd<Ex | Heating quality correlated with overall build quality |
| 23 | KitchenQual | Cat (ordinal) | 0% | — | High MI | — | Ordinal encode: Po<Fa<TA<Gd<Ex | Kitchen quality is a strong buying-decision signal |
| 24 | Functional | Cat (ordinal) | 0% | — | Moderate MI | — | Ordinal encode: Sal<Sev<Maj2<Maj1<Mod<Min2<Min1<Typ | Functionality deductions correlated with condition/price |
| 25 | LotShape | Cat (ordinal) | 0% | — | Low-Mod MI | — | Ordinal encode: IR3<IR2<IR1<Reg | Regularity of lot shape mild but consistent predictor |
| 26 | LandSlope | Cat (ordinal) | 0% | — | Low MI | — | Ordinal encode: Sev<Mod<Gtl | Keep; some signal; low cardinality → safe to ordinal-encode |
| 27 | PavedDrive | Cat (ordinal) | 0% | — | Moderate MI | — | Ordinal encode: N<P<Y | Driveway quality is a convenience/curb-appeal signal |
| 28 | MSSubClass | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode (treat as nominal despite integer coding) | Building class; no natural ordinal order; OHE or target-encode appropriate |
| 29 | MSZoning | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode | Zoning classification — important location quality signal |
| 30 | Neighborhood | Cat (nominal) | 0% | — | High MI | — | Target encode (mean SalePrice per neighbourhood) or OHE | 25 unique values; high cardinality → target encoding prevents OHE dimension explosion |
| 31 | Condition1 | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode | Proximity to roads/railroads — location quality factor |
| 32 | BldgType | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode | Type of dwelling (single family, duplex, etc.) — structural signal |
| 33 | HouseStyle | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode | Style of dwelling correlated with size/age |
| 34 | RoofStyle | Cat (nominal) | 0% | — | Low-Mod MI | — | One-hot encode | Some signal; 6 categories → manageable OHE |
| 35 | Foundation | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode | Foundation type correlates with age and build quality |
| 36 | GarageType | Cat (nominal) | 5.5% | — | Moderate MI | — | One-hot encode after fill 'None' | Already listed above in impute section — repeat for encode step |
| 37 | SaleType | Cat (nominal) | 0% | — | Low-Mod MI | — | One-hot encode | Type of sale may capture market condition effects |
| 38 | SaleCondition | Cat (nominal) | 0% | — | Low-Mod MI | — | One-hot encode | Abnormal sales can inflate/deflate price; important to encode explicitly |
| 39 | CentralAir | Cat (nominal) | 0% | — | Moderate MI | — | Binary encode: Y=1, N=0 | Binary feature; simple integer encoding sufficient |
| 40 | Exterior1st | Cat (nominal) | 0% | — | Moderate MI | — | One-hot encode (or target encode — 15 categories) | Material quality signal; moderate cardinality |
| 41 | Exterior2nd | Cat (nominal) | 0% | — | Low-Mod MI | — | One-hot encode | Correlated with Exterior1st but captures second material where applicable |
| 42 | LandContour | Cat (nominal) | 0% | — | Low-Mod MI | — | One-hot encode | Flatness of property — some influence on desirability |
| 43 | LotConfig | Cat (nominal) | 0% | — | Low-Mod MI | — | One-hot encode | Lot configuration (corner, cul-de-sac, etc.) |
| 44 | MoSold | Numeric | 0% | skew≈0 | Low | — | Convert to cyclical features: sin/cos(2π·month/12) | Month of sale — cyclic nature means January and December are adjacent; sin/cos encoding captures seasonality |
| 45 | OpenPorchSF | Numeric | 0% | skew≫2 | Low-Mod | — | Create Has_OpenPorchSF flag + log1p transform | Zero-dominant and right-skewed; binary existence flag + log1p for non-zero values |
| 46 | WoodDeckSF | Numeric | 0% | skew≫2 | Moderate | — | Create Has_WoodDeckSF flag + log1p transform | Same zero-inflation + skew pattern as OpenPorchSF |
| 47 | EnclosedPorch | Numeric | 0% | skew≫2 | Low | — | Create Has_ flag; consider keeping only flag | Majority zero; very low MI; flag captures presence; raw area likely noise |
| 48 | 3SsnPorch | Numeric | 0% | skew≫2 | Very low MI | — | Create Has_ flag; raw area likely drop after flag | Extremely sparse; almost all zeros; presence flag is the only useful signal |
| 49 | ScreenPorch | Numeric | 0% | skew≫2 | Low | — | Create Has_ flag + log1p if keeping | Sparse; binary flag most informative part |
| 50 | PoolArea | Numeric | 0% | skew≫2 | Low | — | Create Has_Pool flag; drop PoolArea (redundant with PoolQC dropped) | 0.6% have pools; flag captures presence; PoolQC dropped → PoolArea provides minimal additional info |
| 51 | Fireplaces | Numeric | 0% | skew≈0.6 | Moderate-High | — | Keep as-is (low skew, decent MI) | Count of fireplaces correlates well with size/quality of home |
| 52 | HalfBath | Numeric | 0% | skew≈0.7 | Moderate | — | Keep as-is | Half-bath count — modest predictor, low skew |
| 53 | BsmtFullBath | Numeric | 0% | skew≈1.1 | Moderate | — | Keep as-is | Basement full bathroom count — quality/comfort signal |
| 54 | BsmtHalfBath | Numeric | 0.07% | skew≫2 | Very low MI | — | Fill NA→0; consider Has_BsmtHalfBath flag | Very sparse (most = 0); flag may outperform raw count |
| 55 | LowQualFinSF | Numeric | 0% | skew≫2 | Very low MI | — | Create Has_ flag; consider dropping raw feature | 99%+ zero; extreme sparsity; only meaningful signal is existence |

---

## Key Takeaways

- The dataset is **small (1,460 rows)** — regularization and cross-validation are essential.
- Target `SalePrice` is **right-skewed** → use `log1p` transformation for modeling.
- Many categorical missing values (Alley, PoolQC, Fence, etc.) actually mean **"not present"** → impute with `"None"`.
- Several numeric features have **median = 0** and are mostly zero → create binary `Has_X` flags alongside log-transformed versions.
- High multicollinearity among area and year features was resolved through **feature engineering** (age-based features, dropping redundant sub-components).
- Most informative features for `SalePrice`: `OverallQual`, `GrLivArea`, `Neighborhood`, `KitchenQual`, `ExterQual`, `TotalBsmtSF`, `GarageCars`.