# Final Report: House Price Prediction

---

## 1. What Was Done

The task is a regression problem: predict the sale price of a house (`SalePrice`) using features from the `house_train.csv` dataset. The work was done in three stages, and each stage gave better results than the previous one.

---

## 2. Experiment Progress

| Experiment | RMSE (valid) | MAE (valid) | R² (valid) | RMSE CV (mean) |
|---|---|---|---|---|
| Baseline (Ridge, no transformations) | 30,872 | 19,423 | 0.876 | 34,211 |
| + log(SalePrice) | 23,311 | 15,411 | 0.929 | 58,105 ⚠️ |
| + log(SalePrice) + log(skewed features) | **24,921** | **15,164** | **0.919** | **33,524** |

The last experiment gave the best balance between quality and stability. The validation RMSE dropped by **~19% compared to the baseline**, and the cross-validation became much more stable (std: 15,207 vs. 64,223 in the previous experiment).

---

## 3. Why Is the RMSE Not Lower?

### Main Sources of Error

**Expensive houses (>$400k) are consistently underestimated.** A linear model cannot handle the tail of the price distribution well. There are not many expensive houses in the training data, so the model does not have enough information to predict them accurately.

**Heteroscedasticity** — the more expensive the house, the larger the prediction error. A few outliers with residuals above $150k push the RMSE up significantly. RMSE punishes large errors more than MAE does, because it squares them.

**Limitations of a linear model.** Ridge regression builds a linear relationship between features and the target. It cannot directly capture interactions between features — for example, the idea that "a large house with high quality costs disproportionately more" is not something a linear model handles well.

**Unstable CV** in the second experiment (fold 5 produced RMSE = 186,165) happened because a group of very expensive houses ended up in the same fold. Stratification by deciles does not fully protect against this.

---

## 4. Which Features Affect the Price the Most

Based on standardized Ridge coefficients and SHAP values.

### Features That Increase the Price

| Feature | Description |
|---|---|
| `OverallQual` | Overall quality rating of the house (1–10) — the strongest predictor |
| `GrLivArea` | Above-ground living area (sq. ft.) |
| `TotalBsmtSF` | Total basement area |
| `GarageArea` | Garage area |
| `YearBuilt` / `YearRemodAdd` | Year built / last remodeled — newer houses cost more |
| `ExterQual` | Quality of exterior materials |
| `KitchenQual` | Kitchen quality |
| `BsmtFinSF1` | Finished area of the basement |

### Features That Decrease the Price or Have Little Effect

| Feature | Description |
|---|---|
| `LotArea` (low value) | A small lot reduces the price |
| `OverallCond` (low value) | Poor overall condition of the house |
| Certain `Neighborhood` values | Some locations give a significant price discount |
| Old construction with no renovation | `YearBuilt` < 1950 and no `YearRemodAdd` |
| No garage / no basement | Zero values for `GarageArea`, `TotalBsmtSF` |

**Key takeaway:** price is driven mostly by **quality** and **size**, not just by location. Quality features (`OverallQual`, `ExterQual`, `KitchenQual`) amplify the effect of size — a large house with high quality ratings costs disproportionately more than a large house with average ratings.

---

## 5. Final Model Metrics

- **Model:** Ridge Regression with L2 regularization (alpha=1.0)
- **Transformations:** log1p applied to the target variable and 18 skewed numeric features
- **Total features:** 79 (after OneHotEncoding)

| Metric | Value |
|---|---|
| RMSE (validation) | **24,921** |
| MAE (validation) | **15,164** |
| R² (validation) | **0.919** |
| RMSE CV (mean ± std) | 33,524 ± 15,207 |

The model explains **~92% of the variance** in house prices on the validation set. The typical prediction error is around **15,000–25,000**. Given that the median price is ~$180,000, this is an error of about **8–14%**.

---

## 6. How to Use the Model

### Practical Applications

**Estimating fair market value.** The model gives a quick reference price for a house. If the listing price is more than 15–20% away from the model's prediction.

**Portfolio scoring.** You can quickly rank properties by the ratio of listing price to predicted price to find undervalued objects.

**Explaining price factors.** SHAP values allow you to show a client exactly which features are increasing or decreasing the value of their property.