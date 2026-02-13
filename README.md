# House Prices – EDA (Work in Progress)

## Project Goal
The goal of this project is to predict **SalePrice** (final house sale price).  
The evaluation metric is **RMSE**.

## Dataset
Dataset: Kaggle – House Prices: Advanced Regression Techniques  
The dataset contains **1460 rows** and **81 features**.

## Current Status
This repository contains the **first stage of the project — Exploratory Data Analysis (EDA)**.  
The analysis is **not finished yet**.

## What is Done
- Dataset loading and dataset overview
- Splitting features into **numerical** and **categorical**
- Missing value analysis
- Identification of features with many missing values
- Target (SalePrice) distribution analysis
- Log-transformation check for the target
- Numerical and categorical feature reports
- Initial recommendations:
  - Drop features with very high missing values
  - Remove `Id` column
  - Replace missing values for some features
  - Apply log transformation for skewed features
  - Create binary flags for some features
- Initial feature–target interaction analysis (correlation and mutual information)
- Feature–feature interaction analysis (correlation and VIF)

## Next Steps
Further preprocessing, feature engineering, and model training will be implemented later.
