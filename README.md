# Crop Irrigation Need Prediction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Multiclass%20Classification-2E7D32?style=flat)
![Optimization](https://img.shields.io/badge/Tuning-Optuna-6A1B9A?style=flat)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-E91E63?style=flat)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

This project predicts crop irrigation need from agricultural, weather, and soil-related features. The notebook combines exploratory data analysis, feature engineering, cross-validation, model comparison, leaderboard-style scoring, and SHAP-based explainability.

The project is designed around a practical farming question: given current crop and environmental conditions, can a model estimate whether irrigation is needed?

## Repository Contents

| File | Description |
| --- | --- |
| `Crop-Irrigation-Need.ipynb` | Main notebook for EDA, feature engineering, model training, tuning, evaluation, and final comparison. |
| `images/` | Exported plots used in this README. |
| `README.md` | Project documentation. |

## Dataset

The notebook expects the following files in the project root:

| File | Purpose |
| --- | --- |
| `train.csv` | Training data with the target column `Irrigation_Need`. |
| `test.csv` | Test data used for prediction or leaderboard-style scoring. |

The training workflow separates features from the target:

```python
X = df_train.drop("Irrigation_Need", axis=1)
y = df_train["Irrigation_Need"]
```

## Workflow

1. Load training and test data.
2. Generate EDA summaries with Sweetviz and custom tables.
3. Review target distribution and feature types.
4. Engineer agricultural summary features (water input, dryness index, evaporative stress, soil health, pH deviation, moisture-rain ratio, group-wise soil-moisture deviations).
5. Clean numeric issues and encode categorical variables.
6. Tune LightGBM with Optuna under stratified cross-validation.
7. Train and compare multiple classification models.
8. Inspect feature importance and SHAP explanations.

## Feature Engineering

The notebook derives a small library of agronomic ratio features on top of the raw soil and weather columns:

| Engineered Feature | Definition |
| --- | --- |
| `Total_Water_Input` | `Rainfall_mm + Previous_Irrigation_mm` |
| `Dryness_Index` | `Temperature_C * Sunlight_Hours / (Rainfall_mm + 1)` |
| `Evaporative_Stress` | `Temperature_C * Sunlight_Hours * Wind_Speed_kmh / (Humidity + 1)` |
| `Soil_Health` | `Organic_Carbon * Soil_Moisture / (Electrical_Conductivity + 0.1)` |
| `Moisture_Retention` | `Soil_Moisture * Organic_Carbon` |
| `pH_Deviation` | `|Soil_pH − 6.5|` |
| `Moisture_Rain_Ratio` | `Soil_Moisture / (Rainfall_mm + 1)` |
| Group-wise deviations | `Soil_Moisture` deviation from the mean within each `Soil_Type`, `Season`, and `Crop_Growth_Stage`. |

## Modeling Approach

| Model | Role in Project |
| --- | --- |
| LightGBM Classifier | Primary tuned model, optimized with Optuna under stratified CV. |
| Gradient Boosting Classifier | Baseline boosted-tree model with feature engineering and tuning. |
| CatBoost Classifier | Boosting model suited for categorical-heavy tabular data. |
| Random Forest Classifier | Bagging-based tree ensemble comparison. |
| Logistic Regression | Linear benchmark for comparison. |

## Results Summary

The notebook records the following leaderboard-style scores:

| Model | Score |
| --- | ---: |
| LightGBM | 0.96602 |
| Gradient Boosting | 0.96098 |
| CatBoost | 0.95787 |

### LightGBM Feature Importance

The top-15 LightGBM importances surface the variables the tuned model relies on most. Raw weather and soil readings dominate, with engineered ratios such as `Moisture_Rain_Ratio` and `Total_Water_Input` appearing alongside them.

![LightGBM top 15 feature importances](images/lgbm-feature-importance.png)

### SHAP Explainability for the "High" Class

Tree feature importance shows *which* variables the model uses, but not *how*. SHAP analysis breaks down the model's decision for the "High" irrigation-need class on a sample of 1,000 rows.

The bar plot ranks features by mean absolute SHAP value — the average magnitude of each feature's contribution.

![SHAP bar plot, mean absolute value](images/shap-bar.png)

The beeswarm plot shows the *direction* and *spread* of each feature's effect. High temperatures and low soil moisture push the model toward predicting "High" irrigation need, which matches the agronomic intuition.

![SHAP beeswarm plot](images/shap-beeswarm.png)

A waterfall plot decomposes one individual prediction. Each arrow shows how a single feature shifts the model output from the dataset baseline (`E[f(X)] = -10.25`) to the final value (`f(x) = -11.88`).

![SHAP waterfall plot](images/shap-waterfall.png)

Across both global and local views, soil moisture, temperature, crop-growth-stage moisture summaries, mulching use, and wind speed emerge as the most influential predictors.

## Tools Used

| Category | Libraries |
| --- | --- |
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `sweetviz`, `shap` |
| Modeling | `scikit-learn`, `lightgbm`, `catboost` |
| Tuning | `optuna`, stratified cross-validation |
| Evaluation | `accuracy_score`, `f1_score`, `classification_report`, permutation importance, SHAP |

## How to Run

1. Clone the repository.
2. Add `train.csv` and `test.csv` to the repository root.
3. Install the required libraries.
4. Open and run `Crop-Irrigation-Need.ipynb`.

```bash
pip install pandas numpy matplotlib sweetviz scikit-learn optuna lightgbm catboost shap
```

## Data Note

The repository contains the notebook but not the CSV data files. The notebook will run after `train.csv` and `test.csv` are placed in the expected location.

## Future Improvements

- Add `requirements.txt` for reproducible installation.
- Save generated reports and model-comparison charts in a `reports/` folder.
- Add a final prediction export step for test-set submissions.
- Move feature engineering into reusable functions for easier experimentation.
- Add model cards explaining expected use and limitations.

## Author

Pranika Chandra  
Data science and machine learning portfolio project.
