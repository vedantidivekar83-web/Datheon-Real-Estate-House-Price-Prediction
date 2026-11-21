# Datheon Internship Task 2: Real Estate House Price Prediction

A production-grade machine learning project implementing end-to-end regression modeling, feature engineering, multicollinearity diagnostics, hyperparameter optimization, comprehensive residual analysis, and model persistence on the California Housing dataset.

---

## Table of Contents
1. [Project Title and Executive Overview](#project-title-and-executive-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Source, Schema and Dimensions](#dataset-source-schema-and-dimensions)
4. [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
5. [Data Preprocessing Workflow & Leakage Audit](#data-preprocessing-workflow--leakage-audit)
6. [Feature Engineering & Multicollinearity Analysis](#feature-engineering--multicollinearity-analysis)
7. [Models Implemented](#models-implemented)
8. [Hyperparameter Tuning Methodology (GridSearchCV)](#hyperparameter-tuning-methodology-gridsearchcv)
9. [Comprehensive Model Evaluation & Benchmarks](#comprehensive-model-evaluation--benchmarks)
10. [Residual Analysis & Error Diagnostics](#residual-analysis--error-diagnostics)
11. [Sub-Market Performance Across Ocean Proximity](#sub-market-performance-across-ocean-proximity)
12. [Model Artifact & Persistence Verification](#model-artifact--persistence-verification)
13. [Limitations & Practical Constraints](#limitations--practical-constraints)
14. [Project Directory Structure](#project-directory-structure)
15. [Installation & Execution Instructions](#installation--execution-instructions)
16. [Dependencies & Environment](#dependencies--environment)
17. [Ethical & Broader Societal Considerations](#ethical--broader-societal-considerations)

---

## Project Title and Executive Overview
**Datheon Internship Task 2: Real Estate House Price Prediction**  
This repository presents a machine learning solution to estimate median residential property values across California census block groups. The project follows a structured 9-stage engineering workflow designed to reflect rigorous real-world modeling practices:
- **Baseline Modeling**: Ordinary Least Squares (OLS), Ridge ($L_2$), and Lasso ($L_1$) regression.
- **Hyperparameter Optimization**: Systematic 5-fold cross-validation grid search.
- **Top Performer**: **Baseline Ridge Regression ($\alpha=1.0$)**, achieving a Test RMSE of **$\$66,632.95$**, Test MAE of **$\$48,789.40$**, and Test $R^2$ of **$0.6593$**.
- **Most Parsimonious Model**: **Tuned Lasso Regression ($\alpha=100$)**, driving 4 collinear indicators to zero while maintaining a Test RMSE of **$\$66,785.82$**.
- **Deployment Packaging**: An integrated end-to-end Scikit-Learn `Pipeline` serialized via `joblib`, verified to reproduce test predictions with exact $0.0$ discrepancy on raw input data.

---

## Problem Statement
Accurate residential real estate valuation is a critical operational capability for financial lenders, municipal taxing authorities, urban planners, and individual home buyers. Real estate valuations are governed by complex interactions between household demographics, structural capacity, and spatial geography (distance to economic hubs and coastal coastlines).

The primary objective is to construct a reproducible regression pipeline that accurately estimates `median_house_value` for unseen census districts while controlling for multicollinearity, avoiding data leakage, evaluating regression assumptions, and diagnosing structural model limitations.

---

## Dataset Source, Schema and Dimensions
- **Source**: 1990 California Housing Dataset, originally compiled by Pace and Barry (1997) from the 1990 U.S. Census, hosted via Hands-on ML repository:  
  `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv`
- **Total Dimensions**: $20,640$ observations $\times$ $10$ attributes ($9$ input features + $1$ continuous target).
- **Target Variable**: `median_house_value` (Continuous, measured in USD; Range: $\$14,999$ to $\$500,001$, Mean: $\$206,855.82$, Median: $\$179,700.00$).
- **Features**:
  1. `longitude` (Float64): Spatial east-west coordinate.
  2. `latitude` (Float64): Spatial north-south coordinate.
  3. `housing_median_age` (Float64): Median building age within block group (1 to 52 years).
  4. `total_rooms` (Float64): Total room count across all homes in block group.
  5. `total_bedrooms` (Float64): Total bedroom count across all homes (207 missing values; $1.003\%$).
  6. `population` (Float64): Total residents residing in block group.
  7. `households` (Float64): Total occupied residential units.
  8. `median_income` (Float64): Median income in tens of thousands of USD ($0.4999$ to $15.0001$).
  9. `ocean_proximity` (Categorical String, 5 categories): `<1H OCEAN` ($9,136$), `INLAND` ($6,551$), `NEAR OCEAN` ($2,658$), `NEAR BAY` ($2,290$), `ISLAND` ($5$).

---

## Exploratory Data Analysis (EDA) Highlights
1. **Target Distribution Right-Skew**: `median_house_value` exhibits a pronounced right skew with a major secondary spike at the upper truncation threshold of **$\$500,001$** ($965$ records in full dataset; $4.68\%$).
2. **Dominant Feature Predictor**: `median_income` exhibits the strongest linear correlation with property value ($r = +0.6881$).
3. **Geographic Clustering**: High property valuations cluster along coastal corridors (San Francisco Bay Area and coastal Southern California), while lower valuations dominate the Central Valley / Inland regions.

---

## Data Preprocessing Workflow & Leakage Audit
To preserve strict statistical isolation between training and testing data, the following preprocessing design was executed:
1. **Stratified Partitioning**: Partitioned 80% Training ($16,512$ samples) and 20% Testing ($4,128$ samples) using `train_test_split` stratified across 5 `median_income` tiers to maintain proportional income representation.
2. **Median Missing Value Imputation**: `SimpleImputer(strategy='median')` fitted solely on training numerical attributes (`total_bedrooms` median $= 435.0$). Zero missing values remain. Target values were never imputed.
3. **Standard Scaling**: `StandardScaler()` applied strictly post-imputation and post-feature engineering, fitting $\mu$ and $\sigma$ only on training data.
4. **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to `ocean_proximity`, generating 5 binary indicator columns resilient to unseen levels during inference.
5. **Leakage Audit**: Verified that test feature matrices (`X_test_final`) were generated strictly using `.transform()`, ensuring zero test information leaked into training transformations.

---

## Feature Engineering & Multicollinearity Analysis

### Domain Ratios & Interaction Features
To extract meaningful economic signals from aggregate census block group counts, four domain features and five spatial interactions were engineered:
1. `rooms_per_household = total_rooms / households`: Average living space per household ($r = +0.151$).
2. `bedrooms_per_room = total_bedrooms / total_rooms`: Structural room density; strong negative correlation with price ($r = -0.255$) as crowded homes correlate with lower property valuations.
3. `population_per_household = population / households`: Average household size.
4. `income_age_interaction = median_income * housing_median_age`: Joint demographic effect of neighborhood maturity and earning capacity.
5. `income_x_<ocean_proximity>`: Five interaction terms multiplying standardized `median_income` by the coastal indicators (`<1H OCEAN`, `INLAND`, `NEAR BAY`, `NEAR OCEAN`, `ISLAND`), capturing localized economic premiums.

**Final Feature Matrix**: Exactly **22 engineered features** (12 numerical + 5 one-hot categorical + 5 coastal interactions).

### Multicollinearity & Variance Inflation Factor (VIF)
- Extreme collinearity was identified between spatial coordinates (`latitude` VIF $= 22.86$, `longitude` VIF $= 20.48$) and volume counts (`households` VIF $= 35.85$, `total_bedrooms` VIF $= 29.56$).
- Rather than discarding predictive geographic coordinates, regularized regression ($L_1$ and $L_2$) was selected to constrain coefficient variance mathematically.

---

## Models Implemented
1. **Ordinary Least Squares (OLS) Linear Regression**: Baseline unpenalized estimator solving the closed-form normal equations $\hat{\beta} = (X^T X)^{-1} X^T y$.
2. **Ridge Regression ($L_2$ Regularization)**: Penalizes sum of squared coefficients ($\alpha \sum \beta_j^2$) to mitigate collinear variance inflation.
3. **Lasso Regression ($L_1$ Regularization)**: Penalizes sum of absolute coefficients ($\alpha \sum |\beta_j|$) enabling automatic feature selection through exact coefficient zeroing.

---

## Hyperparameter Tuning Methodology (GridSearchCV)
Hyperparameter optimization was conducted exclusively on the training matrix (`X_train_final`, $N=16,512$) using `GridSearchCV` with **5-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`) using `neg_root_mean_squared_error` scoring:
- **Ridge Grid**: $\alpha \in [0.01, 0.1, 1, 10, 100, 1000, 10000]$  
  - Optimal Alpha: **$\alpha = 100$** (Mean CV RMSE: **$\$68,808.06 \pm \$1,456.19$**, CV $R^2$: $0.6462$).  
  - While lower $\alpha$ gave marginally lower training loss, $\alpha = 100$ minimized fold-to-fold cross-validation variance ($\sigma = \$1,456.19$).
- **Lasso Grid**: $\alpha \in [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000]$  
  - Optimal Alpha: **$\alpha = 100$** (Mean CV RMSE: **$\$68,760.33 \pm \$1,511.03$**, CV $R^2$: $0.6467$).  
  - Coordinate descent converged cleanly without warnings.
  - Zeroed exactly 4 redundant features: `ocean_proximity_<1H OCEAN`, `ocean_proximity_ISLAND`, `income_x_<1H OCEAN`, and `income_x_ISLAND`.

---

## Comprehensive Model Evaluation & Benchmarks

All candidate models evaluated on identical unseen test instances ($N = 4,128$):

| Model Name | Train RMSE | Test RMSE | Train MAE | Test MAE | Train $R^2$ | Test $R^2$ | Architectural Distinction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ridge Baseline ($\alpha=1.0$)** | $\$68,378.36$ | **$\$66,632.95$** | $\$49,003.62$ | **$\$48,789.40$** | $0.6507$ | **$0.6593$** | **Overall Top Performer** |
| **Tuned Lasso ($\alpha=100$)** | $\$68,402.55$ | $\$66,785.82$ | $\$49,021.46$ | $\$48,841.42$ | $0.6505$ | $0.6578$ | **Best Parsimonious Model** |
| **Tuned Ridge ($\alpha=100$)** | $\$68,443.98$ | $\$66,824.40$ | $\$48,991.81$ | $\$48,823.54$ | $0.6500$ | $0.6574$ | Variance-Stabilized Model |
| **Lasso Baseline ($\alpha=0.01$)** | $\$68,375.25$ | $\$67,012.72$ | $\$48,997.01$ | $\$48,902.21$ | $0.6507$ | $0.6554$ | Under-penalized $L_1$ Baseline |
| **OLS Linear Regression** | $\$68,375.24$ | $\$67,056.11$ | $\$48,996.59$ | $\$48,910.25$ | $0.6507$ | $0.6550$ | Unregularized Baseline |

---

## Residual Analysis & Error Diagnostics

Residuals were computed for the top-performing model as $e_i = y_i - \hat{y}_i$:

### Error Distribution & Percentile Breakdown
- **Mean Absolute Error (MAE)**: **$\$48,789.40$**
- **Median Absolute Error (MedAE)**: **$\$36,734.06$** (Robust 50th percentile)
- **Minimum Absolute Error**: **$\$7.15$**
- **Maximum Absolute Error**: **$\$336,604.95$**
- **90th Percentile Error ($P_{90}$)**: **$\$102,903.88$**
- **95th Percentile Error ($P_{95}$)**: **$\$137,918.32$**

> **MAE vs. MedAE Analysis**: Median Absolute Error is **$\$12,055.34$ ($32.8\%$) lower than MAE**. Because absolute errors are heavily right-skewed with extreme misses exceeding $\$300\text{k}$, the arithmetic mean is inflated. For the central $50\%$ of California homes, the model's typical error is approximately $\approx \$36,700$.

### Diagnostic Assumption Checks
1. **Heteroskedasticity**: Residual variance expands significantly as predicted property values increase (fan-out pattern). High-value homes exhibit greater absolute pricing uncertainty.
2. **Target Capping Boundary**: A prominent diagonal boundary line ($e_i = 500,001 - \hat{y}_i$) appears in the residual space, reflecting census target truncation.
3. **Non-Normality**: The Q-Q plot exhibits heavy tails at both extremes, failing Gaussian normality tests.

---

## Sub-Market Performance Across Ocean Proximity

Grouped using original unencoded test-set labels (`X_test['ocean_proximity']`):

| Ocean Proximity Category | Test Samples ($N$) | Mean Actual Price | Mean Predicted Price | MAE | RMSE | Within-Group $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ISLAND** | 3 | $\$400,000.00$ | $\$327,652.40$ | $\$100,474.87$ | $\$108,859.06$ | N/A ($N=3$) |
| **NEAR BAY** | 443 | $\$261,179.53$ | $\$261,189.90$ | $\$64,011.08$ | $\$81,830.38$ | $0.5605$ |
| **NEAR OCEAN** | 534 | $\$253,707.76$ | $\$250,354.62$ | $\$62,517.41$ | $\$81,366.97$ | $0.5575$ |
| **<1H OCEAN** | 1,859 | $\$237,163.21$ | $\$238,763.06$ | $\$50,416.53$ | $\$68,359.46$ | $0.5558$ |
| **INLAND** | 1,289 | $\$122,702.33$ | $\$123,763.52$ | **$\$35,403.93$** | **$\$49,069.00$** | $0.4883$ |

- **Inland Precision**: Lower base prices ($\approx \$122.7\text{k}$) and low variance produce the lowest error rates.
- **Coastal Volatility**: Bay and Ocean proximity command high amenity premiums and luxury price spikes, yielding MAEs $> \$62,000$.

---

## Model Artifact & Persistence Verification
- **Artifact Path**: `models/best_ridge_pipeline.joblib`
- **Artifact File Size**: **$2,326$ bytes** ($2.27$ KB).
- **Architecture**: A composite Scikit-Learn `Pipeline` wrapping `full_preprocessing_pipeline` (imputer, feature engineers, scalers, encoders) and `ridge_model` ($\alpha=1.0$).
- **Verification Audit**:
  - Successfully reloaded into an independent session.
  - Ingested raw, un-preprocessed test records (`X_test`).
  - Reproduced baseline predictions with **$0.0$ max absolute difference** (`np.allclose = True`).
  - Replicated Test RMSE ($\$66,632.95$), MAE ($\$48,789.40$), and $R^2$ ($0.6593$) exactly.

---

## Limitations & Practical Constraints
1. **Artificial Target Capping ($\$500,001$)**: Properties capped during 1990 census collection prevent the model from learning genuine price dynamics for luxury real estate.
2. **Linear Representation Boundary**: Linear regularized models plateau at $\approx 66\% R^2$. Complex non-linear geographic interactions (fault lines, micro-climates, view corridors, school districts) cannot be captured with linear planes.
3. **Temporal Drift**: Based on 1990 census data; historical dollar magnitudes cannot be applied to modern transactions without CPI adjustment and spatial retraining.
4. **Production Recommendation**: Standalone production deployment is not recommended without augmenting with non-linear tree-based ensembles (e.g., Random Forest, XGBoost, LightGBM).

---

## Project Directory Structure

```
Datheon-Real-Estate-House-Price-Prediction/
├── Real_Estate_Price_Prediction.ipynb       # Complete 9-section master Jupyter Notebook
├── README.md                                # Comprehensive project documentation
├── requirements.txt                         # Verified Python dependencies
├── .gitignore                               # Production gitignore preserving artifacts
├── data/
│   └── housing.csv                          # Original California housing dataset (20,640 rows)
├── models/
│   └── best_ridge_pipeline.joblib           # Serialized end-to-end inference pipeline
└── outputs/                                 # 23 High-resolution generated visualization artifacts
    ├── target_variable_distribution.png
    ├── correlation_matrix_heatmap.png
    ├── vif_scores_barchart.png
    ├── model_test_rmse_comparison.png
    ├── model_test_mae_comparison.png
    ├── model_test_r2_comparison.png
    ├── model_coefficient_comparison.png
    ├── ridge_alpha_vs_cv_rmse.png
    ├── lasso_alpha_vs_cv_rmse.png
    ├── ridge_alpha_vs_cv_r2.png
    ├── lasso_alpha_vs_cv_r2.png
    ├── actual_vs_predicted_ridge_baseline.png
    ├── actual_vs_predicted_ridge_tuned.png
    ├── residual_vs_predicted_ridge_baseline.png
    ├── residual_distribution_histogram.png
    ├── residual_boxplot.png
    ├── residual_qq_plot.png
    ├── residual_diagnostic_panel.png
    ├── mae_by_ocean_proximity.png
    ├── model_evaluation_test_rmse.png
    ├── model_evaluation_test_mae.png
    ├── model_evaluation_test_r2.png
    └── final_model_comparison_metrics.png
```

---

## Installation & Execution Instructions

### 1. Environment Setup
Clone the repository and create a Python 3.11.7 virtual environment:
```bash
git clone <repository-url>
cd Datheon-Real-Estate-House-Price-Prediction
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Real_Estate_Price_Prediction.ipynb
```
Select **Kernel -> Restart & Run All** to execute all 109 cells sequentially from beginning to end.

### 4. Direct Inference with Serialized Pipeline
```python
import joblib
import pandas as pd

# Load the saved end-to-end pipeline
pipeline = joblib.load('models/best_ridge_pipeline.joblib')

# Ingest raw property data (9 original features)
sample_property = pd.DataFrame([{
    'longitude': -122.23,
    'latitude': 37.88,
    'housing_median_age': 41.0,
    'total_rooms': 880.0,
    'total_bedrooms': 129.0,
    'population': 322.0,
    'households': 126.0,
    'median_income': 8.3252,
    'ocean_proximity': 'NEAR BAY'
}])

# Predict median valuation directly
predicted_price = pipeline.predict(sample_property)[0]
print(f"Estimated Market Value: ${predicted_price:,.2f}")
```

---

## Dependencies & Environment
- **Python**: `3.11.7`
- **Core Libraries**: `pandas >= 2.1.0`, `numpy >= 1.26.0`, `scipy >= 1.11.0`
- **Machine Learning**: `scikit-learn >= 1.3.0`, `statsmodels >= 0.14.0`, `joblib >= 1.3.0`
- **Visualization**: `matplotlib >= 3.8.0`, `seaborn >= 0.13.0`
- **Interactive Computing**: `notebook >= 7.0.0`, `ipykernel >= 6.25.0`

---

## Ethical & Broader Societal Considerations
1. **Algorithmic Bias & Redlining**: Historical census data reflect socio-economic and demographic patterns of the 1990 era. Automated valuation models built on spatial proxies can perpetuate historical redlining if used without fair lending oversight.
2. **Assessment Regressivity**: Because the linear model exhibits higher relative error percentage on lower-priced homes ($\approx 30-70\%$) compared to mid-market properties ($\approx 10-20\%$), deploying this model for municipal property taxation would disproportionately misassess lower-income households.
3. **Transparency**: Regularized linear models provide explicit, auditable coefficient weights ($\beta$), satisfying regulatory interpretability standards in financial fair-lending frameworks.
