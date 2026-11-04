# Car Price Prediction

## Overview

This project builds a complete machine learning pipeline to predict car selling prices based on features like manufacturing year, present price, driven kilometers, fuel type, transmission, and owner history. The goal is to compare multiple regression models and select the best one for deployment.

## Dataset

The dataset contains **301 rows** and **9 columns**:

| Column | Description |
|--------|-------------|
| Car_Name | Model of the vehicle (cars and motorcycles) |
| Year | Manufacturing year (2003–2018) |
| Selling_Price | Price at which the vehicle is being sold (target) |
| Present_Price | Current market value of the vehicle |
| Driven_kms | Kilometers driven |
| Fuel_Type | Petrol, Diesel, or CNG |
| Selling_type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

## Workflow

The notebook follows a complete machine learning pipeline:

1. **Data Exploration** — Loaded data, checked shape, info, and statistical summary.
2. **Data Cleaning** — Handled missing values, removed 2 duplicate rows, and identified outliers using Z-score and IQR methods.
3. **Inconsistency Removal** — Removed bike-related rows and imbalanced data points (Owner=3, Year=2018, Car_Name='800').
4. **Feature Engineering** — Created new features: Age_of_car, Car_depreciation, Depreciation_per_year, Depreciation_rate, Brand, Car_Condition, and Car_Mileage.
5. **Type Conversion** — Converted object columns to category type.
6. **Target Transformation** — Applied square root transformation to reduce skewness from 2.44 to 0.60.
7. **Correlation Analysis** — Visualized correlation matrix with heatmap.
8. **Preprocessing Pipeline** — Used ColumnTransformer with MinMaxScaler for numerical columns and OneHotEncoder for categorical columns.
9. **Model Training** — Compared 9 regression models with GridSearchCV hyperparameter tuning.
10. **Model Evaluation** — Used MSE, MAE, and R² Score.
11. **Model Persistence** — Saved the best model pipeline with Pickle for deployment.

## Results

| Model | R² Score | MSE | MAE |
|-------|----------|-----|-----|
| Linear Regression | 0.983 | 0.018 | 0.091 |
| Ridge Regression | 0.982 | 0.019 | 0.096 |
| SVR | 0.982 | 0.020 | 0.105 |
| Decision Tree | 0.953 | 0.051 | 0.151 |
| XGBoost | 0.907 | 0.101 | 0.133 |
| Random Forest | 0.893 | 0.117 | 0.142 |
| Gradient Boosting | 0.891 | 0.119 | 0.136 |
| Lasso Regression | 0.811 | 0.207 | 0.239 |
| KNN | 0.771 | 0.251 | 0.297 |

**Best Model:** Linear Regression (R² = 0.983, MSE = 0.018, MAE = 0.091)

## Tools and Libraries

| Library | Purpose |
|---------|---------|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosting model |
| Pickle | Model serialization |

## How to Run

1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook car_price_prediction.ipynb
   ```
4. Run all cells sequentially.

## Files in This Repository

| File | Description |
|------|-------------|
| `car_price_prediction.ipynb` | Main notebook with full workflow |
| `car data.csv` | Dataset used for training and testing |
| `best_model.pkl` | Saved best model pipeline |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |



