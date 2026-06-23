import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Load dataset
bike_rental = pd.read_csv("Dataset.csv")

# Replace hidden missing values with NaN
bike_rental.replace(['?', '#', 'N/A', 'NA', 'null', 'None', '-', ''], np.nan, inplace=True)

# Drop row index column
bike_rental.drop(columns=['instant'], inplace=True)

# Fix data types
num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth', 'casual', 'registered']
for col in num_cols:
    bike_rental[col] = pd.to_numeric(bike_rental[col], errors='coerce')

# Impute missing values
cat_cols = ['season', 'holiday', 'workingday', 'weathersit']
for col in cat_cols:
    bike_rental[col] = bike_rental[col].fillna(bike_rental[col].mode()[0])

num_only = bike_rental.select_dtypes(include='number').columns
bike_rental[num_only] = bike_rental[num_only].fillna(bike_rental[num_only].median())

# Outlier treatment - hum=0 is a sensor anomaly
bike_rental['hum'] = bike_rental['hum'].replace(0, bike_rental['hum'].median())

# Drop leakage columns and atemp (multicollinearity with temp)
bike_rental.drop(columns=['casual', 'registered', 'atemp'], inplace=True)

# Date feature engineering
bike_rental['dteday'] = pd.to_datetime(bike_rental['dteday'], dayfirst=True)
bike_rental['is_weekend'] = bike_rental['dteday'].dt.dayofweek.isin([5, 6]).astype(int)
bike_rental['quarter'] = bike_rental['dteday'].dt.quarter
bike_rental.drop(columns=['dteday'], inplace=True)

bike_rental['is_summer'] = bike_rental['mnth'].apply(lambda x: 1 if x in [5, 6, 7, 8] else 0)
bike_rental['is_winter'] = bike_rental['mnth'].apply(lambda x: 1 if x in [11, 12, 1, 2] else 0)

# One-hot encoding
categorical_cols = ['season', 'holiday', 'workingday', 'weathersit']
bike_rental = pd.get_dummies(bike_rental, columns=categorical_cols, drop_first=True, dtype=int)

# Standard scaling
scaler = StandardScaler()
scale_cols = ['temp', 'hum', 'windspeed']
bike_rental[scale_cols] = scaler.fit_transform(bike_rental[scale_cols])

print("Final preprocessed shape:", bike_rental.shape)

# Train test split
X = bike_rental.drop('cnt', axis=1)
y = bike_rental['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost with GridSearchCV...")

# Hyperparameter tuning - XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6]
}

xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    xgb_params, cv=5, scoring='r2', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)

best_model = xgb_grid.best_estimator_
print("Best Params:", xgb_grid.best_params_)

# Save model, scaler and column order for deployment
# json instead of pickle - more stable across xgboost versions
best_model.save_model('best_bike_rental_model.json')
# npy instead of pkl - avoids sklearn version mismatch on deployment
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)
joblib.dump(list(X.columns), 'model_columns.pkl')

print("Saved: best_bike_rental_model.json, scaler_mean.npy, scaler_scale.npy, model_columns.pkl")
print("Done. Now run: streamlit run app.py")
