# Bike Rental Demand Prediction

Predicts hourly bike rental demand from weather and time data, using the UCI Bike Sharing Dataset (2011-2012). Includes EDA, model comparison across 6 algorithms, hyperparameter tuning, and a Streamlit app for live predictions.

XGBoost ended up being the best model after tuning:

| Metric | Score |
|---|---|
| R² | 0.950 |
| MAE | 24.51 |
| RMSE | 39.96 |

## Project Structure

```
bike-rental-demand-prediction/
├── bike_rental_demand_prediction.ipynb   # Full EDA + model building notebook
├── Dataset.csv                            # UCI Bike Sharing dataset
├── train_model.py                         # Cleans data, trains XGBoost, saves model
├── app.py                                 # Streamlit app for predictions
├── requirements.txt                       # Python dependencies
└── README.md
```

## Dataset

UCI Bike Sharing Dataset. 17,379 hourly records from a bike-sharing system (2011-2012), with weather conditions (temperature, humidity, windspeed, weather situation), calendar info (season, holiday, working day), and rental counts.

## Key Steps

The raw data had missing values hidden as `'?'` instead of NaN, and a few columns had the wrong dtype because of it. After fixing types, categorical columns were imputed with mode and numeric columns with median. One real anomaly turned up during outlier analysis: 22 rows had `humidity = 0`, which is physically impossible, all from a single date, so these were replaced with the median.

For features, `is_weekend`, `quarter`, `is_summer`, and `is_winter` were derived from the date and month columns, and the categorical columns (season, holiday, working day, weather) were one-hot encoded. `casual` and `registered` were dropped since they sum directly to the target `cnt` (data leakage), and `atemp` was dropped for being almost perfectly correlated with `temp` (r = 0.99).

Six regression models were trained and compared, then tuned with `GridSearchCV` using 5-fold cross-validation, evaluated on MAE, RMSE, and R².

The notebook ends with the model comparison and best model selection. For deployment, `train_model.py` reruns this same pipeline end to end and saves the final model, scaler, and feature column order, which `app.py` then loads for live predictions.

Full notebook: [`bike_rental_demand_prediction.ipynb`](./bike_rental_demand_prediction.ipynb)

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/code-with-muskan17/bike-rental-demand-prediction.git
cd bike-rental-demand-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train and save the model (creates the model and scaler files used by the app):
```bash
python train_model.py
```

4. Run the app:
```bash
streamlit run app.py
```

The app opens automatically in your browser at `localhost:8501`.

## Model Comparison

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **XGBoost** | **0.950** | **24.51** | **39.96** |
| Random Forest | 0.944 | 24.76 | 42.06 |
| Gradient Boosting | 0.937 | 28.51 | 44.68 |
| Decision Tree | 0.903 | 31.85 | 55.48 |
| KNN | 0.891 | 37.89 | 58.67 |
| Linear Regression | 0.404 | 103.29 | 137.32 |

## Example

June, 5 PM, Wednesday, summer, clear weather, working day → predicted 794 rentals

January, 3 AM, Sunday, heavy rain, non-working day → predicted 0 rentals

## Limitations

- The model is trained only on 2011–2012 data and cannot reliably predict for other years without retraining on newer data.
- Tree-based models like XGBoost do not extrapolate trends beyond their training range.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn

## Author

Muskan Agrawal ([code-with-muskan17](https://github.com/code-with-muskan17))
