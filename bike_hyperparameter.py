import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("hour.csv")

# Convert date
df['dteday'] = pd.to_datetime(df['dteday'])

# ============================================
# FEATURE ENGINEERING
# ============================================

df['day'] = df['dteday'].dt.day
df['dayofyear'] = df['dteday'].dt.dayofyear

def time_period(hour):
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 17:
        return 2
    elif 17 <= hour < 21:
        return 3
    else:
        return 0

df['time_period'] = df['hr'].apply(time_period)

# ============================================
# FEATURES AND TARGET
# ============================================

features = [
    'season',
    'yr',
    'mnth',
    'hr',
    'day',
    'dayofyear',
    'time_period',
    'holiday',
    'weekday',
    'workingday',
    'weathersit',
    'temp',
    'atemp',
    'hum',
    'windspeed'
]

X = df[features]
y = df['cnt']

# ============================================
# TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================
# HYPERPARAMETER TUNING
# ============================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ============================================
# BEST MODEL
# ============================================

best_model = grid_search.best_estimator_

print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

# ============================================
# PREDICTION
# ============================================

y_pred = best_model.predict(X_test)

# ============================================
# EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== TUNED MODEL RESULTS ==========")

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))