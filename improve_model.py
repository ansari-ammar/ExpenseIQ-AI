
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("monthly_dataset.csv")
df = df.sort_values("month_index")

# Feature Engineering
df["month_number"] = pd.to_datetime(
    df["Month_Year"]
).dt.month

df["previous_expense"] = df["total_expense"].shift(1)

df["rolling_average"] = (
    df["total_expense"].shift(1).rolling(3).mean()
)

df = df.dropna()

# Input features
features = [
    "month_index",
    "monthly_income",
    "month_number",
    "previous_expense",
    "rolling_average"
]

X = df[features]
y = df["total_expense"]

# Chronological splitting
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Define models
models = {
    "Historical Mean": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=10),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        random_state=42
    )
}

cv = TimeSeriesSplit(n_splits=5)

scores = {}

print("\nMODEL COMPARISON\n")

# Compare models
for name, model in models.items():

    mae = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error"
    ).mean()

    scores[name] = mae

    print(name, "Validation MAE:", round(mae, 2))

# Select best model
best_name = min(scores, key=scores.get)
best_model = models[best_name]

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nFINAL RESULTS")
print("Best Model:", best_name)
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save improved model separately
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/improved_model.pkl"
)

# Save model comparison
comparison = pd.DataFrame({
    "Model": list(scores.keys()),
    "Validation_MAE": list(scores.values())
})

comparison.to_csv(
    "model_comparison.csv", index=False
)

# Save evaluation
pd.DataFrame({
    "Model": [best_name],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2": [r2]
}).to_csv("improved_results.csv", index=False)

print("\nModel Comparison Completed!")
