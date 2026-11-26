
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load monthly dataset
df = pd.read_csv("monthly_dataset.csv")
df = df.sort_values("month_index")

# 2. Select input features and target
X = df[["month_index", "monthly_income"]]
y = df["total_expense"]

# 3. Split data: 80% training, 20% testing
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 4. Define models
models = {
    "Historical Mean": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

# 5. Compare models using training data
cv = TimeSeriesSplit(n_splits=5)

scores = {}

for name, model in models.items():

    score = -cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="neg_mean_absolute_error"
    ).mean()

    scores[name] = score

    print(name, "Validation MAE:", round(score, 2))

# 6. Select best model
best_name = min(scores, key=scores.get)
best_model = models[best_name]

# 7. Train selected model
best_model.fit(X_train, y_train)

# 8. Predict expenses
predictions = best_model.predict(X_test)

# 9. Model evaluation
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\nBest Model:", best_name)
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 10. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/expense_model.pkl")

# 11. Save evaluation results
results = pd.DataFrame({
    "Model": [best_name],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2": [r2]
})

results.to_csv("model_results.csv", index=False)

# 12. Save actual vs predicted values
comparison = pd.DataFrame({
    "Month": X_test["month_index"].values,
    "Actual": y_test.values,
    "Predicted": predictions
})

comparison.to_csv("predictions.csv", index=False)

# 13. Generate prediction graph
os.makedirs("charts", exist_ok=True)

plt.figure(figsize=(12, 6))

plt.plot(
    comparison["Month"],
    comparison["Actual"],
    marker="o",
    label="Actual Expense"
)

plt.plot(
    comparison["Month"],
    comparison["Predicted"],
    marker="s",
    label="Predicted Expense"
)

plt.title("Actual vs Predicted Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Expense")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("charts/model_prediction.png")
plt.close()

print("\nModel Training Completed!")
print("Model Saved Successfully!")
