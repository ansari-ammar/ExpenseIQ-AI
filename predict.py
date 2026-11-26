
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/expense_model.pkl")

# Load monthly dataset
df = pd.read_csv("monthly_dataset.csv")

# Calculate next month index
next_month = df["month_index"].max() + 1

# Take user input
income = float(input("Enter Expected Monthly Income: "))

# Prepare input data
data = pd.DataFrame({
    "month_index": [next_month],
    "monthly_income": [income]
})

# Predict expense
expense = max(0, model.predict(data)[0])

# Calculate estimated savings
savings = income - expense

print("\n----- ExpenseIQ AI -----")
print("Monthly Income:", round(income, 2))
print("Predicted Expense:", round(expense, 2))
print("Estimated Savings:", round(savings, 2))
