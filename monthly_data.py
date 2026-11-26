
import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["Month_Year"] = df["Date"].dt.to_period("M")

# Separate income and expenses
income = df[df["Type"] == "Income"]
expense = df[df["Type"] == "Expense"]

# Calculate monthly totals
monthly_income = income.groupby("Month_Year")["Amount"].sum()
monthly_expense = expense.groupby("Month_Year")["Amount"].sum()

# Combine monthly data
monthly = pd.DataFrame({
    "monthly_income": monthly_income,
    "total_expense": monthly_expense
}).fillna(0)

# Add expense categories
categories = expense.pivot_table(
    index="Month_Year",
    columns="Category",
    values="Amount",
    aggfunc="sum",
    fill_value=0
)

monthly = monthly.join(categories).fillna(0)

# Arrange months chronologically
monthly = monthly.sort_index().reset_index()

monthly["month_index"] = range(1, len(monthly) + 1)

print("\nMonthly Dataset:")
print(monthly.head())

print("\nDataset Shape:", monthly.shape)

monthly.to_csv("monthly_dataset.csv", index=False)

print("\nMonthly Aggregation Completed!")
