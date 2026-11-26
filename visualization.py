
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("cleaned_dataset.csv")
monthly = pd.read_csv("monthly_dataset.csv")

os.makedirs("charts", exist_ok=True)

# Graph 1: Monthly expenses
plt.figure(figsize=(12, 5))
plt.plot(monthly["month_index"],
         monthly["total_expense"],
         color="blue")

plt.title("Monthly Expense Trend")
plt.xlabel("Month")
plt.ylabel("Total Expense")
plt.tight_layout()
plt.savefig("charts/monthly_expenses.png")
plt.close()

# Graph 2: Income vs Expense
plt.figure(figsize=(12, 5))

plt.plot(monthly["month_index"],
         monthly["monthly_income"],
         label="Income")

plt.plot(monthly["month_index"],
         monthly["total_expense"],
         label="Expense")

plt.title("Monthly Income vs Expense")
plt.legend()
plt.tight_layout()
plt.savefig("charts/income_vs_expense.png")
plt.close()

# Graph 3: Category-wise expenses
expenses = df[df["Type"] == "Expense"]

category = expenses.groupby("Category")["Amount"].sum()

plt.figure(figsize=(9, 9))
plt.pie(category, labels=category.index,
        autopct="%1.1f%%")

plt.title("Category Wise Expenses")
plt.tight_layout()
plt.savefig("charts/category_expenses.png")
plt.close()

print("All Charts Generated Successfully!")
