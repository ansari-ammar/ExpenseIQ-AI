
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("monthly_dataset.csv")

# Select numerical columns
numeric_data = df.select_dtypes(include="number")

# Calculate correlation
correlation = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation)

os.makedirs("charts", exist_ok=True)

# Generate heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Expense Correlation Heatmap")
plt.tight_layout()

plt.savefig("charts/correlation_heatmap.png")
plt.close()

print("\nCorrelation Analysis Completed!")
