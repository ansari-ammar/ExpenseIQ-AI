
import pandas as pd

# 1. Load dataset
df = pd.read_csv("expense_dataset.csv")

print("Original Dataset Shape:", df.shape)

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Remove missing values
df = df.dropna()

# 4. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Convert date into datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 7. Convert Amount into numeric format
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

# 8. Remove invalid records
df = df.dropna()

# 9. Extract year and month
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 10. Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Dataset Shape:", df.shape)

# 11. Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nData Cleaning Successfully Completed!")
