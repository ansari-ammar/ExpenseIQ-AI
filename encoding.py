
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Encode transaction type
df["Type_Encoded"] = df["Type"].map({
    "Income": 0,
    "Expense": 1
})

# One-hot encoding of categories
encoded_df = pd.get_dummies(
    df, columns=["Category"], dtype=int
)

print("\nEncoded Dataset:")
print(encoded_df.head())

# Save encoded dataset
encoded_df.to_csv("encoded_dataset.csv", index=False)

print("\nEncoding Successfully Completed!")
