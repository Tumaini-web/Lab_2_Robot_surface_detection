import pandas as pd

# Read the CSV file
df = pd.read_csv("data/processed/train.csv")

# Sort by the 'Type' column (ascending order: 0 first, 1 later)
# If you want 1 first, set ascending=False
df_sorted = df.sort_values(by="Type", ascending=False)

# Reset index (optional, for a clean sequential index)
df_sorted.reset_index(drop=True, inplace=True)

# Save the sorted CSV (optional)
df_sorted.to_csv("data/processed/train_sorted.csv", index=False)

# Print first few rows to verify
print(df_sorted.head())
