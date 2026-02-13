from src.preprocess import preprocess_data

# Path to your SMS dataset
filepath = "data/smsspamcollection.txt"

# Preprocess
df = preprocess_data(filepath)

# Show first 5 rows для теста
print(df.head())

# Save processed CSV (optional) можно либо так оставим
df.to_csv("data/smsspamcollection_processed.csv", index=False)
print("Preprocessed CSV saved as data/smsspamcollection_processed.csv")
