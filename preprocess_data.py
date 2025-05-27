import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load your CSV file
df = pd.read_csv("generated_quote_data.csv")  #####//data loaded from the generated data file

# 2. Show preview
print("Preview of the data:")
print(df.head())

# 3. Handle missing values
df["surface_treatment"] = df["surface_treatment"].fillna("None")
df = df.fillna(method='ffill')  # Forward fill for any remaining missing values

# 4. Encode categorical features
df = pd.get_dummies(df, columns=[
    "profile_name", "material", "alloy", "tolerance", "surface_treatment"
])

# 5. Extract features from date
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df = df.drop("date", axis=1)  # Drop original date column

# 6. Normalize numerical columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 7. Save new file
df.to_csv("preprocessed_quotes.csv", index=False)
print("✅ Preprocessing complete. Saved as 'preprocessed_quotes.csv'")
