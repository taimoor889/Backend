import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("generated_quote_data.csv")  #####//data loaded from the generated data file


# print("Preview of the data:")
# print(df.head())

# 3. check data is nullll
df["surface_treatment"] = df["surface_treatment"].fillna("None")
df = df.fillna(method='ffill')  


df = pd.get_dummies(df, columns=[
    "profile_name", "material", "alloy", "tolerance", "surface_treatment"
])


df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df = df.drop("date", axis=1)  


scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


df.to_csv("preprocessed_quotes.csv", index=False)
print("✅ Preprocessing complete. Saved as 'preprocessed_quotes.csv'")
