import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("price_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input (replace these with your actual values)
input_data = {
    "profile_name": "Stomme",
    "weight_kg_per_m": 0.39,
    "length_m": 23.2,
    "alloy": "EN-AW-6063",
    "temper": "T5",
    "tolerance": "EN 755-9",
    "surface_treatment": "Powder Coated",
    "lme_price_eur_per_kg": 3.34,
    "date": "2025-01-28"
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Create 'month', 'quarter', 'year' from date
input_df["date"] = pd.to_datetime(input_df["date"])
input_df["month"] = input_df["date"].dt.month
input_df["quarter"] = input_df["date"].dt.quarter
input_df["year"] = input_df["date"].dt.year
input_df = input_df.drop("date", axis=1)

# Combine alloy and temper (same as in training)
input_df["alloy_temp"] = input_df["alloy"] + "-" + input_df["temper"]
input_df = input_df.drop(["alloy", "temper"], axis=1)

# One-hot encoding (same as training)
input_df = pd.get_dummies(input_df)

# 🔒 Ensure columns match model training
# Load the columns used during training
columns_used = joblib.load("model_columns.pkl")  # <-- We need to save this in training

# Add missing columns with 0, and reorder
for col in columns_used:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns_used]  # Reorder columns

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
predicted_price = model.predict(scaled_input)[0]
print(f"✅ Predicted Price: {predicted_price:.2f} SEK")
