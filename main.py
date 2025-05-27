from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("price_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")
columns_used = joblib.load("model_columns.pkl")

# Define input schema using Pydantic
class QuoteInput(BaseModel):
    profile_name: str
    weight_kg_per_m: float
    length_m: float
    alloy: str
    temper: str
    tolerance: str
    surface_treatment: str
    lme_price_eur_per_kg: float
    date: str

app = FastAPI()

@app.post("/predict")
def predict_price(data: QuoteInput):
    input_df = pd.DataFrame([data.dict()])

    # Date features
    input_df["date"] = pd.to_datetime(input_df["date"])
    input_df["month"] = input_df["date"].dt.month
    input_df["quarter"] = input_df["date"].dt.quarter
    input_df["year"] = input_df["date"].dt.year
    input_df = input_df.drop("date", axis=1)

    # Feature engineering
    input_df["alloy_temp"] = input_df["alloy"] + "-" + input_df["temper"]
    input_df = input_df.drop(["alloy", "temper"], axis=1)

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    for col in columns_used:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns_used]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    predicted_price = model.predict(scaled_input)[0]

    return {"predicted_price": round(predicted_price, 2)}
