import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load preprocessed dataset
df = pd.read_csv("preprocessed_quotes.csv")

# Step 2: Separate features and target
X = df.drop("price_per_piece", axis=1)
y = df["price_per_piece"]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for use in prediction script
joblib.dump(scaler, "scaler.pkl")

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 5: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model training complete.")
print(f"📊 MAE: {mae:.3f}")
print(f"📊 RMSE: {rmse:.3f}")
print(f"📈 R² Score: {r2:.3f}")

# Step 7: Save the model
joblib.dump(model, "price_predictor_model.pkl")
print("\n📦 Model saved as 'price_predictor_model.pkl'")
print("📐 Scaler saved as 'scaler.pkl'")

# ✅ Step 8: Save column names
joblib.dump(list(X.columns), "model_columns.pkl")
print("📄 Feature columns saved as 'model_columns.pkl'")
