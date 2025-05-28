import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


df = pd.read_csv("preprocessed_quotes.csv")


X = df.drop("price_per_piece", axis=1)
y = df["price_per_piece"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# data splitinggg
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print( f"MAE: {mae:.3f}")
print(f" RMSE: {rmse:.3f}")
print(f" R² Score: {r2:.3f}")

# Save the model
joblib.dump(model, "price_predictor_model.pkl")



joblib.dump(list(X.columns), "model_columns.pkl")
# print("saved")
