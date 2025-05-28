# 🏗️ Aluminum Quote Price Prediction App

A machine learning web application built with **FastAPI** (backend) and **React.js** (frontend) that predicts the price per aluminum piece based on material, dimensions, and treatment.

---

## 🚀 Live Demo

https://odens-assignmnet-frontend-aseu.vercel.app/

---

## 🎯 Features

- Predicts `price_per_piece` based on:
  - Profile name
  - Weight and length
  - Alloy and temper
  - Tolerance and surface treatment
  - LME price and quote date



## 🧠 Machine Learning Details

- **Algorithm**: `RandomForestRegressor`
- **Preprocessing**:
  - Categorical encoding via `pd.get_dummies()`
  - Date transformed into year, month, and quarter
  - Features scaled using `StandardScaler`
