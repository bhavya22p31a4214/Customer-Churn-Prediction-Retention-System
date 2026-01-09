# train_model.py
# Customer Churn Prediction - Model Training

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("data/churn.csv")

print("Dataset loaded successfully")
print("Shape:", data.shape)

# -----------------------------
# 2. Drop unnecessary column
# -----------------------------
if "customerID" in data.columns:
    data.drop("customerID", axis=1, inplace=True)

# -----------------------------
# 3. Handle missing values
# -----------------------------
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"].fillna(data["TotalCharges"].mean(), inplace=True)

# -----------------------------
# 4. Encode categorical columns
# -----------------------------
label_encoder = LabelEncoder()

for col in data.select_dtypes(include=["object"]).columns:
    data[col] = label_encoder.fit_transform(data[col])

# -----------------------------
# 5. Split features & target
# -----------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# -----------------------------
# 6. Logistic Regression Model
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred))

# -----------------------------
# 7. Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

# -----------------------------
# 8. Save Best Model
# -----------------------------
os.makedirs("model", exist_ok=True)

best_model = rf_model if rf_acc > lr_acc else lr_model
joblib.dump(best_model, "model/churn_model.pkl")

print("\nBest model saved successfully at: model/churn_model.pkl")
