# Real-life Use Case: Bank Loan Approval Prediction
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Sample training dataset (Income, Age, Loan Amount)
X_train = np.array([
    [50000, 25, 200000],
    [60000, 30, 250000],
    [80000, 35, 300000],
    [120000, 40, 400000],
    [70000, 28, 280000]
])
y_train = np.array([0, 0, 1, 1, 0])  # 0=Not Approved, 1=Approved

# New user input
income = float(input("Enter Income (₹): "))
age = float(input("Enter Age (years): "))
loan_amount = float(input("Enter Loan Amount (₹): "))
X_new = np.array([[income, age, loan_amount]])

print("\nOriginal Input Features:", X_new)

# 1️⃣ Apply StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_new_std = scaler_std.transform(X_new)

# 2️⃣ Apply MinMaxScaler
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_new_mm = scaler_mm.transform(X_new)

# Train Logistic Regression model on Standard Scaled Data
model_std = LogisticRegression()
model_std.fit(X_train_std, y_train)
pred_std = model_std.predict(X_new_std)

# Train Logistic Regression model on MinMax Scaled Data
model_mm = LogisticRegression()
model_mm.fit(X_train_mm, y_train)
pred_mm = model_mm.predict(X_new_mm)

print("\nPredicted Approval (StandardScaler):", "Approved" if pred_std[0]==1 else "Not Approved")
print("Predicted Approval (MinMaxScaler):", "Approved" if pred_mm[0]==1 else "Not Approved")

print("\nScaled Input Features (StandardScaler):", X_new_std)
print("Scaled Input Features (MinMaxScaler):", X_new_mm)
