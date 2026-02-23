# Step 0: Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Sample Dataset (simulate telecom data)
data = pd.DataFrame({
    'tenure': [1, 5, 12, 24, 3, 8, 18, 30],
    'monthly_charges': [50, 70, 80, 60, 55, 75, 65, 85],
    'contract': [0, 1, 1, 2, 0, 1, 2, 2],  # 0=month-to-month, 1=one-year, 2=two-year
    'churn': [1, 0, 0, 0, 1, 0, 0, 0]      # target: 1=churn, 0=stay
})

X = data[['tenure','monthly_charges','contract']]
y = data['churn']

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Define model + GridSearchCV
model = LogisticRegression(max_iter=200)
param_grid = {'C':[0.1,1,10], 'solver':['liblinear','lbfgs']}
grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')  # 3-fold CV for example
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Step 5: Test set evaluation
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: User Input Prediction
print("\n--- Predict Customer Churn ---")
tenure = float(input("Enter tenure (months): "))
monthly_charges = float(input("Enter monthly charges ($): "))
contract = int(input("Enter contract type (0=month,1=1yr,2=2yr): "))

user_data = scaler.transform([[tenure, monthly_charges, contract]])
prediction = best_model.predict(user_data)

if prediction[0]==1:
    print("Prediction: Customer is likely to CHURN 🔴")
else:
    print("Prediction: Customer is likely to STAY 🟢")
