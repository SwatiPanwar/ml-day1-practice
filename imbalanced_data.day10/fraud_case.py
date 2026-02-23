# fraud_case_corrected.py

# Step 1: Import libraries
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np

# Step 2: Create imbalanced dataset (simulate credit card transactions)
# 0 = Normal transaction, 1 = Fraud
X, y = make_classification(
    n_samples=100,       # total 100 transactions
    n_features=2,        # features: TransactionAmount, TransactionTime
    n_informative=2,     # both features are informative
    n_redundant=0,       # no redundant features
    n_repeated=0,        # no repeated features
    n_classes=2,         # binary classification
    weights=[0.9, 0.1],  # 90% normal, 10% fraud
    random_state=42
)

# Step 3: Convert to DataFrame for user-friendly display
df = pd.DataFrame(X, columns=['TransactionAmount', 'TransactionTime'])
df['Label'] = y

print("Original dataset class distribution:", Counter(y))
print("\nSample Transactions (first 5 rows):")
print(df.head())

# Step 4: Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("\nAfter applying SMOTE, class distribution:", Counter(y_res))

# Step 5: User input simulation
print("\n--- User Transaction Simulation ---")
try:
    user_amount = float(input("Enter Transaction Amount: "))
    user_time = float(input("Enter Transaction Time (hour): "))
except ValueError:
    print("Invalid input! Enter numeric values.")
    exit()

# Step 6: Simple nearest-neighbor logic to explain prediction
distances = np.sqrt(np.sum((X_res - np.array([user_amount, user_time]))**2, axis=1))
nearest_index = np.argmin(distances)
predicted_label = y_res[nearest_index]

print(f"\nBased on SMOTE balanced data, this transaction is predicted as:",
      "Fraud" if predicted_label == 1 else "Normal")
