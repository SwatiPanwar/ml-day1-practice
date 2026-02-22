# Python Example - Random Forest Feature Importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importance
importance = rf.feature_importances_
for i, v in enumerate(importance):
    print(f"Feature: {X.columns[i]}, Importance: {v:.3f}")

# Visualization
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.show()
