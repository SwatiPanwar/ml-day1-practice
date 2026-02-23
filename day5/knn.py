# Step 1: Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Sample Dataset
# Features: [Number of links, Email length]
X = np.array([
    [1, 50],
    [2, 60],
    [10, 300],
    [12, 350],
    [3, 70],
    [15, 400],
    [2, 55],
    [11, 320]
])

# Labels
y = np.array(["Not Spam", "Not Spam", "Spam", "Spam",
              "Not Spam", "Spam", "Not Spam", "Spam"])

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Feature Scaling (Important in KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
