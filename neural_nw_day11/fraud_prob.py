import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

# -----------------------------
# Step 1: Dummy Training Data
# -----------------------------
# Features: [TransactionAmount, TransactionTime]
X_train = np.array([[100, 10], [200, 50], [5000, 5], [50, 60], [4000, 20]])
# Labels: 0 = Normal, 1 = Fraud
y_train = np.array([0, 0, 1, 0, 1])

# -----------------------------
# Step 2: Build Model
# -----------------------------
model = Sequential([
    layers.Dense(4, activation='relu', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')  # Output = probability
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# Step 3: Train Model
# -----------------------------
model.fit(X_train, y_train, epochs=50, verbose=0)  # verbose=0 → quiet

# -----------------------------
# Step 4: User Input
# -----------------------------
amount = float(input("Enter transaction amount: "))
time = float(input("Enter transaction time (hour 0-23): "))

user_input = np.array([[amount, time]])

# -----------------------------
# Step 5: Predict Fraud Probability
# -----------------------------
prob = model.predict(user_input)[0][0]
prob_percent = prob * 100

print(f"\nFraud Probability: {prob_percent:.2f}%")

if prob > 0.5:
    print("⚠️ Alert: This transaction is likely FRAUD!")
else:
    print("✅ Transaction seems NORMAL.")
