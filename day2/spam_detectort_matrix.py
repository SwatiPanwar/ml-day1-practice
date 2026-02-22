# Step 1: Import libraries
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# Step 2: Sample data (0 = Not Spam, 1 = Spam)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Actual labels
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Model predictions

# Step 3: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# Format: [[TN, FP], [FN, TP]]

# Step 4: Precision, Recall, F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
