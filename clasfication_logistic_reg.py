from sklearn.linear_model import LogisticRegression
import numpy as np

# Hours studied vs Pass(1)/Fail(0)
X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([0,0,0,1,1])

model = LogisticRegression()
model.fit(X, y)

print("Prediction for 3.5 hours:", model.predict([[3.5]]))