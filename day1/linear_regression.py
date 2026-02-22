from sklearn.linear_model import LinearRegression
import numpy as np

# Hours studied vs marks
X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,5,4,5])

model = LinearRegression()
model.fit(X, y)

print("Predicted marks for 6 hours:", model.predict([[6]]))