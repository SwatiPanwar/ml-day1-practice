from sklearn.metrics import mean_squared_error
import numpy as np

y_actual = [5000000, 7000000, 4000000]
y_pred = [5200000, 6800000, 4100000]

mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)  # manually RMSE calculate
print("MSE =", mse)
print("RMSE =", rmse)