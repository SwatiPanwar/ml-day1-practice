# Day 2 – Linear Regression + MSE + RMSE

## 1️⃣ Linear Regression

Goal: Continuous value predict karna (e.g., House Price)

Formula:
y = b0 + b1x

b0 = intercept
b1 = slope
x = feature
y = prediction

Model best fit line banata hai.

---

## 2️⃣ Model Flow

1. Data load karo
2. X (features) aur y (target) separate karo
3. Model train karo (fit)
4. Prediction lo
5. Error calculate karo

---

## 3️⃣ Error

Error = Actual - Predicted

---

## 4️⃣ MSE (Mean Squared Error)

Steps:
1. Difference nikalo
2. Square karo
3. Add karo
4. Divide by total samples

MSE = average squared error

Trick: D → S → A → D
(Difference → Square → Add → Divide)

---

## 5️⃣ RMSE (Root Mean Squared Error)

RMSE = √MSE

Same unit me error deta hai.
Easy to understand.

---

## Difference

MSE → squared unit
RMSE → real unit

Lower value = better model
