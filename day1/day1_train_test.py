from sklearn.model_selection import train_test_split
import pandas as pd

data = {
    "Experience": [1,2,3,4,5],
    "Salary": [20000,30000,40000,50000,60000]
}

df = pd.DataFrame(data)

X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:\n", X_train)
print("Testing Data:\n", X_test)