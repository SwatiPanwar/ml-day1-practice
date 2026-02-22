# 1️⃣ Libraries import
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 2️⃣ Data load
iris = load_iris()
X = iris.data      # features
y = iris.target    # labels

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4️⃣ Decision Tree without limit (overfitting possible)
dt_overfit = DecisionTreeClassifier(random_state=42)
dt_overfit.fit(X_train, y_train)

# 5️⃣ Decision Tree with depth limit (less overfitting)
dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)

# 6️⃣ Accuracy check
print("Overfit tree train accuracy:", dt_overfit.score(X_train, y_train))
print("Overfit tree test accuracy:", dt_overfit.score(X_test, y_test))

print("Limited tree train accuracy:", dt_limited.score(X_train, y_train))
print("Limited tree test accuracy:", dt_limited.score(X_test, y_test))

# 7️⃣ Visualize the limited tree
plt.figure(figsize=(12,8))
plot_tree(dt_limited, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
