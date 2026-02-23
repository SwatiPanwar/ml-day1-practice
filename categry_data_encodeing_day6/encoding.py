# 1️⃣ Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 2️⃣ Sample Dataset
data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'],
    'Size': ['Small', 'Medium', 'Large', 'Large', 'Small', 'Medium'],
    'Price': [5, 6, 7, 8, 5.5, 7.5]  # Target variable
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# 3️⃣ Encode Color using One-Hot Encoding (Unordered)
ohe = OneHotEncoder(sparse_output=False)
color_encoded = ohe.fit_transform(df[['Color']])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['Color']))

# 4️⃣ Encode Size using Label Encoding (Ordered)
le = LabelEncoder()
df['Size_encoded'] = le.fit_transform(df['Size'])

# 5️⃣ Prepare final dataset
X = pd.concat([color_df, df[['Size_encoded']]], axis=1)
y = df['Price']

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7️⃣ Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 8️⃣ Predict
predictions = model.predict(X_test)
print("\nPredictions:", predictions)
