import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'],
    'Size': ['Small', 'Medium', 'Large', 'Large', 'Small', 'Medium'],
    'Price': [5, 6, 7, 8, 5.5, 7.5]
}
df = pd.DataFrame(data)

# Encoders
le = LabelEncoder()
df['Size_encoded'] = le.fit_transform(df['Size'])
ohe = OneHotEncoder(sparse_output=False)
color_encoded = ohe.fit_transform(df[['Color']])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['Color']))

# Model training
X = pd.concat([color_df, df[['Size_encoded']]], axis=1)
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Interactive input loop
while True:
    color_input = input("Enter Color (Red/Green/Blue) or 'q' to quit: ")
    if color_input.lower() == 'q':
        break
    size_input = input("Enter Size (Small/Medium/Large): ")
    
    try:
        # Encode input
        new_color = ohe.transform(pd.DataFrame({'Color':[color_input]}))
        new_color_df = pd.DataFrame(new_color, columns=ohe.get_feature_names_out(['Color']))
        new_size = le.transform([size_input])
        X_new = pd.concat([new_color_df, pd.DataFrame({'Size_encoded': new_size})], axis=1)
        
        # Predict
        predicted_price = model.predict(X_new)[0]
        print(f"Predicted Price for {color_input}, {size_input}: {predicted_price}\n")
    except Exception as e:
        print("Invalid input, please try again.\n")
