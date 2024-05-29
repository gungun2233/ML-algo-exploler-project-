import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the Wine dataset
wine = load_wine()
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['target'] = wine.target

# Split the data into training and testing sets
X = data.drop(columns='target')
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

# Title
st.title("Wine Classification using SVM")

# Explanation of Wine Classes
st.write("""
### Wine Classes
- **Class 0**: Class of wine 0
- **Class 1**: Class of wine 1
- **Class 2**: Class of wine 2
""")

# User input for predictions
st.header("Input Features")
user_input = {}
for feature in wine.feature_names:
    user_input[feature] = st.number_input(feature, value=float(X_train[feature].mean()))

# Convert user input to DataFrame
user_input_df = pd.DataFrame(user_input, index=[0])

# Standardize the user input
user_input_scaled = scaler.transform(user_input_df)

# Predict button
if st.button("Predict"):
    user_prediction = svm.predict(user_input_scaled)
    prediction_label = f"Class {user_prediction[0]}"
    st.write(f"### Prediction: {prediction_label}")
