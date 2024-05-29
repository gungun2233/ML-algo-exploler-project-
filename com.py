import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :3]  # Consider only sepal length, sepal width, petal length
y_species = iris.target  # Species (target for classification)
y_petal_length = iris.data[:, 2]  # Petal length (target for regression)

# Split data into training and testing sets
X_train, X_test, y_species_train, y_species_test, y_petal_length_train, y_petal_length_test = train_test_split(
    X, y_species, y_petal_length, test_size=0.2, random_state=42
)

# Define the model (Random Forest Regressor as an example)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# MultiOutputRegressor for both classification and regression
multioutput_regressor = MultiOutputRegressor(regressor)

# Prepare target variables for MultiOutputRegressor
y_train = list(zip(y_species_train, y_petal_length_train))

# Train the model
multioutput_regressor.fit(X_train, y_train)

# Define Streamlit app title
st.title("Iris Species and Petal Length Predictor")

# Define sidebar inputs (user input)
st.sidebar.header("Enter Iris Measurements")

sepal_length = st.sidebar.slider("Sepal Length", float(np.min(X[:, 0])), float(np.max(X[:, 0])), float(np.mean(X[:, 0])))
sepal_width = st.sidebar.slider("Sepal Width", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))
petal_length = st.sidebar.slider("Petal Length", float(np.min(X[:, 2])), float(np.max(X[:, 2])), float(np.mean(X[:, 2])))

# Predict function based on user input
def predict_species_and_petal_length(sepal_length, sepal_width, petal_length):
    input_features = [[sepal_length, sepal_width, petal_length]]
    species_pred, petal_length_pred = multioutput_regressor.predict(input_features)[0]
    species_name = iris.target_names[int(species_pred)]
    return species_name, petal_length_pred

# Display prediction based on user input
if st.sidebar.button("Predict"):
    species, predicted_petal_length = predict_species_and_petal_length(sepal_length, sepal_width, petal_length)
    st.write(f"Predicted Species: {species}")
    st.write(f"Predicted Petal Length: {predicted_petal_length:.2f}")
