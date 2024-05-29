import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict function
def predict(query):
    return knn.predict([query])[0]

# Evaluate accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy}")

# Streamlit App
st.title('KNN Classifier')

# Sidebar
st.sidebar.header('User Input')

# Collect user input features
sepal_length = st.sidebar.slider('Sepal Length', float(iris.data[:, 0].min()), float(iris.data[:, 0].max()))
sepal_width = st.sidebar.slider('Sepal Width', float(iris.data[:, 1].min()), float(iris.data[:, 1].max()))
petal_length = st.sidebar.slider('Petal Length', float(iris.data[:, 2].min()), float(iris.data[:, 2].max()))
petal_width = st.sidebar.slider('Petal Width', float(iris.data[:, 3].min()), float(iris.data[:, 3].max()))

query = [sepal_length, sepal_width, petal_length, petal_width]

# Make prediction
prediction = predict(query)

species = iris.target_names[prediction]

# Display prediction
st.subheader('Prediction')
st.write(f'The predicted species is {species}')
