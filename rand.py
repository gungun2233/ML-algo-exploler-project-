import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
@st.cache_data  # Cache the data for faster reload
def load_data():
    iris_df = pd.read_csv("iris.csv")
    return iris_df

# Create a Random Forest classifier
def train_model():
    iris_df = load_data()
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = iris_df['species']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Define the main function for the Streamlit app
def main():
    st.title("Iris Species Prediction with Random Forest")

    # Load the data
    iris_df = load_data()

    # Display a sample of the data
    if st.checkbox("Show Dataset Sample"):
        st.write(iris_df.head())

    # Train the model
    model = train_model()

    # User input for prediction
    st.sidebar.header("Input Features")
    sepal_length = st.sidebar.slider("Sepal Length", float(iris_df['sepal_length'].min()), float(iris_df['sepal_length'].max()), float(iris_df['sepal_length'].mean()))
    sepal_width = st.sidebar.slider("Sepal Width", float(iris_df['sepal_width'].min()), float(iris_df['sepal_width'].max()), float(iris_df['sepal_width'].mean()))
    petal_length = st.sidebar.slider("Petal Length", float(iris_df['petal_length'].min()), float(iris_df['petal_length'].max()), float(iris_df['petal_length'].mean()))
    petal_width = st.sidebar.slider("Petal Width", float(iris_df['petal_width'].min()), float(iris_df['petal_width'].max()), float(iris_df['petal_width'].mean()))

    # Prediction button
    if st.sidebar.button("Make Prediction"):
        # Make prediction
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        predicted_species = prediction[0]

        # Display prediction result
        st.subheader("Prediction")
        st.write(f"The predicted species is: {predicted_species}")

# Run the app
if __name__ == "__main__":
    main()
