import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    # Train the model
    model = LogisticRegression()
    model.fit(X, y)
    return model

def make_prediction(model, feature_values):
    # Make prediction
    prediction = model.predict([feature_values])
    return prediction[0]

# File upload
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the dataset based on file type
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write(data.head())

    # Display column names
    st.write("Column names:", data.columns.tolist())

    # Get feature column names from the user
    feature_cols = st.multiselect("Select the feature columns", data.columns.tolist())

    # Get target column name from the user
    target_col = st.selectbox("Select the target column", data.columns.tolist())

    # Check if the selected columns are valid
    valid_columns = len(feature_cols) > 0 and target_col in data.columns

    if valid_columns:
        # Select the features and target variable
        X = data[feature_cols]
        y = data[target_col]

        # Train the model section
        st.subheader("Train the Model")
        train_model_button = st.button("Train the Model")

        if 'model' not in st.session_state:
            st.session_state['model'] = None

        if train_model_button:
            # Train the model and store it in the session state
            st.session_state['model'] = train_model(X, y)
            st.write("Model trained successfully.")

        if st.session_state['model'] is not None:
            # Prediction section
            st.subheader("Make Prediction")
            feature_values = []
            for i, col in enumerate(feature_cols):
                feature_value = st.number_input(f"Enter value for {col}", key=f"feature_{i}")
                feature_values.append(feature_value)

            if st.button("Predict"):
                # Make prediction using the trained model
                prediction = make_prediction(st.session_state['model'], feature_values)
                st.write(f"The predicted value is: {prediction}")
    else:
        st.write("Invalid column selection. Please check and try again.")
