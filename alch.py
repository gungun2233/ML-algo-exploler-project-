import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

def train_model(X, y, algorithm):
    if algorithm == 'Linear Regression': 
        model = LinearRegression()
    elif algorithm == 'Logistic Regression':
        model = LogisticRegression()
    model.fit(X, y)
    return model

def make_prediction(model, feature_values):
    prediction = model.predict([feature_values])
    return prediction[0]

# File upload
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the dataset
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        

    st.write(data.head())

    # Display column names
    st.write("Column names:", data.columns.tolist())

    # Get feature column names from the user
    feature_cols = st.multiselect("Select the feature columns", data.columns.tolist())

    # Get target column name from the user
    target_col = st.selectbox("Select the target column", data.columns.tolist())

    # Get algorithm choice from the user
    algorithm_choice = st.selectbox("Choose the algorithm", ["Linear Regression", "Logistic Regression"])

    # Check if the selected columns and algorithm are valid
    valid_columns = len(feature_cols) > 0 and target_col in data.columns and target_col not in feature_cols
    target_is_continuous = data[target_col].dtype.kind in 'bifc'  # Check if target is continuous

    if valid_columns and (algorithm_choice == 'Linear Regression') == target_is_continuous:
        # Select the features and target variable
        X = data[feature_cols]
        y = data[target_col]

        # Train the model section
        st.subheader("Train the Model")
        train_model_button = st.button("Train the Model")

        if 'model' not in st.session_state:
            st.session_state['model'] = None

        if train_model_button:
            # Train the model and store it in a session state
            st.session_state['model'] = train_model(X, y, algorithm_choice)
            st.write(f"{algorithm_choice} model trained successfully.")

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
        if not valid_columns:
            st.write("Invalid column selection. Please select at least one feature column and one target column (different from feature columns).")
        else:
            if algorithm_choice == 'Linear Regression' and not target_is_continuous:
                st.write("Linear Regression is not applicable for categorical target variables. Please choose Logistic Regression or select a continuous target variable.")
            elif algorithm_choice == 'Logistic Regression' and target_is_continuous:
                st.write("Logistic Regression is not applicable for continuous target variables. Please choose Linear Regression or select a categorical target variable.")