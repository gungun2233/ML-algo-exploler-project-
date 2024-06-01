import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'age': [25, 30, 35, 40, 45],
    'premium': [18000, 32000, 42000, 47000, 55000]
}

df = pd.DataFrame(data)
reg = LinearRegression()
X = df[['age']]  # Feature: age
y = df['premium']  # Target: premium
reg.fit(X, y)

# Streamlit App to Predict Insurance Premium
st.title("Predict Insurance Premium")

# Input field to enter age for prediction
input_age = st.number_input("Enter Age:", min_value=0, max_value=100, value=30, step=1)

# Button to trigger prediction
if st.button("Predict Premium"):
    # Predict premium based on input age
    predicted_premium = reg.predict([[input_age]])
    st.success(f"Predicted Premium for Age {input_age}: ${predicted_premium[0]:.2f}")
