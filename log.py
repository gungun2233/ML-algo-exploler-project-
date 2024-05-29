import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load the dataset
dataset = pd.read_excel(r"C:\Users\Asus\OneDrive\Desktop\logistics.xlsx")
dataset['Bought insurance'].replace({'no': 0, 'yes': 1}, inplace=True)

# Split data into features (X) and target (y)
x = dataset[['Age']]
y = dataset['Bought insurance']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train a logistic regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Define the Streamlit app
def main():
    st.title('Insurance Purchase Prediction')

    # Prediction input form
    st.subheader('Make a Prediction')
    age = st.number_input('Enter Age:', min_value=0, max_value=100, value=30)

    # Prediction button
    if st.button('Predict'):
        prediction = lr.predict([[age]])

        # Show prediction result
        if prediction[0] == 1:
            st.write('The person can buy insurance.')
        else:
            st.write('The person can not buy insurance.')

if __name__ == '__main__':
    main()
