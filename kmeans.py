import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit page
st.title('Customer Segmentation using K-means Clustering')
st.write('This app segments customers based on their age and income.')

# Input fields for customer data
st.subheader('Enter Customer Data')
ages = st.text_input('Enter ages (comma-separated)', '25,34,45,23,35,64,33,45,32,55')
incomes = st.text_input('Enter incomes (comma-separated)', '50000,60000,80000,120000,30000,40000,70000,90000,60000,100000')

# Convert input strings to lists
age_list = [int(x) for x in ages.split(',')]
income_list = [int(x) for x in incomes.split(',')]

# Create DataFrame from input data
data = {'Age': age_list, 'Income': income_list}
df = pd.DataFrame(data)

# Display the data
st.subheader('Customer Data')
st.write(df)

# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Slider to select number of clusters
k = st.slider('Select number of clusters', 2, 5, 3)

# Button to perform clustering
if st.button('Perform Clustering'):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # Visualize the clusters
    st.subheader('Cluster Visualization')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Income', hue='Cluster', data=df, palette='viridis', ax=ax)
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Customer Segmentation')
    st.pyplot(fig)
