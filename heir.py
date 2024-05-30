import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Set up the Streamlit page
st.title('Hierarchical Clustering on Iris Dataset')
st.write('This app performs hierarchical clustering on the Iris dataset.')

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the data
st.subheader('Iris Dataset')
st.write(data.head())

# Select features for clustering
st.subheader('Select Features for Clustering')
features = st.multiselect(
    'Select features',
    options=list(data.columns[:-1]),  # Convert to list explicitly
    default=list(data.columns[:2])  # Convert to list explicitly
)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Slider to select number of clusters
num_clusters = st.slider('Select number of clusters', 2, 5, 3)

# Button to perform clustering
if st.button('Perform Clustering'):
    # Perform hierarchical clustering
    Z = linkage(data_scaled, method='ward')
    
    # Assign cluster labels
    data['Cluster'] = fcluster(Z, num_clusters, criterion='maxclust')

    # Visualize the clusters using a scatter plot
    st.subheader('Cluster Visualization')
    fig, ax = plt.subplots()
    sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=data, palette='viridis', ax=ax)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Iris Dataset Clustering')
    st.pyplot(fig)
    
    # Visualize the dendrogram
    st.subheader('Dendrogram')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, truncate_mode='level', p=5)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    st.pyplot(fig)
