import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load the Seeds Dataset
@st.cache_data
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
    column_names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel', 'Width of kernel', 'Asymmetry coefficient', 'Length of kernel groove', 'Target']
    data = pd.read_csv(url, delim_whitespace=True, names=column_names)
    return data

data = load_data()

# Display the first few rows of the dataset
st.title("K-Means Clustering of Seeds Data")
st.write("### Seeds Dataset")
st.dataframe(data.head())

# Step 2: Preprocess the Data
X = data.drop(columns='Target')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Perform K-Means Clustering
# Allow users to select the number of clusters
n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original data
data['Cluster'] = y_kmeans

# Step 4: Visualize the Results
# Plotting the clusters for two dimensions (e.g., 'Area' vs. 'Perimeter')
st.write("### Clustering Results")

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

for i in range(n_clusters):
    ax.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], s=100, c=colors[i % len(colors)], label=f'Cluster {i + 1}')
ax.set_title('K-Means Clustering of Seeds Data')
ax.set_xlabel('Area')
ax.set_ylabel('Perimeter')
ax.legend()

st.pyplot(fig)

# Additional insights: displaying cluster centers
st.write("### Cluster Centers")
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
st.dataframe(cluster_centers_df)
