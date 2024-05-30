import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to load selected dataset
def load_dataset(dataset_name):
    if dataset_name == "Iris":
        return load_iris()
    elif dataset_name == "Wine":
        return load_wine()
    elif dataset_name == "Breast Cancer":
        return load_breast_cancer()

# Function to perform t-SNE on the selected dataset
def run_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    return X_embedded

# Function to visualize t-SNE embeddings
def plot_tsne(X_embedded, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar()
    st.pyplot()

def main():
    st.title("t-SNE Visualization")

    # Select dataset
    dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer"])

    # Load selected dataset
    dataset = load_dataset(dataset_name)
    X = dataset.data
    y = dataset.target

    # Feature selection
    selected_features = st.sidebar.multiselect("Select Features", dataset.feature_names)

    # Perform t-SNE
    if st.sidebar.button("Run t-SNE"):
        if len(selected_features) > 0:
            selected_indices = [list(dataset.feature_names).index(feature) for feature in selected_features]
            X_selected = X[:, selected_indices]
            X_embedded = run_tsne(X_selected, y)
            plot_tsne(X_embedded, y)
        else:
            st.write("Please select at least one feature.")

if __name__ == "__main__":
    main()
