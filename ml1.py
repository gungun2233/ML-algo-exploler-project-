import streamlit as st
import subprocess
import webbrowser

def main():
    st.title("Machine Learning Algorithm Explorer")

    # Display initial options for supervised or unsupervised learning
    learning_type = st.radio("Select Learning Type", ("Supervised Learning", "Unsupervised Learning"))

    if learning_type == "Supervised Learning":
        st.subheader("Select a supervised learning algorithm:")
        supervised_algorithms = {
            "Linear Regression": "https://gungun2233-crw-app-awvgyv.streamlit.app/",
            "Logistic Regression": "https://gungun2233-crw-new-h8t8u4.streamlit.app/",
            "Random Forest": "https://gungun2233-crw-rand-esbmul.streamlit.app/",
            "K-Nearest Neighbors (KNN)": "https://gungun2233-crw-knn-tgy1kn.streamlit.app/",
            "Support Vector Machines (SVM)": "https://gungun2233-crw-svm-3bhdhk.streamlit.app/"
        }
        selected_algorithm = st.selectbox("Choose an Algorithm", list(supervised_algorithms.keys()))
        
        # Run specific project based on selected algorithm
        if st.button("Run Project"):
            open_url(supervised_algorithms[selected_algorithm])

    elif learning_type == "Unsupervised Learning":
        st.subheader("Select an unsupervised learning algorithm:")
        unsupervised_algorithms = {
            "K-Means Clustering": "https://gungun2233-crw-kmeans-opyfdf.streamlit.app/",
            "Hierarchical Clustering": "https://gungun2233-crw-heir-5y2iu8.streamlit.app/",
            "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)": "https://gungun2233-crw-dbscan-jd22el.streamlit.app/",
            "Principal Component Analysis (PCA)": "https://gungun2233-crw-pca-01b1kf.streamlit.app/",
            "t-Distributed Stochastic Neighbor Embedding (t-SNE)": "https://gungun2233-crw-tsne-n1s2fc.streamlit.app/"
        }
        selected_algorithm = st.selectbox("Choose an Algorithm", list(unsupervised_algorithms.keys()))
        
        # Run specific project based on selected algorithm
        if st.button("Run Project"):
            open_url(unsupervised_algorithms[selected_algorithm])

def open_url(url):
    # Open the specified URL in the default web browser
    webbrowser.open_new_tab(url)

if __name__ == "__main__":
    main()
