import streamlit as st
import subprocess

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
            run_project(selected_algorithm)

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
            run_project(selected_algorithm)

    # Displaying link for the selected algorithm
    if learning_type == "Supervised Learning":
        st.markdown(f"[Link to Documentation]({supervised_algorithms[selected_algorithm]})")
    elif learning_type == "Unsupervised Learning":
        st.markdown(f"[Link to Documentation]({unsupervised_algorithms[selected_algorithm]})")

def run_project(selected_algorithm):
    # Run specific project based on the selected algorithm
    if selected_algorithm == "Linear Regression":
        subprocess.run(["streamlit", "run", "app.py"])
    elif selected_algorithm == "Logistic Regression":
        subprocess.run(["streamlit", "run", "new.py"])
    elif selected_algorithm == "Random Forest":
        subprocess.run(["streamlit", "run", "rand.py"])
    elif selected_algorithm == "K-Nearest Neighbors (KNN)":
        subprocess.run(["streamlit", "run", "knn.py"])
    elif selected_algorithm == "Support Vector Machines (SVM)":
        subprocess.run(["streamlit", "run", "svm.py"])
    elif selected_algorithm == "K-Means Clustering":
        subprocess.run(["streamlit", "run", "kmeans.py"])
    elif selected_algorithm == "Hierarchical Clustering":
        subprocess.run(["streamlit", "run", "heir.py"])
    elif selected_algorithm == "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)":
        subprocess.run(["streamlit", "run", "dbscan.py"])
    elif selected_algorithm == "Principal Component Analysis (PCA)":
        subprocess.run(["streamlit", "run", "pca.py"])
    elif selected_algorithm == "t-Distributed Stochastic Neighbor Embedding (t-SNE)":
        subprocess.run(["streamlit", "run", "tsne.py"])

if __name__ == "__main__":
    main()
