import streamlit as st
import subprocess

def main():
    st.title("Machine Learning Algorithm Explorer")

    # Display initial options for supervised or unsupervised learning
    learning_type = st.radio("Select Learning Type", ("Supervised Learning", "Unsupervised Learning"))

    if learning_type == "Supervised Learning":
        st.subheader("Select a supervised learning algorithm:")
        supervised_algorithms = ["Linear Regression", "Logistic Regression", "Decision Trees", "Random Forest", "K-Nearest Neighbors (KNN)", "Support Vector Machines (SVM)"]
        selected_algorithm = st.selectbox("Choose an Algorithm", supervised_algorithms)
        
        # Run specific project based on selected algorithm
        if st.button("Run Project"):
            if selected_algorithm == "Linear Regression":
                run_project("app.py")
            elif selected_algorithm == "Logistic Regression":
                run_project("new.py")
            elif selected_algorithm == "Decision Trees":
                run_project("dec.py")
            elif selected_algorithm == "Random Forest":
                run_project("rand.py")
            elif selected_algorithm == "K-Nearest Neighbors (KNN)":
                run_project("knn.py")
            elif selected_algorithm == "Support Vector Machines (SVM)":
                run_project("svm.py")

    elif learning_type == "Unsupervised Learning":
        st.subheader("Select an unsupervised learning algorithm:")
        unsupervised_algorithms = [
            "K-Means Clustering",
            "Hierarchical Clustering",
            "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)",
            "Principal Component Analysis (PCA)",
            "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
           
        ]
        selected_algorithm = st.selectbox("Choose an Algorithm", unsupervised_algorithms)
        
        # Run specific project based on selected algorithm
        if st.button("Run Project"):
            if selected_algorithm == "K-Means Clustering":
                run_project("kmeans.py")
            elif selected_algorithm == "Hierarchical Clustering":
                run_project("heir.py")
            elif selected_algorithm == "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)":
                run_project("dbscan.py")
            elif selected_algorithm == "Principal Component Analysis (PCA)":
                run_project("pca.py")  
            elif selected_algorithm == "t-Distributed Stochastic Neighbor Embedding (t-SNE)":
                run_project("tsne.py")
def run_project(file_name):
    # Use subprocess to run the specified Python script containing the project
    subprocess.run(["streamlit", "run", file_name])

if __name__ == "__main__":
    main()
