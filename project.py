import streamlit as st
import subprocess

def main():
    st.title("Machine Learning Algorithm Explorer")

    st.subheader("Select a supervised learning algorithm:")
    supervised_algorithms = ["Linear Regression", "Logistic Regression", "Random Forest"]
    selected_algorithm = st.selectbox("Choose an Algorithm", supervised_algorithms)
        
    # Run specific project based on selected algorithm
    if st.button("Run Project"):
        if selected_algorithm == "Linear Regression":
            run_project("app.py")
        elif selected_algorithm == "Logistic Regression":
            run_project("new.py")
        elif selected_algorithm == "Random Forest":
            run_project("rand.py")

def run_project(file_name):
    # Use subprocess to run the specified Python script containing the project
    command = f"streamlit run {file_name}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        st.error(f"Error encountered: {stderr.decode('utf-8')}")
    else:
        st.success(f"Project executed successfully!")

if __name__ == "__main__":
    main()
