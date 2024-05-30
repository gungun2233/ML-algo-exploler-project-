import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load the Digits dataset
digits = load_digits()
df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
df['target'] = digits.target

# Streamlit app
st.title('PCA on Digits Dataset')
st.write('This is a demonstration of PCA using the Digits dataset.')

# Sidebar for user input
st.sidebar.header('User Input Features')
all_features = list(df.columns[:-1])
selected_features = st.sidebar.multiselect('Select features for PCA:', all_features)

# Button to perform PCA
if st.sidebar.button('Perform PCA'):
    if selected_features:
        # Perform PCA based on user-selected features
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df[selected_features])
        pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
        pca_df['Target'] = df['target']

        # Scatter plot of the PCA result
        fig, ax = plt.subplots()
        targets = range(10)
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']

        for target, color in zip(targets, colors):
            indices = pca_df['Target'] == target
            ax.scatter(pca_df.loc[indices, 'Principal Component 1'], pca_df.loc[indices, 'Principal Component 2'], c=color, s=50)

        ax.legend(targets)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2 Component PCA')

        st.pyplot(fig)

        # Display the PCA dataframe
        st.write('PCA DataFrame')
        st.write(pca_df)
    else:
        st.write("Please select at least one feature to perform PCA.")
