import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import urllib.request
import folium
import numpy as np

# Download the crime data from the Chicago Data Portal
url = "https://data.cityofchicago.org/resource/6zsd-86xi.csv"
response = urllib.request.urlopen(url)
data = response.read().decode('utf-8').split('\n')[1:-1]  # Skip the header row and the last empty row

# Create a DataFrame from the data
columns = ['id', 'case_number', 'date', 'block', 'iucr', 'primary_type', 'description', 'location_description', 'arrest', 'domestic', 'beat', 'district', 'ward', 'community_area', 'fbi_code', 'x_coordinate', 'y_coordinate', 'year', 'updated_on', 'latitude', 'longitude', 'location', 'human_address', 'x_coordinate_geo', 'y_coordinate_geo', 'latitude_geo', 'longitude_geo', 'geolocation', 'location_geo']
chicago_crime_df = pd.DataFrame([row.split(',') for row in data], columns=columns)

# Remove double quotes from x_coordinate, y_coordinate, latitude, and longitude columns
chicago_crime_df[['x_coordinate', 'y_coordinate', 'latitude', 'longitude']] = chicago_crime_df[['x_coordinate', 'y_coordinate', 'latitude', 'longitude']].replace('"', '', regex=True)

# Replace empty strings with NaN for x_coordinate, y_coordinate, latitude, and longitude columns
chicago_crime_df[['x_coordinate', 'y_coordinate', 'latitude', 'longitude']] = chicago_crime_df[['x_coordinate', 'y_coordinate', 'latitude', 'longitude']].replace('', np.nan)

# Convert relevant columns to numeric data types and handle date-time strings
chicago_crime_df = chicago_crime_df.astype({'x_coordinate': float, 'y_coordinate': float})
chicago_crime_df[['latitude', 'longitude']] = chicago_crime_df[['latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')

# Streamlit app
st.title("Crime Hotspot Visualization")

# Get user input for DBSCAN parameters
eps = st.sidebar.slider("Epsilon (eps)", 0.0001, 0.01, 0.001, 0.0001)
min_samples = st.sidebar.slider("Minimum Samples (min_samples)", 5, 50, 10, 1)

# Prediction button
if st.button("Predict Hotspots"):
    # Create a new DataFrame with only valid latitude and longitude values
    valid_coords_df = chicago_crime_df[['latitude', 'longitude']].dropna()

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(valid_coords_df)

    # Add the cluster labels to the original DataFrame
    chicago_crime_df = chicago_crime_df.merge(pd.DataFrame(clusters, index=valid_coords_df.index, columns=['cluster']), how='left', left_index=True, right_index=True)

    # Create a map centered on Chicago
    chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Add markers for each crime incident, colored by cluster
    for index, row in chicago_crime_df.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            marker_color = 'red' if row['cluster'] == -1 else 'blue'
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=marker_color,
                fill=True
            ).add_to(chicago_map)

    # Display the map in Streamlit
    map_html = chicago_map._repr_html_()
    st.components.v1.html(map_html, height=500, width=800)

    # Analyze the clusters
    cluster_summary = chicago_crime_df.groupby('cluster').agg({
        'latitude': ['mean'],
        'longitude': ['mean'],
        'primary_type': ['count']
    }).reset_index()

    st.subheader("Cluster Summary")
    st.write(cluster_summary)