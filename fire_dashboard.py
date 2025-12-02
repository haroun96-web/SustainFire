import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
import numpy as np

# ----------------------------------------------
# Page configuration
# ----------------------------------------------
st.set_page_config(page_title="🔥 SustainFire – Forest Fire Prediction Dashboard", layout="wide")

st.title("🔥 SustainFire – Forest Fire Prediction Dashboard")
st.markdown("A prototype to display Random Forest model results using Geomatics and GIS.")

# ----------------------------------------------
# Input file path
# ----------------------------------------------
st.sidebar.header("📁 Select Data File (Local Path)")
shp_path = st.sidebar.text_input("Enter the full path to the SHP file:")

# ----------------------------------------------
# Load Random Forest model
# ----------------------------------------------
MODEL_PATH = "model_fire_rf.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except:
    st.warning("⚠️ No saved Random Forest model found. Using dummy predictions.")
    model_loaded = False

# ----------------------------------------------
# Read data
# ----------------------------------------------
if shp_path:

    if not os.path.exists(shp_path):
        st.error("❌ Invalid path. Please check and try again.")
    else:
        st.success("✔ File found – Loading...")

        # Read Shapefile
        gdf = gpd.read_file(shp_path)

        # Ensure coordinate system is EPSG:4326
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        st.subheader("📊 Sample of the data")
        st.write(gdf.head())

        # ------------------------------------------
        # Predictions
        # ------------------------------------------
        st.subheader("🔥 Predictions")

        if model_loaded:
            # Use numeric columns for prediction
            features = gdf.select_dtypes(include=['float64', 'int64'])
            y_pred = model.predict(features)
        else:
            # Dummy random predictions
            y_pred = np.random.choice([0, 1, 2, 3], size=len(gdf))

        gdf["risk_level"] = y_pred
        st.dataframe(gdf[["risk_level"]].head())

        # ------------------------------------------
        # Determine geometry type
        # ------------------------------------------
        geom_type = gdf.geometry.geom_type.unique()[0]

        # ------------------------------------------
        # Determine map center
        # ------------------------------------------
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        # ------------------------------------------
        # Create map
        # ------------------------------------------
        st.subheader("🗺 Risk Level Map")
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

        if geom_type == "Point":
            # HeatMap for points
            gdf["lat"] = gdf.geometry.y
            gdf["lon"] = gdf.geometry.x
            heat_data = gdf[["lat", "lon", "risk_level"]].values.tolist()
            HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        else:
            # Choropleth for polygons or lines
            gdf["id"] = gdf.index
            folium.Choropleth(
                geo_data=gdf,
                data=gdf,
                columns=["id", "risk_level"],
                key_on="feature.properties.id",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Risk Level",
            ).add_to(m)

        st_folium(m, width=900, height=500)

        # ------------------------------------------
        # Statistics
        # ------------------------------------------
        st.subheader("📈 Risk Level Distribution")
        risk_counts = gdf["risk_level"].value_counts().sort_index()
        st.bar_chart(risk_counts)

else:
    st.info("👈 Enter the path to a Shapefile (.shp) using the sidebar.")
