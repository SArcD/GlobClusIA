import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import plotly.express as px

# Título de la aplicación
st.title("Visualizador de Archivos CSV en GitHub")

# URL del repositorio de GitHub
repo_url = st.text_input("Introduce the repository's URL:", "")

# Función para obtener la lista de archivos CSV en el repositorio utilizando la GitHub API
def get_csv_files(repo_url):
    try:
        # Obtener el nombre de usuario y el nombre del repositorio desde la URL
        parts = repo_url.split("/")
        username = parts[-2]
        repository = parts[-1]
        
        # Hacer una solicitud a la GitHub API para obtener la lista de archivos
        api_url = f"https://api.github.com/repos/{username}/{repository}/contents"
        response = requests.get(api_url)
        data = response.json()
        
        # Filtrar los archivos CSV y obtener sus nombres y URL de descarga
        csv_files = [(item["name"], item["download_url"]) for item in data if item["name"].endswith(".csv")]
        
        return csv_files
    except Exception as e:
        st.error(f"There was an error obtainging the list of CSV files: {str(e)}")
        return []

# Obtener la lista de archivos CSV en el repositorio
csv_files = get_csv_files(repo_url)

# Muestra la lista de archivos CSV encontrados en un menú desplegable
if csv_files:
    selected_file_tuple = st.selectbox("Select a CSV file:", csv_files)
    selected_file = selected_file_tuple[1]  # Obtén la URL de descarga del archivo seleccionado
else:
    st.warning("No CSV files where found within the repository.")

# Función para cargar y mostrar el DataFrame seleccionado
def load_and_display_dataframe(selected_file):
    try:
        # selected_file debe ser una cadena que representa la URL de descarga
        csv_url = selected_file
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"There was an error loading the CSV file: {str(e)}")
        return None

# Muestra el DataFrame seleccionado si se ha elegido un archivo
if "selected_file" in locals():
    df = load_and_display_dataframe(selected_file)
    if df is not None:
        st.write("DataFrame Cargado:")
        st.dataframe(df)

        # Obtener las columnas numéricas del DataFrame
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_columns) < 2:
            st.warning("There must be two selected columns.")
        else:
            st.write("Select columns to plot:")
            # Menús desplegables para seleccionar columnas
            column1 = st.selectbox("Select the horizontal axis for the plot:", numeric_columns)
            column2 = st.selectbox("Select the vertical axis for the plot", numeric_columns)

            # Botón para generar el gráfico
            if st.button("Plot"):
                # Crear gráfico bidimensional en Plotly
                fig = px.scatter(df, x=column1, y=column2, title=f"Plot of {column1} vs. {column2}")
                st.plotly_chart(fig)
