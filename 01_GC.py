import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Título de la aplicación
st.title("Visualizador de Archivos CSV en GitHub")

# URL del repositorio de GitHub
repo_url = st.text_input("Introduce la URL del repositorio de GitHub:", "")

# Función para obtener la lista de archivos CSV en el repositorio
def get_csv_files(repo_url):
    try:
        # Realiza una solicitud HTTP para obtener el contenido del repositorio
        response = requests.get(repo_url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Busca los enlaces a archivos en el repositorio
        links = soup.find_all("a", href=re.compile(r'\.csv$'))
        
        # Extrae los nombres de archivo de los enlaces
        csv_files = [link["href"] for link in links]
        
        return csv_files
    except Exception as e:
        st.error(f"Error al obtener la lista de archivos CSV: {str(e)}")
        return []

# Obtiene la lista de archivos CSV en el repositorio
csv_files = get_csv_files(repo_url)

# Muestra la lista de archivos CSV encontrados
if csv_files:
    selected_file = st.selectbox("Selecciona un archivo CSV:", csv_files)
else:
    st.warning("No se encontraron archivos CSV en el repositorio.")

# Función para cargar y mostrar el DataFrame seleccionado
def load_and_display_dataframe(selected_file):
    try:
        csv_url = f"{repo_url.rstrip('/')}/{selected_file}"
        df = pd.read_csv(csv_url)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {str(e)}")

# Muestra el DataFrame seleccionado
if "selected_file" in locals():
    load_and_display_dataframe(selected_file)
