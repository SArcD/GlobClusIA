import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Título de la aplicación
st.title("Visualizador de Archivos CSV en GitHub")

# URL del repositorio de GitHub
repo_url = st.text_input("Introduce la URL del repositorio de GitHub:", "")


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
        st.error(f"Error al obtener la lista de archivos CSV: {str(e)}")
        return []
#*********
#OK
#*********

