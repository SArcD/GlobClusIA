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
#def get_csv_files(repo_url):
#    try:
#        # Realiza una solicitud HTTP para obtener el contenido del repositorio
#        response = requests.get(repo_url)
#        soup = BeautifulSoup(response.text, "html.parser")
#        
#        # Busca los enlaces a archivos en el repositorio
#        links = soup.find_all("a", href=re.compile(r'\.csv$'))
#        
#        # Extrae los nombres de archivo de los enlaces
#        csv_files = [link["href"] for link in links]
#        
#        return csv_files
#    except Exception as e:
#        st.error(f"Error al obtener la lista de archivos CSV: {str(e)}")
#        return []
#import requests

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

# Obtener la lista de archivos CSV en el repositorio
csv_files = get_csv_files(repo_url)



# Obtiene la lista de archivos CSV en el repositorio
csv_files = get_csv_files(repo_url)

# Muestra la lista de archivos CSV encontrados en un menú desplegable
if csv_files:
    selected_file = st.selectbox("Selecciona un archivo CSV:", csv_files)
else:
    st.warning("No se encontraron archivos CSV en el repositorio.")

# Función para cargar y mostrar el DataFrame seleccionado
# Función para cargar y mostrar el DataFrame seleccionado
def load_and_display_dataframe(selected_file):
    try:
        # Construir la URL del archivo CSV
        csv_url = selected_file  # selected_file debe ser solo la URL de descarga
        df = pd.read_csv(csv_url)
        st.write("DataFrame Cargado:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {str(e)}")


# Muestra el DataFrame seleccionado si se ha elegido un archivo
if "selected_file" in locals():
    load_and_display_dataframe(selected_file)
