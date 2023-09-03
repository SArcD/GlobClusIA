import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Título de la aplicación
st.title("Color-Magnitude Diagram of galactic globular clusters")

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
        st.error(f"There was an error obtaining the list of CSV files: {str(e)}")
        return []

# Obtener la lista de archivos CSV en el repositorio
csv_files = get_csv_files(repo_url)

# Muestra la lista de archivos CSV encontrados en un menú desplegable
if csv_files:
    selected_file_tuple = st.selectbox("Select a CSV file:", csv_files)
    selected_file = selected_file_tuple[1]  # Obtén la URL de descarga del archivo seleccionado
else:
    st.warning("No CSV files were found within the repository.")

# Función para cargar y mostrar el DataFrame seleccionado
def load_and_display_dataframe(selected_file):
    try:
        # selected_file debe ser una cadena que representa la URL de descarga
        csv_url = selected_file
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"There was an error loading the CSV file:  {str(e)}")
        return None

# Muestra el DataFrame seleccionado si se ha elegido un archivo
if "selected_file" in locals():
    df = load_and_display_dataframe(selected_file)
    if df is not None:
        st.write("DataFrame:")
        st.dataframe(df)

#

        import streamlit as st
        from tabulate import tabulate
        # Crea un expansor con un título
        with st.expander("Parámetros tomados de Gaia DR3"):
            # Tabla con los nombres de los parámetros y sus significados en inglés
            table_data = [
                ["Parameter", "Meaning"],
                ["**source_id**", "Identificador único de la fuente"],
                ["**ra**", "Ascensión recta en el sistema de referencia ICRS"],
                ["**ra_error**", "Error estándar de la ascensión recta"],
                ["**dec**", "Declinación en el sistema de referencia ICRS"],
                ["**dec_error**", "Error estándar de la declinación"],
                ["**parallax**", "Paralaje en el sistema de referencia ICRS"],
                ["**pmra**", "Movimiento propio en ascensión recta en el sistema de referencia ICRS"],
                ["**pmdec**", "Movimiento propio en declinación en el sistema de referencia ICRS"],
                ["**phot_g_mean_mag**", "Magnitud media integrada en la banda G"],
                ["**phot_bp_mean_mag**", "Magnitud media integrada en la banda BP"],
                ["**phot_rp_mean_mag**", "Magnitud media integrada en la banda RP"],
                ["**bp_rp**", "Índice de color BP-RP"],
                ["**bp_g**", "Índice de color BP-G"],
                ["**g_rp**", "Índice de color G-RP"],
                ["**radial_velocity**", "Velocidad radial combinada"],
                ["**grvs_mag**", "Magnitud media integrada en la banda RVS"],
                ["**grvs_error**", "Error estándar de la magnitud media integrada en la banda RVS"],
                ["**non_single_star**", "Indicador de estrella no simple (binaria, variable, etc.)"],
                ["**teff_gspphot**", "Temperatura efectiva estimada a partir del fotometría GSP-Phot"],
                ["**logg_gspphot**", "Gravedad superficial estimada a partir del fotometría GSP-Phot"],
                ["**mh_gspphot**", "Metalicidad estimada a partir del fotometría GSP-Phot"],
                ["**azero_gspphot**", "Extinción estimada a partir del fotometría GSP-Phot"],
                ["**ebpminrp_gspphot**", "Índice de color E(BP-RP) estimado a partir del fotometría GSP-Phot"]
            ]
    
            # Renderiza la tabla con formato Markdown
            st.markdown(tabulate(table_data, tablefmt="pipe", headers="firstrow"))



                #

        

        # Obtener las columnas numéricas del DataFrame
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_columns) < 2:
            st.warning("There must be two selected columns.")
        else:
            st.write("Select columns to plot:")

            # Menús desplegables para seleccionar columnas
            column1 = st.selectbox("Select the horizontal axis for the plot:", numeric_columns)
            column2 = st.selectbox("Select the vertical axis for the plot", numeric_columns)
            
            # Verifica si el parámetro seleccionado para el eje vertical es "phot_g_mean_mag" o "phot_rp_mean_mag"
            if column2 in ["phot_g_mean_mag", "phot_rp_mean_mag"]:
                # Botón para generar el gráfico con eje vertical invertido
                if st.button("Generar Gráfico"):
                    # Crear gráfico bidimensional en Plotly
                    fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
                    
                    # Invertir el eje vertical
                    fig.update_yaxes(autorange="reversed")
                    
                    st.plotly_chart(fig)
            else:
                # Botón para generar el gráfico sin inversión del eje vertical
                if st.button("Generar Gráfico"):
                    # Crear gráfico bidimensional en Plotly
                    fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
                    st.plotly_chart(fig)

resume=st.dataframe(df.describe())


