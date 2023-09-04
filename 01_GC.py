import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Título de la aplicación
st.title("Analysis of Color-Magnitude Diagrams of galactic globular clusters")

st.subheader("Individual analysis")
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
        #Supongamos que tienes un DataFrame llamado df
        df['source_id'] = df['source_id'].astype(str)
        st.write("DataFrame:")
        st.dataframe(df)
#

        import streamlit as st
        from tabulate import tabulate
        # Crea un expansor con un título
        with st.expander("Parameters taken from Gaia DR3"):
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
            #if column2 in [["phot_g_mean_mag", "phot_rp_mean_mag, phot_bp_mean_mag"]]:
                # Botón para generar el gráfico con eje vertical invertido
             #   if st.button("Generar Gráfico"):
                    # Crear gráfico bidimensional en Plotly
             #       fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
                    
                    # Invertir el eje vertical
             #       fig.update_yaxes(autorange="reversed")
                    
             #       st.plotly_chart(fig)

            if column2 in ["phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag"]:
                # Botón para generar el gráfico con eje vertical invertido
                if st.button("Generar Gráfico"):
                    # Crear gráfico bidimensional en Plotly
                    fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
        
                    # Verificar si se debe invertir el eje vertical
                    if column2 in ["phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag"]:
                        fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig)
            else:
                # Botón para generar el gráfico sin inversión del eje vertical
                if st.button("Generar Gráfico"):
                    # Crear gráfico bidimensional en Plotly
                    fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
                    st.plotly_chart(fig)

# Seleccionar las columnas deseadas del DataFrame original
columnas_seleccionadas = ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]
df_cmd = df[columnas_seleccionadas]        
#st.dataframe(df_cmd)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

# Cargar tu DataFrame df_cmd (asegúrate de tenerlo cargado previamente)
# df_cmd = ...

# Eliminar filas con valores NaN
df_cmd = df_cmd.dropna()

# Seleccionar solo las columnas numéricas
columnas_numericas = df_cmd.select_dtypes(include=[np.number])

# Normalizar los datos (opcional, pero recomendado para clustering)
scaler = StandardScaler()
columnas_numericas_scaled = scaler.fit_transform(columnas_numericas)

# Calcular la matriz de distancias
dist_matrix = pdist(columnas_numericas_scaled, metric='euclidean')

# Calcular la matriz de enlace utilizando el método de enlace completo (complete linkage)
Z = linkage(dist_matrix, method='complete')

# Realizar el clustering jerárquico y especificar el número de clusters deseado
num_clusters = 5  # Cambia esto al número de clusters deseado
cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')

# Agregar la columna de clusters "gc" al DataFrame original
df_cmd['gc'] = cluster_labels

# Crear un dendrograma
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z, labels=df_cmd.index, leaf_rotation=90)
plt.title("Dendrograma de Clustering Jerárquico")
plt.xlabel("Índice de la Muestra")
plt.ylabel("Distancia")
st.pyplot(fig)
st.dataframe(df_cmd)


# Puedes ajustar los parámetros del dendrograma para obtener una visualización más adecuada
# También puedes cortar el dendrograma para obtener grupos específicos


######################## Diagramas de Caja ########################

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.markdown(
    """
    La Figura muestra los **diagramas de caja** en los que se comparan cada uno de los **clusters** formados por la técnica de clustering jerárquico. Cada caja corresponde a un cluster en particular (donde a la izquierda de cada una pueden verse los puntos que corresponden a los pacientes contenidos). **Las cinturas de cada caja son una ayuda visual para determinar si hay evidencia suficiente sobre la diferencia entre los clusters** (si las cinturas coinciden en altura, entonces no hay evidencia de que los clusters puedan diferenciarse de acuerdo a sus valores en esa variable. **Si no coinciden en altura, entonces puede concluirse que los clusters pueden diferenciarse respecto a esa variable**."""
)

# Obtener los nombres de las columnas numéricas
columnas_numericas= df_cmd.select_dtypes(include='number').drop(columns=['gc']).columns

# Calcular el número de filas y columnas del panel
num_rows = len(columnas_numericas)
num_cols = 1  # Una columna para cada parámetro

# Ajustar el espacio vertical y la altura de los subplots
subplot_height = 400  # Ajusta la altura según tu preferencia    
vertical_spacing = 0.004  # Ajusta el espacio vertical según tu preferencia

# Crear subplots para cada parámetro
fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=columnas_numericas, vertical_spacing=vertical_spacing)

# Crear un gráfico de caja para cada parámetro y comparar los 10 clusters
for i, column in enumerate(columnas_numericas):
    # Obtener los datos de cada cluster para el parámetro actual
    cluster_data = [df_cmd[df_cmd['gc'] == cluster][column] for cluster in range(10)]

    # Agregar el gráfico de caja al subplot correspondiente
    for j in range(10):
        box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'group {j}')
        box.hovertemplate = 'id: %{text}'  # Agregar el valor de la columna 'Nombre' al hovertemplate
        box.text = df_cmd[df_cmd['gc'] == j]['source_id']  # Asignar los valores de la columna 'Nombre' al texto
        fig.add_trace(box, row=i+1, col=1)

# Actualizar el diseño y mostrar el panel de gráficos
fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
                  title_text='Comparación de Clusters - Gráfico de Caja',
                  margin=dict(t=100, b=100, l=50, r=50))  # Ajustar los márgenes del layout

# Mostrar la gráfica de caja en Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("""La Figura muestra los **diagramas de caja** en los que se comparan cada uno de los **clusters** formados por la técnica de clustering jerárquico. Cada caja corresponde a un cluster en particular (donde a la izquierda de cada una pueden verse los puntos que corresponden a los pacientes contenidos). **Las cinturas de cada caja son una ayuda visual para determinar si hay evidencia suficiente sobre la diferencia entre los clusters** (si las cinturas coinciden en altura, entonces no hay evidencia de que los clusters puedan diferenciarse de acuerdo a sus valores en esa variable. **Si no coinciden en altura, entonces puede concluirse que los clusters pueden diferenciarse respecto a esa variable**).""")

################### tsne ##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

st.markdown("""Los clusters representan datos que, de acuerdo con los valores que tienen en cada variable, pueden considerarse como mas **similares entre sí que con el resto**. Sin embargo, en muchos casos es dificil graficar los clusters, debido a que el número de variables que pueden estar involucradas puede ser muy alto. Las técnicas tsne y PCA pueden usarse en conjunto para crear una gráfica de todos los puntos en un plano. La Gráfica muestra los puntos agrupados en cada clusters, una vez que se han aplicado las técnicas de **análisis de componentes principales (PCA) y t-distributed neighbor embeding (tsne)**. Los contornos al rededor de cada cluster responden a la densidad de puntos (**donde las líneas están mas concentradas, significa una mayor densidad de puntos). Los pacientes con diagnóstico confirmado de sarcopenia, se muestran encerrados en círculos rojos.**""")


numeric_data=df_cmd.select_dtypes(include='number')
m = TSNE(learning_rate=100)
# Ajustar y transformar el modelo de t-SNE en el conjunto de datos numéricos
tsne_features = m.fit_transform(numeric_data.drop(['gc'],axis=1))


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Normalizar los datos
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numeric_data)
# Crear una instancia de PCA
pca = PCA()
# Aplicar PCA a los datos normalizados
pca_data = pca.fit_transform(normalized_data)
# Crear un nuevo DataFrame con las componentes principales
pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, pca.n_components_+1)])

df_cmd = df_cmd.reset_index(drop=True)
# Reset the indices of the DataFrames
pca_df.reset_index(drop=True, inplace=True)
df_cmd.reset_index(drop=True, inplace=True)
# Concatenate the "Nombre" column from data_2021_fem to pca_df
pca_df = pd.concat([pca_df, df_cmd["source_id"]], axis=1)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

data=df_cmd
labels = df_cmd['gc']
# Crear una instancia de t-SNE con los hiperparámetros deseados
tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=10, learning_rate=5)

# Ajustar t-SNE a los datos de PCA
tsne_data = tsne.fit_transform(pca_data)

# Crear una figura y un eje
fig, ax = plt.subplots()

# Colorear los puntos según las agrupaciones originales
for gc in np.unique(labels):
    indices = np.where(labels == gc)
    ax.scatter(tsne_data[indices, 0], tsne_data[indices, 1], label=f'Cluster {gc}')

    # Estimar la densidad de los puntos en el cluster actual
    kde = gaussian_kde(tsne_data[indices].T)
    x_range = np.linspace(np.min(tsne_data[:, 0]-1), np.max(tsne_data[:, 0]+1), 100)
    y_range = np.linspace(np.min(tsne_data[:, 1]-1), np.max(tsne_data[:, 1]+1), 100)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)

    # Agregar las curvas de densidad de kernel al gráfico
    ax.contour(xx, yy, zz, colors='k', alpha=0.5)

# Agregar leyenda y título al gráfico
ax.legend()
ax.set_title('Gráfico de Dispersión de t-SNE con Curvas de Densidad de Kernel')

# Mostrar el gráfico
plt.show()

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

# Crear una instancia de t-SNE con los hiperparámetros deseados

#filtered_df = df2021_pred[df2021_pred['SARCOPENIA'] == 1.0]


# Crear una figura
fig = go.Figure()
cluster_colors = ['blue', 'cyan', 'red', 'pink', 'green']
# Estimar la densidad de los puntos en cada cluster y agregar la superficie de contorno correspondiente
for cluster in np.unique(labels):
    indices = np.where(labels == gc)

    kde = gaussian_kde(tsne_data[indices].T)
    x_range = np.linspace(np.min(tsne_data[:, 0])-5, np.max(tsne_data[:, 0])+5, 100)
    y_range = np.linspace(np.min(tsne_data[:, 1])-5, np.max(tsne_data[:, 1])+5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)

    #if cluster in [0]:
    #    opacity = 0.9
    #    levels = 10
    #elif cluster ==1:
    #    opacity = 0.5
    #    levels = 7
    #else:
    opacity = 0.3
    levels = 5
    contour_trace = go.Contour(
        x=x_range,
        y=y_range,
        z=zz,
        colorscale='Blues',
        opacity=opacity,
        showscale=False,
        name=f'Contorno Cluster {gc}'
            )
    fig.add_trace(contour_trace)

# Colorear los puntos según las agrupaciones originales
for gc in np.unique(labels):
    indices = np.where(labels == gc)

    scatter_trace = go.Scatter(
        x=tsne_data[indices, 0].flatten(),
        y=tsne_data[indices, 1].flatten(),
        mode='markers',
        text=data.loc[labels == gc, ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
        hovertemplate="%{text}",
        marker=dict(
            size=7,
            line=dict(width=0.5, color='black')
            ),
        name=f'Cluster {gc}'
            )
    fig.add_trace(scatter_trace)

# Agregar círculos rojos alrededor de los puntos cuyos nombres coincidan con la lista filtered_df['Nombre']
#for name in filtered_df['folio']:
#    indices = data_2021[data_2021['folio'] == name].index
#    x_coords = tsne_data_2021[indices, 0].flatten()
#    y_coords = tsne_data_2021[indices, 1].flatten()

#    circle_trace = go.Scatter(
#        x=x_coords,
#        y=y_coords,
#        mode='markers',
#        marker=dict(
#            color='black',
#            size=10,
#            symbol='circle-open'
#        ),
#        hoverinfo='none',  # Para que no aparezcan tooltips adicionales para los círculos
#        showlegend=False  # Para que los círculos no aparezcan en la leyenda
#    )
#    fig.add_trace(circle_trace)



# Configurar el diseño del gráfico con el ancho de pantalla ajustado
fig.update_layout(
    title='Gráfico de Dispersión de t-SNE con Curvas de Densidad de Kernel',
    xaxis_title='Dimensión 1',
    yaxis_title='Dimensión 2',
    showlegend=True,
    legend_title='Clusters',
    width=1084  # Ajustar el ancho del gráfico
)
# Mostrar el gráfico
st.plotly_chart(fig, use_container_width=True)




