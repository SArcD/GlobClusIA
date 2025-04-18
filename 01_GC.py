import streamlit as st
import pandas as pd
import requests
from tabulate import tabulate
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64

# Inyectar estilo CSS para aumentar tamaño del texto en la barra lateral
st.markdown("""
    <style>
    /* Aumentar tamaño de títulos y opciones en la barra lateral */
    [data-testid="stSidebar"] .css-ng1t4o {
        font-size: 20px !important;
    }
    [data-testid="stSidebar"] .css-1cpxqw2 {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)



# Cargar imagen (ajusta el nombre al archivo real)
logo = Image.open("logo_glocluster.png")

# Crear dos columnas
col1, col2 = st.columns([3, 6])  # proporción: imagen / texto

# Mostrar imagen en la columna izquierda
with col1:
    st.image(logo, width=600)  # puedes ajustar el tamaño

# Mostrar título en la columna derecha
with col2:
    st.title("Exploring Globular Clusters with Machine Learning")


# --- BARRA LATERAL ---
st.sidebar.title("📊 Analysis Steps")

etapas = [
    "1. Data Download",
    "2. Preprocessing",
    "3. Dimensionality Reduction",
    "4. Clustering",
    "5. Red Giant Identification",
    "6. RGB Tip Estimation",
    "7. Horizontal Branch Detection",
    "8. ΔM_G Estimation"
]


seleccion = st.sidebar.radio("Go to section:", etapas)


# URL de la imagen en GitHub
#image_url = "https://github.com/SArcD/GlobClusIA/raw/main/descargar%20-%202024-05-24T172020.397.png"
#image_url = "cluster_imag.PNG"
# Descargar la imagen
#response = requests.get(image_url)
#image = Image.open(BytesIO(response.content))
# Cargar imagen local directamente
#image = Image.open("cluster_imag.PNG")
#st.image(image, caption="Color-Magnitude Diagram", use_column_width=True)
# Cargar imagen local
from PIL import Image
import streamlit as st

# Cargar imagen
image = Image.open("cluster_imag.PNG")

# Reducir tamaño a la mitad
new_width = image.width // 2
new_height = image.height // 2
image_resized = image.resize((new_width, new_height))

# Mostrar imagen reducida
st.image(image_resized, caption="Graphical representation of the appearance of a globular cluster (made from a point distribution that follows the King mass distribution).", use_container_width=True)



#st.image(image, caption="Color-Magnitude Diagram", use_column_width=True)



# Mostrar la imagen
#st.image(image, caption="Graphical representation of the appearance of a globular cluster (made from a point distribution that follows the King mass distribution).", use_container_width=True)

st.subheader("Overview of the application")
st.markdown("""
<div style="text-align: justify;">
This application allows the analysis of the color-magnitude diagrams of globular clusters in the Milky Way using machine learning. By applying the hierarchical clustering algorithm, the stars in the database are grouped into sets according to their similarity in their photometric data, surface temperature, and metallicity. In many cases, these sets correspond to different stages of stellar evolution. Using a decision tree algorithm, the rules and cut-off points are obtained in the variables of interest that define each set of stars.
</div>
""", unsafe_allow_html=True)


st.subheader("Color-Magnitude Diagram")
st.markdown("""
<div style="text-align: justify">
**Instructions:** Please select the files with the **photometry** (Cluster-name_photo.csv) and **observable parameters** (Cluster-name_metal.csv) for any of the globular clusters displayed below to analyze. The third GAIA data release (DR3) obtained data for each globular cluster (https://gea.esac.esa.int/archive/).

</div>
""", unsafe_allow_html=True)

# Lista de archivos CSV en el repositorio
csv_files = [
    "M13_metal.csv", "M13_photo.csv", "M92_metal.csv", "M92_photo.csv",
    "OmegaCen_metal.csv", "OmegaCen_photo.csv", "Tuc47_metal.csv", "Tuc47_photo.csv"
]

# URL base del repositorio de GitHub
base_url = "https://raw.githubusercontent.com/SArcD/GlobClusIA/main/"

# Muestra la lista de archivos CSV encontrados en un menú desplegable
selected_files_tuple = st.multiselect("Select CSV files to merge:", csv_files)

# Función para cargar y fusionar los DataFrames seleccionados por "source_id"
def load_and_merge_dataframes(selected_files):
    try:
        if len(selected_files) < 2:
            st.warning("Please select at least two CSV files for merging.")
            return None

        # Cargar y fusionar los DataFrames seleccionados
        dfs = []
        for selected_file in selected_files:
            csv_url = f"{base_url}{selected_file}"
            df = pd.read_csv(csv_url)
            dfs.append(df)

        merged_df = dfs[0]  # Tomar el primer DataFrame como base

        # Fusionar los DataFrames restantes en función de la columna "source_id"
        for df_to_merge in dfs[1:]:
            merged_df = pd.merge(merged_df, df_to_merge, on="source_id", how="outer")

        return merged_df
    except Exception as e:
        st.error(f"There was an error loading and merging the selected CSV files: {str(e)}")
        return None

# Muestra el DataFrame fusionado si se han seleccionado al menos dos archivos
if len(selected_files_tuple) >= 2:
    merged_df = load_and_merge_dataframes(selected_files_tuple)
    if merged_df is not None:
        st.write("**Merged DataFrame:**")
        df = merged_df
        df['source_id'] = df['source_id'].astype(str)
        st.dataframe(df)

        # Obtener información sobre el DataFrame
        num_rows, num_columns = df.shape
        num_missing = df.isnull().sum().sum()
        with st.expander("**Additional information**"):
        # Mostrar información adicional
            st.write(f"**Number of rows:** {num_rows}")
            st.write(f"**Number of columns:** {num_columns}")

        
            # Mostrar el número de filas con datos faltantes
            filas_con_faltantes = (df.isna().any(axis=1)).sum()
            st.write(f"**Number of rows with missing data:** {filas_con_faltantes}")

        # Crear un botón de descarga para el dataframe
        def download_button(df, filename, button_text):
            # Crear un objeto ExcelWriter
            excel_writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            # Guardar el dataframe en el objeto ExcelWriter
            df.to_excel(excel_writer, index=False)
            # Cerrar el objeto ExcelWriter
            excel_writer.save()
            # Leer el archivo guardado como bytes
            with open(filename, 'rb') as f:
                file_bytes = f.read()
                # Generar el enlace de descarga
                href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_bytes).decode()}" download="{filename}">{button_text}</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Crear un botón de descarga para el dataframe
        def download_button_CSV(df, filename, button_text):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
            st.markdown(href, unsafe_allow_html=True)


        # Dividir la página en dos columnas
        col1, col2 = st.columns(2)
        # Agregar un botón de descarga para el dataframe en la primera columna
        with col1:
            download_button(df, 'Cluster_data_(GDR3).xlsx', 'Download .xlsx file')
            st.write('')
        # Agregar un botón de descarga para el dataframe en la segunda columna
        with col2:
            download_button_CSV(df, 'Cluster_data_(GDR3).csv', 'Download .csv file')
            st.write('')
    


        
        # Expansor con parámetros tomados de Gaia DR3
        with st.expander("Parameters taken from Gaia DR3"):
            table_data = [
                ["Parameter", "Meaning"], 
                ["**source_id**", "Unique identifier of the source"], 
                ["**phot_g_mean_mag**", "Mean integrated magnitude in the G band"],
                ["**phot_bp_mean_mag**", "Mean integrated magnitude in the BP band"],
                ["**phot_rp_mean_mag**", "Mean integrated magnitude in the RP band"],
                ["**bp_rp**", "BP-RP color index"],
                ["**bp_g**", "BP-G color index"],
                ["**g_rp**", "G-RP color index"],
                ["**teff_gspphot**", "Effective temperature estimated from GSP-Phot photometry"],
                ["**logg_gspphot**", "Surface gravity estimated from GSP-Phot photometry"],
                ["**mh_gspphot**", "Metallicity estimated from GSP-Phot photometry"],
            ]
            st.markdown(tabulate(table_data, tablefmt="pipe", headers="firstrow"))

        st.subheader("Two dimensional plots of cluster parameters")
        st.markdown("""
        <div style="text-align: justify">
        In this section, you can visualize the color-magnitude diagrams of the selected globular cluster. Please select the variables to represent the horizontal and vertical axes of the bar. The variables "bp_rp", "bp_g" and "g_rp" correspond to the colors, while "phot_g_mean_mag", "phot_bp_mean_mag" and "phot_rp_mean_mag" correspond to the magnitudes integrated in the G, BP and RP bands. In addition to color-magnitude diagrams, you can create graphs from other variables, such as estimated effective temperature, metallicity, or surface gravity.

        **Instructions:** Select at least two variables to generate a two-dimensional plot. Some of the plot's settings can be manipulated on the menu in its upper right corner. The resulting plot can be saved by clicking on the icon with the shape of a camera.
        </div>  
        """, unsafe_allow_html=True)



        
        # Obtener las columnas numéricas del DataFrame
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_columns) < 2:
            st.warning("There must be two selected columns.")
        else:
            st.write("Plot:")
            column1 = st.selectbox("Select the horizontal axis for the plot:", numeric_columns)
            column2 = st.selectbox("Select the vertical axis for the plot:", numeric_columns)
            if column2 in ["phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag"]:
                if st.button("Make Plot"):
                    fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
                    if column2 in ["phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag"]:
                        fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig)
            else:
                if st.button("Generar Gráfico"):
                    fig = px.scatter(df, x=column1, y=column2, title=f"Plot {column1} vs. {column2}")
                    st.plotly_chart(fig)

# Seleccionar las columnas deseadas del DataFrame original
columnas_seleccionadas = ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]
df_cmd = df[columnas_seleccionadas]


################################################

st.subheader("Photometry")

st.markdown("""
<div style="text-align: justify">
In this section, you can visualize the color-magnitude diagrams of the selected globular cluster. Please select the variables to represent the horizontal and vertical axes of the bar. The variables "bp_rp", "bp_g" and "g_rp" correspond to the colors, while "phot_g_mean_mag", "phot_bp_mean_mag" and "phot_rp_mean_mag" correspond to the magnitudes integrated in the G, BP and RP bands. In addition to color-magnitude diagrams, you can create graphs from other variables, such as estimated effective temperature, metallicity, or surface gravity.

**Instructions:** Select at least two variables to generate a two-dimensional plot. Some of the plot's settings can be manipulated on the menu in its upper right corner. The resulting plot can be saved by clicking on the icon with the shape of a camera.
</div>  
""", unsafe_allow_html=True)

import streamlit as st
import numpy as np
import plotly.express as px

df_cmd = df_cmd.dropna()

# Create the histogram with a bin size of 0.15 magnitudes
bin_size = 0.15
magnitudes = df_cmd["phot_bp_mean_mag"]  # Reemplaza con el nombre real de la columna

# Calculate the histogram
hist, bins = np.histogram(magnitudes, bins=int((max(magnitudes) - min(magnitudes)) / bin_size))

# Apply logarithm to the values on the vertical axis (y)
hist_log = np.log(hist)

# Create a Plotly figure
fig = px.bar(x=bins[:-1], y=hist_log, labels={'x': 'Apparent Magnitude (Bp-Band)', 'y': 'Log(Number of stars)'})
fig.update_xaxes(type='log')  # Escala logarítmica en el eje horizontal

# Set plot title
fig.update_layout(title='Differential Histogram of Apparent Magnitude (Log Scale)')

# Show the plot in Streamlit
st.plotly_chart(fig)


#################################333
import streamlit as st
import numpy as np
import plotly.express as px

#df_cmd = df_cmd.dropna()
# Create the histogram with a bin size of 0.15 magnitudes
bin_size = 0.15
magnitudes = df_cmd["phot_bp_mean_mag"]  # Reemplaza con el nombre real de la columna
# Calculate the number of clusters in each bin
hist, bins = np.histogram(magnitudes, bins=int((max(magnitudes) - min(magnitudes)) / bin_size))
# Cumulative sum of the histogram
cumulative_hist = np.cumsum(hist)
# Aplicar logaritmo a los valores en el eje vertical (y)
cumulative_hist_log = np.log(cumulative_hist)
# Create a Plotly figure
fig = px.line(x=bins[:-1], y=cumulative_hist_log, labels={'x': 'Apparent Magnitude (Bp-Band)', 'y': 'Log(Number of stars)'})
fig.update_xaxes(type='log')  # Escala logarítmica en el eje horizontal
# Set plot title
fig.update_layout(title='Cumulative Histogram of Apparent Magnitude (Log Scale)')
# Show the plot in Streamlit
st.plotly_chart(fig)


################################################
################################################

import streamlit as st
import numpy as np
import plotly.express as px

df_cmd = df_cmd.dropna()

# Create the histogram with a bin size of 0.15 magnitudes
bin_size = 0.15
magnitudes = df_cmd["phot_g_mean_mag"]  # Reemplaza con el nombre real de la columna

# Calculate the histogram
hist, bins = np.histogram(magnitudes, bins=int((max(magnitudes) - min(magnitudes)) / bin_size))

# Apply logarithm to the values on the vertical axis (y)
hist_log = np.log(hist)

# Create a Plotly figure
fig = px.bar(x=bins[:-1], y=hist_log, labels={'x': 'Apparent Magnitude (G-Band)', 'y': 'Log(Number of stars)'})
fig.update_xaxes(type='log')  # Escala logarítmica en el eje horizontal

# Set plot title
fig.update_layout(title='Differential Histogram of Apparent Magnitude (Log Scale)')

# Show the plot in Streamlit
st.plotly_chart(fig)


#################################333
import streamlit as st
import numpy as np
import plotly.express as px

#df_cmd = df_cmd.dropna()
# Create the histogram with a bin size of 0.15 magnitudes
bin_size = 0.15
magnitudes = df_cmd["phot_g_mean_mag"]  # Reemplaza con el nombre real de la columna
# Calculate the number of clusters in each bin
hist, bins = np.histogram(magnitudes, bins=int((max(magnitudes) - min(magnitudes)) / bin_size))
# Cumulative sum of the histogram
cumulative_hist = np.cumsum(hist)
# Aplicar logaritmo a los valores en el eje vertical (y)
cumulative_hist_log = np.log(cumulative_hist)
# Create a Plotly figure
fig = px.line(x=bins[:-1], y=cumulative_hist_log, labels={'x': 'Apparent Magnitude (G-Band)', 'y': 'Log(Number of stars)'})
fig.update_xaxes(type='log')  # Escala logarítmica en el eje horizontal
# Set plot title
fig.update_layout(title='Cumulative Histogram of Apparent Magnitude (Log Scale)')
# Show the plot in Streamlit
st.plotly_chart(fig)




################################################

import streamlit as st
import numpy as np
import plotly.express as px

df_cmd = df_cmd.dropna()

# Create the histogram with a bin size of 0.15 magnitudes
bin_size = 0.15
magnitudes = df_cmd["phot_rp_mean_mag"]  # Reemplaza con el nombre real de la columna

# Calculate the histogram
hist, bins = np.histogram(magnitudes, bins=int((max(magnitudes) - min(magnitudes)) / bin_size))

# Apply logarithm to the values on the vertical axis (y)
hist_log = np.log(hist)

# Create a Plotly figure
fig = px.bar(x=bins[:-1], y=hist_log, labels={'x': 'Apparent Magnitude (Rp-Band)', 'y': 'Log(Number of stars)'})
fig.update_xaxes(type='log')  # Escala logarítmica en el eje horizontal

# Set plot title
fig.update_layout(title='Differential Histogram of Apparent Magnitude (Log Scale)')

# Show the plot in Streamlit
st.plotly_chart(fig)


#################################333
import streamlit as st
import numpy as np
import plotly.express as px

#df_cmd = df_cmd.dropna()
# Create the histogram with a bin size of 0.15 magnitudes
bin_size = 0.15
magnitudes = df_cmd["phot_rp_mean_mag"]  # Reemplaza con el nombre real de la columna
# Calculate the number of clusters in each bin
hist, bins = np.histogram(magnitudes, bins=int((max(magnitudes) - min(magnitudes)) / bin_size))
# Cumulative sum of the histogram
cumulative_hist = np.cumsum(hist)
# Aplicar logaritmo a los valores en el eje vertical (y)
cumulative_hist_log = np.log(cumulative_hist)
# Create a Plotly figure
fig = px.line(x=bins[:-1], y=cumulative_hist_log, labels={'x': 'Apparent Magnitude (Rp-Band)', 'y': 'Log(Number of stars)'})
fig.update_xaxes(type='log')  # Escala logarítmica en el eje horizontal
# Set plot title
fig.update_layout(title='Cumulative Histogram of Apparent Magnitude (Log Scale)')
# Show the plot in Streamlit
st.plotly_chart(fig)


################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

# Eliminar filas con valores NaN
df_cmd = df_cmd.dropna()

columnas_numericas = df_cmd.select_dtypes(include=['number'])
# Normalizar los datos (opcional, pero recomendado para clustering)
scaler = StandardScaler()
columnas_numericas_scaled = scaler.fit_transform(columnas_numericas)

# Calcular la matriz de distancias
dist_matrix = pdist(columnas_numericas_scaled, metric='euclidean')

# Agregar una interfaz de usuario para ingresar el número deseado de clusters

st.title("Hierarchical Clustering")
#st.write("Enter the desired number of clusters:")
st.markdown("""
<div style="text-align:justify">

The data will be analyzed under the hierarchical clustering algorithm (groups of objects that show high similarity in their astrophysical parameters). The predefined number of clusters is **five**.
</div>

""", unsafe_allow_html=True)

# Agregar un campo de entrada para el número de clusters
#num_clusters = st.number_input("Number of clusters", min_value=1, value=4)
num_clusters=5
# Verificar si el usuario ha ingresado un número de clusters válido
#if st.button("Make Clustering"):
# Calcular la matriz de enlace utilizando el método de enlace completo (complete linkage)
Z = linkage(dist_matrix, method='ward')

# Realizar el clustering jerárquico aglomerativo
clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
cluster_labels = clustering.fit_predict(columnas_numericas_scaled)

# Agregar la columna de clusters "gc" al DataFrame original
df_cmd['gc'] = cluster_labels

# Crear un dendrograma
#fig, ax = plt.subplots(figsize=(12, 6))
#dendrogram(Z, labels=df_cmd.index, leaf_rotation=90)
#plt.title("Dendrograma de Clustering Jerárquico")
#plt.xlabel("Índice de la Muestra")
#plt.ylabel("Distancia")
#st.pyplot(fig)
st.write(f"Número de clusters seleccionado: {num_clusters}")
st.dataframe(df_cmd)
# Mostrar información adicional
num_rows, num_columns = df_cmd.shape
num_missing = df_cmd.isnull().sum().sum()    
st.write(f"**Number of rows:** {num_rows}")
st.write(f"**Number of columns:** {num_columns}")        
# Mostrar el número de filas con datos faltantes
filas_con_faltantes = (df_cmd.isna().any(axis=1)).sum()
# Mostrar las filas con datos faltantes
st.write(f"**Number of rows with missing data:** {filas_con_faltantes}")


######################## Diagramas de Caja ########################

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.markdown("""
       <div style="text-align: justify">
                   
 The figure displays box plots that compare each of the sub-clusters formed by the hierarchical clustering technique. Each box corresponds to a group of stars (the points shown to the left of each box correspond to a specific star). The waistlines of each box are a visual aid to determine if there is enough evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the sub-clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the sub-clusters can differentiate concerning that variable.
       </div>
       """, unsafe_allow_html=True)



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

    #Agregar el gráfico de caja al subplot correspondiente
    for j in range(10):
        box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'group {j}')
        box.hovertemplate = 'id: %{text}'  # Agregar el valor de la columna 'Nombre' al hovertemplate
        box.text = df_cmd[df_cmd['gc'] == j]['source_id']  # Asignar los valores de la columna 'Nombre' al texto
        fig.add_trace(box, row=i+1, col=1)

#Actualizar el diseño y mostrar el panel de gráficos
fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
                    title_text='Comparación de Clusters - Gráfico de Caja',
                    margin=dict(t=100, b=100, l=50, r=50))  # Ajustar los márgenes del layout

#Mostrar la gráfica de caja en Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align: justify">

The Figure displays the box plots in which each of the clusters formed by the hierarchical clustering technique is compared. Each box corresponds to a particular cluster (where on the left side of each one, you can see the points corresponding to the contained patients). The waistlines of each box serve as a visual aid to determine if there is sufficient evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the clusters can differentiate with respect to that variable). 
</div>
        """,unsafe_allow_html=True)

    ################### tsne ##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

st.markdown("""
<div style="text-align: justify">

The sub-clusters represent data that, based on their values in each variable, can be considered as more similar to each other than to the rest. However, in many cases, visualizing the sub-clusters can be challenging because the number of variables involved can be very high. The techniques t-SNE and PCA can be used together to create a plot of all the points on a plane. The plot shows the points grouped within each cluster, once the techniques of Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) have been applied. The contours around each cluster correspond to the density of points (where the lines are more concentrated, it signifies a higher point density).
        </div>
        """,unsafe_allow_html=True)


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
pca_df = pd.concat([pca_df, df_cmd["source_id"]], axis=1)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

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
    ax.scatter(tsne_data[indices, 0], tsne_data[indices, 1], label=f'Group {gc}')

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

# Crear una figura
fig = go.Figure()
cluster_colors = ['blue', 'cyan', 'red', 'pink', 'green']
# Estimar la densidad de los puntos en cada cluster y agregar la superficie de contorno correspondiente
for gc in np.unique(labels):
    indices = np.where(labels == gc)

    kde = gaussian_kde(tsne_data[indices].T)
    x_range = np.linspace(np.min(tsne_data[:, 0])-5, np.max(tsne_data[:, 0])+5, 100)
    y_range = np.linspace(np.min(tsne_data[:, 1])-5, np.max(tsne_data[:, 1])+5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)

    if gc in [0]:
        opacity = 0.9
        levels = 10
    elif gc ==1:
        opacity = 0.5
        levels = 7
    else:
        opacity = 0.3
        levels = 5
    
    contour_trace = go.Contour(
        x=x_range,
        y=y_range,
        z=zz,
        colorscale='Blues',
        opacity=opacity,
        showscale=False,
        name=f'Contour {gc}'
    )
    fig.add_trace(contour_trace)

# Colorear los puntos según las agrupaciones originales
for gc in np.unique(labels):
    indices = np.where(labels == gc)

    scatter_trace = go.Scatter(
        x=tsne_data[indices, 0].flatten(),
        y=tsne_data[indices, 1].flatten(),
        mode='markers',
        text=df_cmd.loc[labels == gc, ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
        hovertemplate="%{text}",
        marker=dict(
            size=7,
            line=dict(width=0.5, color='black')
        ),
        name=f'Cluster {gc}'
    )
    fig.add_trace(scatter_trace)

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


#Actualizar el diseño y mostrar el panel de gráficos
#fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
#                    title_text='Comparación de Clusters - Gráfico de Caja',
#                    margin=dict(t=100, b=100, l=50, r=50))  # Ajustar los márgenes del layout

#Mostrar la gráfica de caja en Streamlit
#st.plotly_chart(fig, use_container_width=True)



    ###

import pandas as pd
import plotly.express as px

# Gráfica 1: phot_g_mean_mag vs g_rp
fig1 = px.scatter(df_cmd, x="g_rp", y="phot_g_mean_mag", color="gc",
                    hover_data=df_cmd.columns, title="phot_g_mean_mag vs g_rp")
fig1.update_yaxes(autorange="reversed")

# Gráfica 2: phot_bp_mean_mag vs bp_rp
fig2 = px.scatter(df_cmd, x="bp_rp", y="phot_bp_mean_mag", color="gc",
                    hover_data=df_cmd.columns, title="phot_bp_mean_mag vs bp_rp")
fig2.update_yaxes(autorange="reversed")

# Gráfica 3: phot_rp_mean_mag vs bp_rp
fig3 = px.scatter(df_cmd, x="bp_rp", y="phot_rp_mean_mag", color="gc",
                    hover_data=df_cmd.columns, title="phot_rp_mean_mag vs bp_rp")
fig3.update_yaxes(autorange="reversed")

# Mostrar las gráficas en Streamlit
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

##############################

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text, export_graphviz
import pydotplus
from IPython.display import Image, display
from sklearn.tree import plot_tree

st.write("**Gráfica del árbol de decisión**")

        
# Convertir las etiquetas de cluster a valores numéricos
label_encoder = LabelEncoder()
df_cmd.loc[:, 'gc'] = label_encoder.fit_transform(df_cmd['gc'])

# Definir las variables de entrada y salida para el Random Forest
numeric_data=df_cmd.select_dtypes(include='number')
X = numeric_data.drop('gc', axis=1)
y = df_cmd['gc']

# Crear y entrenar el Random Forest
random_forest = RandomForestClassifier(random_state=1, min_samples_split=80, ccp_alpha=0.001)
random_forest.fit(X, y)

# Obtener los nombres de las columnas originales y convertirlos en cadenas de texto
column_names = [str(column) for column in X.columns.tolist()]

# Obtener el mejor árbol del Random Forest
best_tree_index = random_forest.feature_importances_.argmax()
best_tree = random_forest.estimators_[best_tree_index]

# Visualizar las reglas del mejor árbol
tree_rules = export_text(best_tree, feature_names=column_names)

# Generar y mostrar la gráfica del árbol
plt.figure(figsize=(40, 30), dpi=400) 
#plt.rcParams.update({'font.size': 20}) 
#plot_tree(best_tree, feature_names=column_names, class_names=[str(cls) for cls in label_encoder.classes_], filled=True, rounded=True)
plot_tree(best_tree, feature_names=column_names, class_names=[str(cls) for cls in label_encoder.classes_], filled=True, rounded=True, fontsize=16)
plt.savefig('tree_plot.png', dpi=400, bbox_inches='tight', format='png')
#plt.savefig('tree_plot.png')  # Guardar la gráfica como imagen
st.write("**Decision tree chart**")

with st.expander("Click to Expand"):
    st.markdown("""
    <div style=text-align:justify>
    
    **In the following figure, a decision tree diagram is shown, which is used to classify stars based on their key astrophysical properties**. This decision tree has been constructed using a dataset of stars and has been trained to identify different types of stars based on their properties.

    **The classification process starts from the top of the tree**, and it moves downward as it evaluates different conditions in the star's characteristics. Each node of the tree represents a question about a particular astrophysical feature, and the branches stemming from each node represent the two possible answers to that question: yes or no.

    - If a star meets the condition specified at a node (answer "yes"), it follows the left branch of the tree.
    - If the star does not meet the condition (answer "no"), it follows the right branch of the tree.

    This decision-making process continues until we reach a terminal node, also known as a leaf of the tree. At the tree's leaves, a label or classification is assigned to the star based on the conditions it has met throughout its journey through the tree. This classification could be, for example, the spectral type of the star (as in the case of A-type, B-type, O-type, etc.) or some other relevant astrophysical category.

    Within each tree node, additional information is provided, such as the Gini index, which measures the impurity of the classification at that point, and the number of stars in the training dataset that met or did not meet the condition at that node.

    **The ultimate goal of the decision tree is to classify each star into one of the predefined categories based on its astrophysical properties**. The tree is constructed in a way that maximizes the purity of the classifications at the final leaves and, thus, becomes a powerful tool for understanding how astrophysical properties influence the classification of stars into different types.
    </div>
    
    """, unsafe_allow_html=True)

st.image('tree_plot.png')



##############################

import pandas as pd
import streamlit as st

# Supongamos que tienes K clusters en tu DataFrame original
K = df_cmd['gc'].nunique()

# Crear un diccionario de DataFrames donde cada clave es el número de cluster
dataframes_por_cluster = {}

# Iterar sobre cada cluster y crear un DataFrame para cada uno
for cluster_num in range(K):
    # Filtrar las filas que pertenecen al cluster actual
    cluster_df = df_cmd[df_cmd['gc'] == cluster_num]
    
    # Almacenar el DataFrame en el diccionario con la clave como el número de cluster
    dataframes_por_cluster[cluster_num] = cluster_df

# Mostrar todos los DataFrames uno por uno
#for cluster_num, cluster_df in dataframes_por_cluster.items():
#    st.write(f"Cluster {cluster_num}:")
#    st.write(cluster_df)
#    # Crear un botón de descarga para el DataFrame actual
#    csv_data = cluster_df.to_csv(index=False)
#    button_label = f"Download Group {cluster_num} (CSV)"
#    st.download_button(label=button_label, data=csv_data, key=f"download_button_{cluster_num}")

# Iterar sobre los clusters
for cluster_num, cluster_data in dataframes_por_cluster.items():
    with st.expander(f"Details for Cluster {cluster_num}"):
        # Eliminar filas con valores NaN
        columnas_numericas = cluster_data.select_dtypes(include=['number'])
        scaler = StandardScaler()
        columnas_numericas_scaled = scaler.fit_transform(columnas_numericas)

        # Calcular la matriz de distancias
        dist_matrix = pdist(columnas_numericas_scaled, metric='euclidean')

        # Agregar una interfaz de usuario para ingresar el número deseado de clusters
        st.title("Hierarchical Clustering")
        #num_clusters = st.slider("Enter the desired number of clusters:", min_value=2, max_value=10, value=3)
        num_clusters = 5
        # Calcular la matriz de enlace utilizando el método de enlace completo (complete linkage)
        Z = linkage(dist_matrix, method='ward')

        # Realizar el clustering jerárquico aglomerativo
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(columnas_numericas_scaled)

        # Agregar la columna de clusters "gc" al DataFrame original
        cluster_data['gc'] = cluster_labels

        # Crear un dendrograma
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, labels=cluster_data.index, leaf_rotation=90)
        plt.title("Dendrograma de Clustering Jerárquico")
        plt.xlabel("Índice de la Muestra")
        plt.ylabel("Distancia")
        st.pyplot(fig)

        st.write(f"Número de clusters seleccionado: {num_clusters}")
        st.dataframe(cluster_data)

        # Obtener información sobre el DataFrame
        num_rows, num_columns = cluster_data.shape
        num_missing = cluster_data.isnull().sum().sum()
        st.write(f"**Number of rows:** {num_rows}")
        st.write(f"**Number of columns:** {num_columns}")

        filas_con_faltantes = (cluster_data.isna().any(axis=1)).sum()
        st.write(f"**Number of rows with missing data:** {filas_con_faltantes}")

        # Resto de tu código aquí

        ######################## Diagramas de Caja ########################

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import streamlit as st

        st.markdown(
        """
        <div style:text-align:justify>
        
        The figure displays box plots that compare each of the sub-clusters formed by the hierarchical clustering technique. Each box corresponds to a group of stars (the points shown to the left of each box correspond to a specific star). The waistlines of each box are a visual aid to determine if there is enough evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the sub-clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the sub-clusters can differentiate concerning that variable."
         </div>
        
        """, unsafe_allow_html=True
        
       )

        # Obtener los nombres de las columnas numéricas
        columnas_numericas= cluster_data.select_dtypes(include='number').drop(columns=['gc']).columns

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
            clusters_data = [cluster_data[cluster_data['gc'] == cluster][column] for cluster in range(10)]

            #Agregar el gráfico de caja al subplot correspondiente
            for j in range(10):
                box = go.Box(y=clusters_data[j], boxpoints='all', notched=True, name=f'group {j}')
                box.hovertemplate = 'id: %{text}'  # Agregar el valor de la columna 'Nombre' al hovertemplate
                box.text = cluster_data[cluster_data['gc'] == j]['source_id']  # Asignar los valores de la columna 'Nombre' al texto
                fig.add_trace(box, row=i+1, col=1)

        #Actualizar el diseño y mostrar el panel de gráficos
        fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
                        title_text='Comparación de Clusters - Gráfico de Caja',
                        margin=dict(t=100, b=100, l=50, r=50))  # Ajustar los márgenes del layout

        #Mostrar la gráfica de caja en Streamlit
        st.plotly_chart(fig, use_container_width=True)

###################


    ################### tsne ##############################

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_digits
        from sklearn.manifold import TSNE

        st.markdown("""
        <div style="text-align:justify">
        
        The sub-clusters represent data that, based on their values in each variable, can be considered as more similar to each other than to the rest. However, in many cases, visualizing the sub-clusters can be challenging because the number of variables involved can be very high. The techniques t-SNE and PCA can be used together to create a plot of all the points on a plane. The plot shows the points grouped within each cluster, once the techniques of Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) have been applied. The contours around each cluster correspond to the density of points (where the lines are more concentrated, it signifies a higher point density).
        </div>
        """,unsafe_allow_html=True)

        numeric_data=cluster_data.select_dtypes(include='number')
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

        cluster_data = cluster_data.reset_index(drop=True)
        # Reset the indices of the DataFrames
        pca_df.reset_index(drop=True, inplace=True)
        df_cmd.reset_index(drop=True, inplace=True)
        pca_df = pd.concat([pca_df, cluster_data["source_id"]], axis=1)

        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE
        from scipy.stats import gaussian_kde

        labels = cluster_data['gc']
        # Crear una instancia de t-SNE con los hiperparámetros deseados
        tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=10, learning_rate=5)

        # Ajustar t-SNE a los datos de PCA
        tsne_data = tsne.fit_transform(pca_data)

        # Crear una figura y un eje
        fig, ax = plt.subplots()

        # Colorear los puntos según las agrupaciones originales
        for gc in np.unique(labels):
            indices = np.where(labels == gc)
            ax.scatter(tsne_data[indices, 0], tsne_data[indices, 1], label=f'Group {gc}')

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
        #plt.show()
        st.pyplot(fig)

        import numpy as np
        import plotly.graph_objects as go
        from sklearn.manifold import TSNE
        from scipy.stats import gaussian_kde

        # Crear una figura
        fig = go.Figure()
        cluster_colors = ['blue', 'cyan', 'red', 'pink', 'green']
        # Estimar la densidad de los puntos en cada cluster y agregar la superficie de contorno correspondiente
        for gc in np.unique(labels):
            indices = np.where(labels == gc)

            kde = gaussian_kde(tsne_data[indices].T)
            x_range = np.linspace(np.min(tsne_data[:, 0])-5, np.max(tsne_data[:, 0])+5, 100)
            y_range = np.linspace(np.min(tsne_data[:, 1])-5, np.max(tsne_data[:, 1])+5, 100)
            xx, yy = np.meshgrid(x_range, y_range)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = np.reshape(kde(positions).T, xx.shape)

            if gc in [0]:
                opacity = 0.9
                levels = 10
            elif gc ==1:
                opacity = 0.5
                levels = 7
            else:
                opacity = 0.3
                levels = 5
    
            contour_trace = go.Contour(
                x=x_range,
                y=y_range,
                z=zz,
                colorscale='Blues',
                opacity=opacity,
                showscale=False,
                name=f'Contour {gc}'
            )
            fig.add_trace(contour_trace)

        # Colorear los puntos según las agrupaciones originales
        for gc in np.unique(labels):
            indices = np.where(labels == gc)

            scatter_trace = go.Scatter(
                x=tsne_data[indices, 0].flatten(),
                y=tsne_data[indices, 1].flatten(),
                mode='markers',
                text=cluster_data.loc[labels == gc, ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
                hovertemplate="%{text}",
                marker=dict(
                    size=7,
                    line=dict(width=0.5, color='black')
                ),
                name=f'Cluster {gc}'
            )
            fig.add_trace(scatter_trace)

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


##########################


        # Define una paleta de colores vibrantes
        #vibrant_colors = ["#FF6347", "#FFD700", "#8A2BE2", "#FF69B4", "#32CD32", "#00CED1", "#FF4500", "#DA70D6"]


        import pandas as pd
        import plotly.express as px

        # Gráfica 1: phot_g_mean_mag vs g_rp
        fig1 = px.scatter(cluster_data, x="g_rp", y="phot_g_mean_mag", color="gc",
                            hover_data=df_cmd.columns, title="phot_g_mean_mag vs g_rp")
        fig1.update_yaxes(autorange="reversed")

        # Gráfica 2: phot_bp_mean_mag vs bp_rp
        fig2 = px.scatter(cluster_data, x="bp_rp", y="phot_bp_mean_mag", color="gc",
                            hover_data=df_cmd.columns, title="phot_bp_mean_mag vs bp_rp")
        fig2.update_yaxes(autorange="reversed")

        # Gráfica 3: phot_rp_mean_mag vs bp_rp
        fig3 = px.scatter(cluster_data, x="bp_rp", y="phot_rp_mean_mag", color="gc",
                            hover_data=df_cmd.columns, title="phot_rp_mean_mag vs bp_rp")
        fig3.update_yaxes(autorange="reversed")

        # Mostrar las gráficas en Streamlit
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)


        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import export_text, export_graphviz
        import pydotplus
        from IPython.display import Image, display
        from sklearn.tree import plot_tree

        st.write("**Gráfica del árbol de decisión**")
        #st.markdown("En la siguiente Figura se muestra el **diagrama de árbol de desición**, creada a partir de un algoritmo de random forest que explica que **condiciones deben cumplirse** en las variables de interés **(ASMI, FA y Marcha)** para que un paciente se clasificado como miembro de un cluster. **El recorrido de clasificación se lee desde la parte superior**. Dependiendo de si el paciente en cuestión cumple o no con la condición que se lee dentro de cada recuadro, el recorrido se mueve a la **izquierda (si cumple la condición) o a la derecha (si no cumple)**. La clasificación está completa cuando se llega a recuadros que ya no tienen ninguna flecha que los conecte con uno que esté por debajo. Dentro de cada recuadro, la información que se muestra de arriba a abajo es: la condición sobre el parámetro de interés, el índice de ganancia de información *gini*, el número de árboles de desición, de un total de 100, en el que se cumplió la misma condición, la distribución de pacientes de cada cluster que cumple la condición del recuadro y la clasificación")
        
        # Convertir las etiquetas de cluster a valores numéricos
        label_encoder = LabelEncoder()
        cluster_data.loc[:, 'gc'] = label_encoder.fit_transform(cluster_data['gc'])

        # Definir las variables de entrada y salida para el Random Forest
        numeric_data=cluster_data.select_dtypes(include='number')
        X = numeric_data.drop('gc', axis=1)
        y = cluster_data['gc']

        # Crear y entrenar el Random Forest
        random_forest = RandomForestClassifier(random_state=1, min_samples_split=80, ccp_alpha=0.001)
        random_forest.fit(X, y)

        # Obtener los nombres de las columnas originales y convertirlos en cadenas de texto
        column_names = [str(column) for column in X.columns.tolist()]

        # Obtener el mejor árbol del Random Forest
        best_tree_index = random_forest.feature_importances_.argmax()
        best_tree = random_forest.estimators_[best_tree_index]

        # Visualizar las reglas del mejor árbol
        tree_rules = export_text(best_tree, feature_names=column_names)

        # Generar y mostrar la gráfica del árbol
        plt.figure(figsize=(25, 15), dpi=400) 
        #plt.rcParams.update({'font.size': 20}) 
        #plot_tree(best_tree, feature_names=column_names, class_names=[str(cls) for cls in label_encoder.classes_], filled=True, rounded=True)
        plot_tree(best_tree, feature_names=column_names, class_names=[str(cls) for cls in label_encoder.classes_], filled=True, rounded=True, fontsize=16)
        plt.savefig('tree_plot.png', dpi=400, bbox_inches='tight', format='png')
        #plt.savefig('tree_plot.png')  # Guardar la gráfica como imagen
        st.write("**Decision tree chart**")


        st.image('tree_plot.png')





















