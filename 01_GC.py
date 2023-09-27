#
import streamlit as st
import pandas as pd
import requests

# Título de la aplicación
st.title("Analysis of Color-Magnitude Diagrams of galactic globular clusters")

st.subheader("Individual analysis")
st.markdown("**Instructions:** Please select **the photometry** file (Name_photo.csv) and **surface parameters** file (Name_metal.csv), **that correspond to the same cluster**. The data for each cluster were obtained from the **Gaia DR3 database** (https://gea.esac.esa.int/archive/).")

# URL del repositorio de GitHub
repo_url = "https://github.com/SArcD/GlobClusIA"

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
    selected_files_tuple = st.multiselect("Select CSV files to merge:", [item[0] for item in csv_files])
else:
    st.warning("No CSV files were found within the repository.")

# Función para cargar y fusionar los DataFrames seleccionados por "source_id"
def load_and_merge_dataframes(selected_files):
    try:
        if len(selected_files) < 2:
            st.warning("Please select at least two CSV files for merging.")
            return None

        # Cargar y fusionar los DataFrames seleccionados
        dfs = []
        for selected_file in selected_files:
            csv_url = next(item[1] for item in csv_files if item[0] == selected_file)
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
if "selected_files_tuple" in locals() and len(selected_files_tuple) >= 2:
    merged_df = load_and_merge_dataframes(selected_files_tuple)
    if merged_df is not None:
        # Supongamos que tienes un DataFrame llamado merged_df
        st.write("Merged DataFrame:")
        df = merged_df
        df['source_id'] = df['source_id'].astype(str)
        st.dataframe(df)

        # Obtener información sobre el DataFrame
        num_rows, num_columns = df.shape
        num_missing = df.isnull().sum().sum()
        
        # Mostrar información adicional
        st.write(f"**Number of rows:** {num_rows}")
        st.write(f"**Number of columns:** {num_columns}")
        # Obtener el número de filas con datos faltantes
        #num_rows_with_missing_data = df.isnull().any(axis=1).sum()
        
        # Mostrar el número de filas con datos faltantes
        #st.write(f"**Number of rows with missing data:** {num_rows_with_missing_data}")
        # Identificar filas con datos faltantes (NaN o None)
        filas_con_faltantes = (df.isna().any(axis=1)).sum()
        

        # Mostrar las filas con datos faltantes
        #st.write(df[filas_con_faltantes])
        st.write(f"**Number of rows with missing data:** {filas_con_faltantes}")
        import streamlit as st
        from tabulate import tabulate
        # Crea un expansor con un título
        with st.expander("Parameters taken from Gaia DR3"):
            # Table with parameter names and their meanings in English
            table_data = [
                ["Parameter", "Meaning"], 
                ["**source_id**", "Unique identifier of the source"], 
                #["**ra**", "Right Ascension in the ICRS reference system"], 
                #["**ra_error**", "Standard error of Right Ascension"],
                #["**dec**", "Declination in the ICRS reference system"],
                #["**dec_error**", "Standard error of Declination"],
                #["**parallax**", "Parallax in the ICRS reference system"],
                #["**pmra**", "Proper motion in Right Ascension in the ICRS reference system"],
                #["**pmdec**", "Proper motion in Declination in the ICRS reference system"],
                ["**phot_g_mean_mag**", "Mean integrated magnitude in the G band"],
                ["**phot_bp_mean_mag**", "Mean integrated magnitude in the BP band"],
                ["**phot_rp_mean_mag**", "Mean integrated magnitude in the RP band"],
                ["**bp_rp**", "BP-RP color index"],
                ["**bp_g**", "BP-G color index"],
                ["**g_rp**", "G-RP color index"],
                #["**radial_velocity**", "Combined radial velocity"],
                #["**grvs_mag**", "Mean integrated magnitude in the RVS band"],
                #["**grvs_error**", "Standard error of mean integrated magnitude in the RVS band"],
                #["**non_single_star**", "Indicator of non-single star (binary, variable, etc.)"],
                ["**teff_gspphot**", "Effective temperature estimated from GSP-Phot photometry"],
                ["**logg_gspphot**", "Surface gravity estimated from GSP-Phot photometry"],
                ["**mh_gspphot**", "Metallicity estimated from GSP-Phot photometry"],
                #["**azero_gspphot**", "Extinction estimated from GSP-Phot photometry"],
                #["**ebpminrp_gspphot**", "E(BP-RP) color index estimated from GSP-Phot photometry"]
            ]

            # Renderiza la tabla con formato Markdown
            st.markdown(tabulate(table_data, tablefmt="pipe", headers="firstrow"))


        import streamlit as st
        import pandas as pd
        import requests
        import plotly.express as px  

        st.subheader("Two dimensional plots of cluster parameters ")
        st.markdown("**Instructions:** Select at least two variables to generate a two dimensional plot. Some of the plot's settings can be manipulated on the menu in its upper right corner. The resulting plot can be saved by clicking on the icon with the shape of a camera.")
        # Obtener las columnas numéricas del DataFrame
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_columns) < 2:
            st.warning("There must be two selected columns.")
        else:
            st.write("Plot:")

            # Menús desplegables para seleccionar columnas
            column1 = st.selectbox("Select the horizontal axis for the plot:", numeric_columns)
            column2 = st.selectbox("Select the vertical axis for the plot", numeric_columns)
            # Contenedores vacíos para las gráficas
            plot_container1 = st.empty()
            plot_container2 = st.empty()

            if column2 in ["phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag"]:
                # Botón para generar el gráfico con eje vertical invertido
                if st.button("Make Plot"):
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
                    #plot_container1.plotly_chart(fig)

# Seleccionar las columnas deseadas del DataFrame original
columnas_seleccionadas = ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]
df_cmd = df[columnas_seleccionadas]        
#st.dataframe(df_cmd)
# Mostrar información adicional
#st.write(f"Número de filas: {num_rows}")
#st.write(f"Número de columnas: {num_columns}")
# Obtener el número de filas con datos faltantes
#num_rows_with_missing_data = df_cmd.isnull().any(axis=1).sum()
        
# Mostrar el número de filas con datos faltantes
#st.write(f"Número de filas con datos faltantes: {num_rows_with_missing_data}")
#
################################################



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
#st.dataframe(df_cmd)
# Mostrar información adicional
#st.write(f"Número de filas: {num_rows}")
#st.write(f"Número de columnas: {num_columns}")
# Obtener el número de filas con datos faltantes
#num_rows_with_missing_data = df_cmd.isnull().any(axis=1).sum()
        
# Mostrar el número de filas con datos faltantes
#st.write(f"Número de filas con datos faltantes: {num_rows_with_missing_data}")
# Seleccionar solo las columnas numéricas
#columnas_numericas = df_cmd.select_dtypes(include=[np.number])
#columnas_numericas = df.select_dtypes(include=[np.number])
#df['source_id'] = df['source_id'].astype(str)
columnas_numericas = df_cmd.select_dtypes(include=['number'])
#columnas_numericas = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]
# Normalizar los datos (opcional, pero recomendado para clustering)
scaler = StandardScaler()
columnas_numericas_scaled = scaler.fit_transform(columnas_numericas)

# Calcular la matriz de distancias
dist_matrix = pdist(columnas_numericas_scaled, metric='euclidean')

# Agregar una interfaz de usuario para ingresar el número deseado de clusters




st.title("Hierarchical Clustering")
st.write("Enter the desired number of clusters:")

# Agregar un campo de entrada para el número de clusters
num_clusters = st.number_input("Number of clusters", min_value=1, value=4)

# Verificar si el usuario ha ingresado un número de clusters válido
if st.button("Make Clustering"):
    # Calcular la matriz de enlace utilizando el método de enlace completo (complete linkage)
    Z = linkage(dist_matrix, method='ward')

    # Realizar el clustering jerárquico aglomerativo
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
    cluster_labels = clustering.fit_predict(columnas_numericas_scaled)

    # Agregar la columna de clusters "gc" al DataFrame original
    df_cmd['gc'] = cluster_labels

    # Crear un dendrograma
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=df_cmd.index, leaf_rotation=90)
    plt.title("Dendrograma de Clustering Jerárquico")
    plt.xlabel("Índice de la Muestra")
    plt.ylabel("Distancia")
    st.pyplot(fig)
    #plot_container2.plotly_chart(fig)
    st.write(f"Número de clusters seleccionado: {num_clusters}")
    st.dataframe(df_cmd)
    # Mostrar información adicional
    # Obtener información sobre el DataFrame
    num_rows, num_columns = df_cmd.shape
    num_missing = df_cmd.isnull().sum().sum()    
    st.write(f"**Number of rows:** {num_rows}")
    st.write(f"**Number of columns:** {num_columns}")
    # Obtener el número de filas con datos faltantes
    #num_rows_with_missing_data = df.isnull().any(axis=1).sum()
        
    # Mostrar el número de filas con datos faltantes
    #st.write(f"**Number of rows with missing data:** {num_rows_with_missing_data}")
    # Identificar filas con datos faltantes (NaN o None)
    filas_con_faltantes = (df_cmd.isna().any(axis=1)).sum()
    # Mostrar las filas con datos faltantes
    st.write(f"**Number of rows with missing data:** {filas_con_faltantes}")


##
# Puedes ajustar los parámetros del dendrograma para obtener una visualización más adecuada
# También puedes cortar el dendrograma para obtener grupos específicos


######################## Diagramas de Caja ########################

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st

    st.markdown(
        """
        The figure displays box plots that compare each of the sub-clusters formed by the hierarchical clustering technique. Each box corresponds to a group of stars (the points shown to the left of each box correspond to a specific star). The waistlines of each box are a visual aid to determine if there is enough evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the sub-clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the sub-clusters can differentiate concerning that variable."





 """
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



    import math
    # Obtener los nombres de las columnas numéricas
    #columnas_numericas = df_cmd.select_dtypes(include='number').drop(columns=['gc']).columns

    # Calcular el número de filas y columnas del panel (2 columnas)
    #num_rows = len(columnas_numericas)
    #num_cols = 2  # Dos columnas para cada parámetro

    # Calcular el número de subplots necesarios para mostrar todos los diagramas de caja
    #num_subplots = math.ceil(len(columnas_numericas) / num_cols)

    # Ajustar el espacio vertical y la altura de los subplots
    #subplot_height = 400  # Ajusta la altura según tu preferencia
    #vertical_spacing = 0.004  # Ajusta el espacio vertical según tu preferencia

    # Crear subplots para cada parámetro
    #fig = make_subplots(rows=num_subplots, cols=num_cols, subplot_titles=columnas_numericas, vertical_spacing=vertical_spacing)

    # Crear un gráfico de caja para cada parámetro y comparar los 10 clusters
    #for i, column in enumerate(columnas_numericas):
    #    # Obtener los datos de cada cluster para el parámetro actual
    #    cluster_data = [df_cmd[df_cmd['gc'] == cluster][column] for cluster in range(10)]

    #    # Calcular la posición del subplot en la fila y la columna
    #    row_pos = i // num_cols + 1
    #    col_pos = i % num_cols + 1

    #    # Agregar el gráfico de caja al subplot correspondiente
    #    for j in range(10):
    #        box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'group {j}')
    #        box.hovertemplate = 'id: %{text}'  # Agregar el valor de la columna 'Nombre' al hovertemplate
    #        box.text = df_cmd[df_cmd['gc'] == j]['source_id']  # Asignar los valores de la columna 'Nombre' al texto
    #        fig.add_trace(box, row=row_pos, col=col_pos)

    ## Actualizar el diseño y mostrar el panel de gráficos
    #fig.update_layout(showlegend=False, height=subplot_height * num_subplots, width=800,  # Ajusta el ancho según tu preferencia
    #                  title_text='Comparación de Clusters - Gráfico de Caja',
    #                  margin=dict(t=100, b=100, l=50, r=50))  # Ajustar los márgenes del layout

    ## Mostrar la gráfica de caja en Streamlit
    #st.plotly_chart(fig, use_container_width=True)



    st.markdown("""The Figure displays the box plots in which each of the clusters formed by the hierarchical clustering technique is compared. Each box corresponds to a particular cluster (where on the left side of each one, you can see the points corresponding to the contained patients). The waistlines of each box serve as a visual aid to determine if there is sufficient evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the clusters can differentiate with respect to that variable).""")

    ################### tsne ##############################

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE

    st.markdown("""
        The sub-clusters represent data that, based on their values in each variable, can be considered as more similar to each other than to the rest. However, in many cases, visualizing the sub-clusters can be challenging because the number of variables involved can be very high. The techniques t-SNE and PCA can be used together to create a plot of all the points on a plane. The plot shows the points grouped within each cluster, once the techniques of Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) have been applied. The contours around each cluster correspond to the density of points (where the lines are more concentrated, it signifies a higher point density).""")


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

    #data=df_cmd
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

    # Crear una instancia de t-SNE con los hiperparámetros deseados

    #filtered_df = df2021_pred[df2021_pred['SARCOPENIA'] == 1.0]


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

################################3

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
for cluster_num, cluster_df in dataframes_por_cluster.items():
    st.write(f"Cluster {cluster_num}:")
    st.write(cluster_df)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Supongamos que tienes un DataFrame llamado cluster_1_data con la columna "brillo"

cluster_1_data=dataframes_por_cluster[1]
magnitudes = cluster_1_data["phot_bp_mean_mag"]

# Calcular el KDE de los datos de magnitudes
kde = gaussian_kde(magnitudes)

# Crear un rango de valores de brillo para la estimación
brillo_range = np.linspace(magnitudes.min(), magnitudes.max(), 1000)

# Calcular la PDF suavizada (KDE) en el rango de brillo
pdf_suavizada = kde(brillo_range)

# Encontrar la posición del máximo en la PDF suavizada, que podría corresponder al RGB Bump
posicion_rgb_bump = brillo_range[np.argmax(pdf_suavizada)]

# Mostrar la posición estimada del RGB Bump en Streamlit
st.title("Estimación del RGB Bump mediante KDE")
st.write(f"La posición estimada del RGB Bump en el Cluster 1 es: {posicion_rgb_bump:.2f}")

# Visualizar el KDE y la posición estimada del RGB Bump en Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(brillo_range, pdf_suavizada, label="KDE")
ax.axvline(x=posicion_rgb_bump, color='red', linestyle='--', label="Posición estimada del RGB Bump")
ax.set_xlabel("Magnitud Aparente")
ax.set_ylabel("Densidad de Probabilidad")
ax.set_title("Estimación del RGB Bump mediante KDE")
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit
st.pyplot(fig)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Supongamos que tienes un DataFrame llamado cluster_1_data con la columna "brillo"

cluster_1_data=dataframes_por_cluster[1]
magnitudes = cluster_1_data["phot_g_mean_mag"]

# Calcular el KDE de los datos de magnitudes
kde = gaussian_kde(magnitudes)

# Crear un rango de valores de brillo para la estimación
brillo_range = np.linspace(magnitudes.min(), magnitudes.max(), 1000)

# Calcular la PDF suavizada (KDE) en el rango de brillo
pdf_suavizada = kde(brillo_range)

# Encontrar la posición del máximo en la PDF suavizada, que podría corresponder al RGB Bump
posicion_rgb_bump = brillo_range[np.argmax(pdf_suavizada)]

# Mostrar la posición estimada del RGB Bump en Streamlit
st.title("Estimación del RGB Bump mediante KDE")
st.write(f"La posición estimada del RGB Bump en el Cluster 1 es: {posicion_rgb_bump:.2f}")

# Visualizar el KDE y la posición estimada del RGB Bump en Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(brillo_range, pdf_suavizada, label="KDE")
ax.axvline(x=posicion_rgb_bump, color='red', linestyle='--', label="Posición estimada del RGB Bump")
ax.set_xlabel("Magnitud Aparente")
ax.set_ylabel("Densidad de Probabilidad")
ax.set_title("Estimación del RGB Bump mediante KDE")
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit
st.pyplot(fig)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Supongamos que tienes un DataFrame llamado cluster_1_data con la columna "brillo"

cluster_1_data=dataframes_por_cluster[1]
magnitudes = cluster_1_data["phot_rp_mean_mag"]

# Calcular el KDE de los datos de magnitudes
kde = gaussian_kde(magnitudes)

# Crear un rango de valores de brillo para la estimación
brillo_range = np.linspace(magnitudes.min(), magnitudes.max(), 1000)

# Calcular la PDF suavizada (KDE) en el rango de brillo
pdf_suavizada = kde(brillo_range)

# Encontrar la posición del máximo en la PDF suavizada, que podría corresponder al RGB Bump
posicion_rgb_bump = brillo_range[np.argmax(pdf_suavizada)]

# Mostrar la posición estimada del RGB Bump en Streamlit
st.title("Estimación del RGB Bump mediante KDE")
st.write(f"La posición estimada del RGB Bump en el Cluster 1 es: {posicion_rgb_bump:.2f}")

# Visualizar el KDE y la posición estimada del RGB Bump en Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(brillo_range, pdf_suavizada, label="KDE")
ax.axvline(x=posicion_rgb_bump, color='red', linestyle='--', label="Posición estimada del RGB Bump")
ax.set_xlabel("Magnitud Aparente")
ax.set_ylabel("Densidad de Probabilidad")
ax.set_title("Estimación del RGB Bump mediante KDE")
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit
st.pyplot(fig)


################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

cluster_1_data=dataframes_por_cluster[1]
magnitudes = cluster_1_data["phot_bp_mean_mag"]

# Calcular el KDE de los datos de magnitudes
kde = gaussian_kde(magnitudes)

# Crear un rango de valores de brillo para la estimación
brillo_range = np.linspace(magnitudes.min(), magnitudes.max(), 1000)

# Calcular la PDF suavizada (KDE) en el rango de brillo
pdf_suavizada = kde(brillo_range)

# Encontrar la posición del máximo en la PDF suavizada, que podría corresponder al RGB-tip
posicion_rgb_tip = brillo_range[np.argmax(pdf_suavizada)]

# Mostrar la posición estimada del RGB-tip en Streamlit
st.title("Estimación del RGB-tip mediante KDE")
st.write(f"La posición estimada del RGB-tip en el Cluster 1 es: {posicion_rgb_tip:.2f}")

# Visualizar el KDE y la posición estimada del RGB-tip en Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(brillo_range, pdf_suavizada, label="KDE")
ax.axvline(x=posicion_rgb_tip, color='red', linestyle='--', label="Posición estimada del RGB-tip")
ax.set_xlabel("Magnitud Aparente")
ax.set_ylabel("Densidad de Probabilidad")
ax.set_title("Estimación del RGB-tip mediante KDE")
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit
st.pyplot(fig)



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

cluster_1_data=dataframes_por_cluster[1]
magnitudes = cluster_1_data["phot_g_mean_mag"]

# Calcular el KDE de los datos de magnitudes
kde = gaussian_kde(magnitudes)

# Crear un rango de valores de brillo para la estimación
brillo_range = np.linspace(magnitudes.min(), magnitudes.max(), 1000)

# Calcular la PDF suavizada (KDE) en el rango de brillo
pdf_suavizada = kde(brillo_range)

# Encontrar la posición del máximo en la PDF suavizada, que podría corresponder al RGB-tip
posicion_rgb_tip = brillo_range[np.argmax(pdf_suavizada)]

# Mostrar la posición estimada del RGB-tip en Streamlit
st.title("Estimación del RGB-tip mediante KDE")
st.write(f"La posición estimada del RGB-tip en el Cluster 1 es: {posicion_rgb_tip:.2f}")

# Visualizar el KDE y la posición estimada del RGB-tip en Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(brillo_range, pdf_suavizada, label="KDE")
ax.axvline(x=posicion_rgb_tip, color='red', linestyle='--', label="Posición estimada del RGB-tip")
ax.set_xlabel("Magnitud Aparente")
ax.set_ylabel("Densidad de Probabilidad")
ax.set_title("Estimación del RGB-tip mediante KDE")
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit
st.pyplot(fig)



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

cluster_1_data=dataframes_por_cluster[1]
magnitudes = cluster_1_data["phot_rp_mean_mag"]

# Calcular el KDE de los datos de magnitudes
kde = gaussian_kde(magnitudes)

# Crear un rango de valores de brillo para la estimación
brillo_range = np.linspace(magnitudes.min(), magnitudes.max(), 1000)

# Calcular la PDF suavizada (KDE) en el rango de brillo
pdf_suavizada = kde(brillo_range)

# Encontrar la posición del máximo en la PDF suavizada, que podría corresponder al RGB-tip
posicion_rgb_tip = brillo_range[np.argmax(pdf_suavizada)]

# Mostrar la posición estimada del RGB-tip en Streamlit
st.title("Estimación del RGB-tip mediante KDE")
st.write(f"La posición estimada del RGB-tip en el Cluster 1 es: {posicion_rgb_tip:.2f}")

# Visualizar el KDE y la posición estimada del RGB-tip en Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(brillo_range, pdf_suavizada, label="KDE")
ax.axvline(x=posicion_rgb_tip, color='red', linestyle='--', label="Posición estimada del RGB-tip")
ax.set_xlabel("Magnitud Aparente")
ax.set_ylabel("Densidad de Probabilidad")
ax.set_title("Estimación del RGB-tip mediante KDE")
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit
st.pyplot(fig)


########################################

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
#df_cmd = df_cmd.dropna()
#st.dataframe(df_cmd)
# Mostrar información adicional
#st.write(f"Número de filas: {num_rows}")
#st.write(f"Número de columnas: {num_columns}")
# Obtener el número de filas con datos faltantes
#num_rows_with_missing_data = df_cmd.isnull().any(axis=1).sum()
        
# Mostrar el número de filas con datos faltantes
#st.write(f"Número de filas con datos faltantes: {num_rows_with_missing_data}")
# Seleccionar solo las columnas numéricas
#columnas_numericas = df_cmd.select_dtypes(include=[np.number])
#columnas_numericas = df.select_dtypes(include=[np.number])
#df['source_id'] = df['source_id'].astype(str)
columnas_numericas = cluster_1_data.select_dtypes(include=['number'])
#columnas_numericas = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]
# Normalizar los datos (opcional, pero recomendado para clustering)
scaler = StandardScaler()
columnas_numericas_scaled = scaler.fit_transform(columnas_numericas)

# Calcular la matriz de distancias
dist_matrix = pdist(columnas_numericas_scaled, metric='euclidean')

# Agregar una interfaz de usuario para ingresar el número deseado de clusters


st.title("Hierarchical Clustering")
st.write("Enter the desired number of clusters:")

# Agregar un campo de entrada para el número de clusters
num_clusters = 3


# Calcular la matriz de enlace utilizando el método de enlace completo (complete linkage)
Z = linkage(dist_matrix, method='ward')

# Realizar el clustering jerárquico aglomerativo
clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
cluster_labels = clustering.fit_predict(columnas_numericas_scaled)

# Agregar la columna de clusters "gc" al DataFrame original
cluster_1_data['gc'] = cluster_labels

# Crear un dendrograma
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z, labels=cluster_1_data.index, leaf_rotation=90)
plt.title("Dendrograma de Clustering Jerárquico")
plt.xlabel("Índice de la Muestra")
plt.ylabel("Distancia")
st.pyplot(fig)
#plot_container2.plotly_chart(fig)
st.write(f"Número de clusters seleccionado: {num_clusters}")
st.dataframe(cluster_1_data)
# Obtener información sobre el DataFrame
num_rows, num_columns = cluster_1_data.shape
num_missing = cluster_1_data.isnull().sum().sum()    
st.write(f"**Number of rows:** {num_rows}")
st.write(f"**Number of columns:** {num_columns}")
    

filas_con_faltantes = (cluster_1_data.isna().any(axis=1)).sum()
# Mostrar las filas con datos faltantes
st.write(f"**Number of rows with missing data:** {filas_con_faltantes}")



######################## Diagramas de Caja ########################

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.markdown(
        """
        The figure displays box plots that compare each of the sub-clusters formed by the hierarchical clustering technique. Each box corresponds to a group of stars (the points shown to the left of each box correspond to a specific star). The waistlines of each box are a visual aid to determine if there is enough evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the sub-clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the sub-clusters can differentiate concerning that variable."
 """
    )

# Obtener los nombres de las columnas numéricas
columnas_numericas= cluster_1_data.select_dtypes(include='number').drop(columns=['gc']).columns

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
    cluster_data = [cluster_1_data[cluster_1_data['gc'] == cluster][column] for cluster in range(10)]

    #Agregar el gráfico de caja al subplot correspondiente
    for j in range(10):
        box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'group {j}')
        box.hovertemplate = 'id: %{text}'  # Agregar el valor de la columna 'Nombre' al hovertemplate
        box.text = cluster_1_data[cluster_1_data['gc'] == j]['source_id']  # Asignar los valores de la columna 'Nombre' al texto
        fig.add_trace(box, row=i+1, col=1)

#Actualizar el diseño y mostrar el panel de gráficos
fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
                      title_text='Comparación de Clusters - Gráfico de Caja',
                      margin=dict(t=100, b=100, l=50, r=50))  # Ajustar los márgenes del layout

#Mostrar la gráfica de caja en Streamlit
st.plotly_chart(fig, use_container_width=True)


st.markdown("""The Figure displays the box plots in which each of the clusters formed by the hierarchical clustering technique is compared. Each box corresponds to a particular cluster (where on the left side of each one, you can see the points corresponding to the contained patients). The waistlines of each box serve as a visual aid to determine if there is sufficient evidence of a difference between the clusters (if the waistlines are at the same height, there is no evidence that the clusters can differentiate based on their values in that variable. If they do not match in height, it can be concluded that the clusters can differentiate with respect to that variable).""")

    ################### tsne ##############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

st.markdown("""
        The sub-clusters represent data that, based on their values in each variable, can be considered as more similar to each other than to the rest. However, in many cases, visualizing the sub-clusters can be challenging because the number of variables involved can be very high. The techniques t-SNE and PCA can be used together to create a plot of all the points on a plane. The plot shows the points grouped within each cluster, once the techniques of Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) have been applied. The contours around each cluster correspond to the density of points (where the lines are more concentrated, it signifies a higher point density).""")


numeric_data=cluster_1_data.select_dtypes(include='number')
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

cluster_1_data = cluster_1_data.reset_index(drop=True)
# Reset the indices of the DataFrames
pca_df.reset_index(drop=True, inplace=True)
cluster_1_data.reset_index(drop=True, inplace=True)
pca_df = pd.concat([pca_df, cluster_1_data["source_id"]], axis=1)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

    #data=df_cmd
labels = cluster_1_data['gc']
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
        text=cluster_1_data.loc[labels == gc, ["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "bp_g", "g_rp", "teff_gspphot", "logg_gspphot", "mh_gspphot"]].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
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


    ###

import pandas as pd
import plotly.express as px

# Gráfica 1: phot_g_mean_mag vs g_rp
fig1 = px.scatter(cluster_1_data, x="g_rp", y="phot_g_mean_mag", color="gc",
                        hover_data=cluster_1_data.columns, title="phot_g_mean_mag vs g_rp")
fig1.update_yaxes(autorange="reversed")

# Gráfica 2: phot_bp_mean_mag vs bp_rp
fig2 = px.scatter(cluster_1_data, x="bp_rp", y="phot_bp_mean_mag", color="gc",
                    hover_data=cluster_1_data.columns, title="phot_bp_mean_mag vs bp_rp")
fig2.update_yaxes(autorange="reversed")

# Gráfica 3: phot_rp_mean_mag vs bp_rp
fig3 = px.scatter(cluster_1_data, x="bp_rp", y="phot_rp_mean_mag", color="gc",
                    hover_data=cluster_1_data.columns, title="phot_rp_mean_mag vs bp_rp")
fig3.update_yaxes(autorange="reversed")

# Mostrar las gráficas en Streamlit
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)





