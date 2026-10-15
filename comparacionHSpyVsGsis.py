import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

def procesar_y_comparar(path_hyperspy, path_genesis, output_csv='comparacion_HyperSpy_vs_Genesis.csv'):
    # 1. Cargar archivos con separador ';'
    df_hyperspy = pd.read_csv(path_hyperspy, sep=';')
    df_genesis = pd.read_csv(path_genesis, sep=';')
    
    # Eliminar columna Total si existe
    if 'Total' in df_hyperspy.columns:
        df_hyperspy = df_hyperspy.drop(columns=['Total'])
    if 'Total' in df_genesis.columns:
        df_genesis = df_genesis.drop(columns=['Total'])
        
    # REEMPLAZAR NaN POR 0 EN LAS TABLAS ORIGINALES
    df_hyperspy = df_hyperspy.fillna(0)
    df_genesis = df_genesis.fillna(0)
        
    # 2. Despivotar (Melt) para pasar elementos a filas
    melt_hyperspy = df_hyperspy.melt(id_vars=['Archivo'], var_name='Elemento', value_name='Masa_HyperSpy')
    melt_genesis = df_genesis.melt(id_vars=['Archivo'], var_name='Elemento', value_name='Masa_Genesis')
    
    # 3. Merge (Left Join o Inner Join)
    # Al hacer merge por Archivo y Elemento se mantendrán los ceros
    df_comparado = pd.merge(melt_hyperspy, melt_genesis, on=['Archivo', 'Elemento'], how='inner')
    
    # 4. Opcional: Si quieres ignorar los casos donde AMBOS programas dieron 0 (elemento ausente en ambos)
    # elimina filas innecesarias con 0 vs 0:
    df_comparado = df_comparado[(df_comparado['Masa_HyperSpy'] > 0) | (df_comparado['Masa_Genesis'] > 0)]
    
    # Ordenar adecuadamente
    df_comparado = df_comparado.sort_values(by=['Archivo', 'Elemento']).reset_index(drop=True)
    
    # 5. Exportar el nuevo archivo resultado
    df_comparado.to_csv(output_csv, sep=';', index=False)
    print(f"✅ Archivo guardado con éxito como: '{output_csv}'")
    
    # 6. Graficar Scatterplot
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        data=df_comparado, 
        x='Masa_Genesis', 
        y='Masa_HyperSpy', 
        hue='Elemento', 
        style='Elemento',
        s=90, 
        alpha=0.8
    )
    
    # Línea de identidad 1:1
    max_val = max(df_comparado['Masa_Genesis'].max(),df_comparado['Masa_HyperSpy'].max())
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=1.5, label='Línea 1:1')
    
    plt.title('Comparación de Masas (wt.%) incluyendo ausencias (0%)', fontsize=14, fontweight='bold')
    plt.xlabel('Masa Genesis (wt.%)', fontsize=12)
    plt.ylabel('Masa HyperSpy (wt.%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Elemento', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def identificar_diferencias(path_hyperspy, path_genesis):
    # 1. Cargar archivos
    df_hyperspy = pd.read_csv(path_hyperspy, sep=';')
    df_genesis = pd.read_csv(path_genesis, sep=';')
    
    # Limpiar columna Total si existe
    for df in [df_hyperspy, df_genesis]:
        if 'Total' in df.columns:
            df.drop(columns=['Total'], inplace=True)
            
    # 2. Despivotar (Melt)
    melt_hyperspy = df_hyperspy.melt(id_vars=['Archivo'], var_name='Elemento', value_name='Masa_HyperSpy')
    melt_genesis = df_genesis.melt(id_vars=['Archivo'], var_name='Elemento', value_name='Masa_Genesis')
    
    # Reemplazar NaN por 0
    melt_hyperspy['Masa_HyperSpy'] = melt_hyperspy['Masa_HyperSpy'].fillna(0)
    melt_genesis['Masa_Genesis'] = melt_genesis['Masa_Genesis'].fillna(0)

    # 3. OUTER MERGE con indicador
    df_outer = pd.merge(
        melt_hyperspy, 
        melt_genesis, 
        on=['Archivo', 'Elemento'], 
        how='outer', 
        indicator=True
    )
    
    # 4. Filtrar registros que NO cruzaron (quedaron fuera del inner join)
    fuera_de_inner = df_outer[df_outer['_merge'] != 'both']
    
    print(f"Total de registros fuera del inner merge: {len(fuera_de_inner)}")
    
    if not fuera_de_inner.empty:
        print("\n--- REGISTROS QUE QUEDARON FUERA ---")
        print(fuera_de_inner[['Archivo', 'Elemento', 'Masa_HyperSpy', 'Masa_Genesis', '_merge']])
        
        # Obtener nombres únicos de archivos (espectros) no coincidentes
        archivos_excluidos = fuera_de_inner['Archivo'].unique()
        print(f"\nArchivos/espectros afectados ({len(archivos_excluidos)}):")
        for arc in archivos_excluidos:
            print(f" - {arc}")
    else:
        print("\n¡Todos los archivos y elementos coinciden perfectamente en ambos datos!")

    # Opcional: Guardar reporte de exclusiones
    fuera_de_inner.to_csv('registros_excluidos.csv', sep=';', index=False)


def graficar_con_plotly(path_hyperspy, path_genesis,path_salida):
    # 1. Cargar y preparar los datos
    df_hyperspy = pd.read_csv(path_hyperspy, sep=';')
    df_genesis = pd.read_csv(path_genesis, sep=';')
    
    # Eliminar columna 'Total' si existe
    if 'Total' in df_hyperspy.columns:
        df_hyperspy = df_hyperspy.drop(columns=['Total'])
    if 'Total' in df_genesis.columns:
        df_genesis = df_genesis.drop(columns=['Total'])
        
    # Reemplazar NaN por 0
    df_hyperspy = df_hyperspy.fillna(0)
    df_genesis = df_genesis.fillna(0)

        # Renombrar 'wt.%' antes del merge para evitar la confusión de sufijos
    if 'wt.%' in df_hyperspy.columns:
        df_hyperspy = df_hyperspy.rename(columns={'wt.%': 'Masa_HyperSpy'})
    if 'wt.%' in df_genesis.columns:
        df_genesis = df_genesis.rename(columns={'wt.%': 'Masa_Genesis'})
        
    # Despivotar (Melt)
    #melt_hyperspy = df_hyperspy.melt(id_vars=['Archivo'], var_name='Elemento', value_name='Masa_HyperSpy')
    #melt_genesis = df_genesis.melt(id_vars=['Archivo'], var_name='Elemento', value_name='Masa_Genesis')
    
    # Unir ambas tablas (Merge)
    #df_comparado = pd.merge(melt_hyperspy, melt_genesis, on=['Archivo', 'Elemento'], how='inner')
    df_comparado = pd.merge(df_hyperspy, df_genesis, on=['Archivo', 'Elemento'], how='inner')
    
    # Filtrar casos donde ambos valen 0
    df_comparado = df_comparado[(df_comparado['Masa_HyperSpy'] > 0) | (df_comparado['Masa_Genesis'] > 0)]

    df_comparado.to_csv(path_salida, sep=';', index=False)
    print(f"DataFrame fusionado guardado exitosamente en: {path_salida}")

    # 2. Crear el gráfico de dispersión interactivo con Plotly Express
    fig = px.scatter(
        df_comparado, 
        x='Masa_Genesis', 
        y='Masa_HyperSpy', 
        color='Elemento',           # Diferencia colores por Elemento
        symbol='Elemento',          # Diferencia símbolos por Elemento
        hover_name='Archivo',       # Título destacado en el tooltip (Nombre del archivo)
        hover_data={                # Personalización de las etiquetas emergentes (tags)
            'Archivo': True,
            'Elemento': True,
            'Masa_Genesis': ':.2f', # Formato decimal
            'Masa_HyperSpy': ':.2f'
        },
        title='Comparación de Masas (wt.%) incluyendo ausencias (0%)'
    )

    # 3. Personalizar la etiqueta dinámica (Hover Template)
    fig.update_traces(
        marker=dict(size=16),
        hovertemplate="<b>Archivo:</b> %{customdata[0]}<br>" +
                      "<b>Elemento:</b> %{fullData.name}<br>" +
                      "<b>Wt.% Genesis:</b> %{x:.2f}<br>" +
                      "<b>Wt.% HyperSpy:</b> %{y:.2f}<extra></extra>"
    )

    # 4. Agregar la Línea de Identidad 1:1
    max_val = max(df_comparado['Masa_HyperSpy'].max(), df_comparado['Masa_Genesis'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val], 
            y=[0, max_val], 
            mode='lines', 
            name='Línea 1:1',
            line=dict(color='black', dash='dash', width=1.5)
        )
    )

    # 5. Ajustes estéticos del diseño
    fig.update_layout(
        xaxis_title='Masa Genesis (wt.%)',
        yaxis_title='Masa HyperSpy (wt.%)',
        template='plotly_white',
        legend_title_text='Elemento',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        )
    )

    # Mostrar gráfico en navegador o notebook
    fig.show()

    # Opcional: Guardar como un archivo HTML interactivo interactivo
    fig.write_html("grafico_comparativo_interactivo.html")


#procesar_y_comparar('HyperSpy.txt', 'Genesis.txt')
#identificar_diferencias('HyperSpy.txt', 'Genesis.txt')
graficar_con_plotly('HyperSpy.txt', 'Genesis.txt', 'comparacion_HyperSpy_vs_Genesis_excluidos.csv')