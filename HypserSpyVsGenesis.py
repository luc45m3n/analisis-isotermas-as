import pandas as pd
import plotly.express as px


# 1. Cargar los datos (simulando un archivo CSV)
data = "F:/TEM/Muestra5C/GenesisVsHyperSPY_5C.csv"

# Leemos los datos usando pandas
df = pd.read_csv(data, sep=';', skipinitialspace=True)
# 2. Filtrar los datos
df_clean = df[df['Eliminado'] != 'X'].copy()

# Crear columna con el nombre del archivo acortado
df_clean['Muestra_Area'] = df_clean['Archivo'].apply(
    lambda x: x.split('|')[-1].strip() if '|' in x else x.split('_')[-1].strip()
)
df_clean['Archivo_Corto'] = df_clean['Archivo'].apply(
    lambda x: x.split()[0] if x else ''
)

# 3. Crear el gráfico
fig = px.scatter(
    df_clean,
    x='Genesis_wt.',
    y='HyperSpy_wt.',
    color='Elemento',
    symbol= 'Elemento',
    hover_data={'Elemento': True, 'Archivo_Corto': True, 'Muestra_Area':True,
                'Genesis_wt.': ':.2f', 'HyperSpy_wt.': ':.2f'},
    title='Comparación de Cuantificación (wt%): Genesis vs HyperSpy',
    labels={
        'Genesis_wt.': 'Genesis (wt%)',
        'HyperSpy_wt.': 'HyperSpy (wt%)',
        'Elemento': 'Elemento',
        'Archivo_Corto': 'Archivo',
        'Muestra_Area': 'Muestra / Área'
    },
    template='plotly_white',
    size_max=15
)

# 4. Línea de referencia 1:1
max_val = max(df_clean['Genesis_wt.'].max(), df_clean['HyperSpy_wt.'].max())
fig.add_shape(
    type='line',
    x0=0, y0=0, x1=max_val, y1=max_val,
    line=dict(color='gray', dash='dash', width=2),
    name='Línea 1:1'
)

# 5. Ajustes finales (CORREGIDO)
fig.update_layout(
    legend_title_text='Elemento',
    # ✅ Corrección: usar hoverlabel con font en lugar de hoverlabel_style
    hoverlabel=dict(font=dict(size=14)),
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)

fig.update_traces(
    marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)
fig.write_html('comparacion_genesis_vs_hyperspy.html')
fig.show()