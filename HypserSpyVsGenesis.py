import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# Configuración global para calidad de Publicación (Journal Standard)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8,             # Tamaño del texto para 1 columna (8-9 pt)
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'pdf.fonttype': 42,         # Permite editar texto en Illustrator/Inkscape
    'ps.fonttype': 42
})

# 1. Cargar los datos (simulando un archivo CSV)
data = "GenesisVsHyperspy.csv"

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
print(len(df_clean['Archivo_Corto'].unique()))
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

# 3. Preparar paleta amigable para daltónicos
elementos = df_clean['Elemento'].unique()
# Paleta amigable 'colorblind'
colores = plt.cm.tab10(np.linspace(0, 1, len(elementos)))
marcadores = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# 4. Crear la figura (Ancho de 1 columna: 3.5 polegadas)
fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)

# Graficar cada elemento con un color y marcador distinto (accesibilidad)
for i, elem in enumerate(elementos):
    sub_df = df_clean[df_clean['Elemento'] == elem]
    ax.scatter(
        sub_df['Genesis_wt.'], 
        sub_df['HyperSpy_wt.'],
        label=elem,
        color=colores[i % len(colores)],
        marker=marcadores[i % len(marcadores)],
        s=30,                   # Tamaño de punto adecuado
        edgecolor='black',
        linewidth=0.25,
        alpha=0.85
    )

# 5. Línea de referencia 1:1
max_val = max(df_clean['Genesis_wt.'].max(), df_clean['HyperSpy_wt.'].max()) * 1.05
ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1, label='1:1')

# 6. Formato de ejes y grilla
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_xlabel('Genesis (wt%)')
ax.set_ylabel('HyperSpy (wt%)')
ax.grid(True, linestyle=':', alpha=0.6, color='gray')
ax.legend(frameon=True, edgecolor='none', facecolor='white', framealpha=0.8)

plt.tight_layout()

# 7. Guardar en formatos Vectorial (PDF) y Ráster (PNG a alta resolución)
plt.savefig('comparacion_genesis_vs_hyperspy.pdf', format='pdf', bbox_inches='tight')
plt.savefig('comparacion_genesis_vs_hyperspy.png', dpi=300, bbox_inches='tight')
plt.show()