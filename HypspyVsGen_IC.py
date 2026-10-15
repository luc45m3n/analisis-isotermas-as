import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Cargar y filtrar los datos
df = pd.read_csv("GenesisVsHyperspy.csv", sep=';', skipinitialspace=True)
df_clean = df[df['Eliminado'] != 'X'].copy()

x = pd.to_numeric(df_clean['Genesis_wt.'], errors='coerce').values
y = pd.to_numeric(df_clean['HyperSpy_wt.'], errors='coerce').values

# 2. Ajuste Lineal (OLS) e Intervalo de Predicción al 95%
n = len(x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value**2

max_val = max(np.nanmax(x), np.nanmax(y)) * 1.05
x_lin = np.linspace(0, max_val, 200)
y_fit = slope * x_lin + intercept

y_pred_data = slope * x + intercept
residuals = y - y_pred_data
s_err = np.sqrt(np.sum(residuals**2) / (n - 2))

t_val = stats.t.ppf(0.975, df=n-2)
x_mean = np.mean(x)
pi = t_val * s_err * np.sqrt(1 + 1/n + (x_lin - x_mean)**2 / np.sum((x - x_mean)**2))

# 3. Configuración del Gráfico
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'pdf.fonttype': 42
})

elementos = df_clean['Elemento'].unique()
colores = plt.cm.tab10(np.linspace(0, 1, len(elementos)))
marcadores = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# Se ajusta la figura un poco más ancha para dar espacio a la leyenda
fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)

# Graficar datos
for i, elem in enumerate(elementos):
    sub_df = df_clean[df_clean['Elemento'] == elem]
    ax.scatter(
        sub_df['Genesis_wt.'], 
        sub_df['HyperSpy_wt.'],
        label=elem,
        color=colores[i % len(colores)],
        marker=marcadores[i % len(marcadores)],
        s=20,
        edgecolor='black',
        linewidth=0.25,
        alpha=0.85
    )

# Referencias y Regresión
ax.plot([0, max_val], [0, max_val], color='gray', alpha=0.6 , linestyle='--', linewidth=0.5, label='1:1 Line')
ax.plot(x_lin, y_fit, color='red', linestyle='-', linewidth=0.5, 
        label=f'Fit (y={slope:.2f}x+{intercept:.2f})')

ax.plot(x_lin, y_fit + pi, color='darkred', linestyle=':', linewidth=0.5, label='Prediction Band 95%')
ax.plot(x_lin, y_fit - pi, color='darkred', linestyle=':', linewidth=0.5)
ax.fill_between(x_lin, y_fit - pi, y_fit + pi, color='red', alpha=0.06)

ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_xlabel('Genesis (wt%)')
ax.set_ylabel('HyperSpy (wt%)')
ax.grid(True, linestyle=':', alpha=0.5, color='gray')


# =========================================================================
# 4. Anotación de R^2 dentro del área del gráfico (esquina superior izquierda)
# =========================================================================
texto_stats = f"$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r_squared:.4f}$"

ax.annotate(
    texto_stats,
    xy=(0.05, 0.92),             # Posición en coordenadas relativas al eje (5% X, 92% Y)
    xycoords='axes fraction',    # Usa ejes (0 a 1) para que no dependa de la escala de wt%
    va='top', ha='left',
    fontsize=8,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.85)
)
# 4. Ubicar la leyenda en el margen derecho exterior
ax.legend(
    loc='lower right',
    #bbox_to_anchor=(1.02, 0.5),  # Desplaza la leyenda justo a la derecha del eje Y central
    fontsize=7,
    frameon=True,
    edgecolor='none',
    facecolor='white',
    framealpha=0.85,
    alignment='right'
)

plt.tight_layout()

# Al guardar con bbox_inches='tight', se recalcula el lienzo para incluir la leyenda externa
plt.savefig('comparacion_genesis_vs_hyperspy_leyenda_externa.pdf', format='pdf', bbox_inches='tight')
plt.savefig('comparacion_genesis_vs_hyperspy_leyenda_externa.png', dpi=300, bbox_inches='tight')
plt.show()