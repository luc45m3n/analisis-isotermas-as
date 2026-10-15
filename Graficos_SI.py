import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Cargar el archivo CSV
df = pd.read_csv("indices_saturacion.csv", sep=",")
print(df.head())  # Mostrar las primeras filas para verificar la carga
# Definir los minerales y sus columnas correspondientes
pahse = {
    "Calcite": "SI_Calcite",
    "Aragonite": "SI_Aragonite",
    "Dolomite": "SI_Dolomite",
    "Magnesite": "SI_Magnesite",
    "Gypsum": "SI_Gypsum",
    "Anhydrite": "SI_Anhydrite",
    "Epsomite": "SI_Epsomite",
    "Glauberite": "SI_Glauberite",
    "Halite": "SI_Halite",
    "Sylvite": "SI_Sylvite",
    "Fluorite": "SI_Fluorite",
    "Nahcolite": "SI_Nahcolite",
    "Natron": "SI_Natron",
    "Thenardite": "SI_Thenardite",
    "Mirabilite": "SI_Mirabilite",
}

# Configurar la figura
plt.figure(figsize=(14, 8))

# Línea punteada en y = 0
plt.axhline(0, color="black", linestyle="--", linewidth=1, label="_nolegend_")

# Generar colores y marcadores distintos para cada mineral (> 0)
colors = plt.cm.get_cmap("tab20", len(pahse))
markers = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "d",
    "|",
]

x = df["Sample"]

for i, (pahse_name, col) in enumerate(pahse.items()):
  if col not in df.columns:
    print(f"Advertencia: La columna {col} no se encuentra en el archivo.")
    continue

  y = df[col]

  # Máscaras para separar valores <= 0 y > 0
  mask_le_0 = y <= 0
  mask_gt_0 = y > 0

  # Graficar valores <= 0 en gris claro (sin agregar múltiples entradas en la leyenda)
  plt.scatter(
      x[mask_le_0],
      y[mask_le_0],
      color="lightgray",
      marker="o",
      s=30,
      alpha=0.6,
      label="_nolegend_",
  )

  # Graficar valores > 0 con color y símbolo distintivo para la leyenda
  plt.scatter(
      x[mask_gt_0],
      y[mask_gt_0],
      color=colors(i),
      marker=markers[i % len(markers)],
      s=50,
      label=pahse_name,
  )

  # Opcional: conectar con líneas suaves los puntos de cada mineral
  plt.plot(x, y, color="lightgray", linestyle="-", alpha=0.3, zorder=0)

# Configuración del gráfico
plt.xlabel("Sample", fontsize=12)
plt.ylabel("Saturation Index (SI)", fontsize=12)
plt.title(
    "Saturation indices (SI) of Ca phases by sample", fontsize=14, fontweight="bold"
)
plt.axhline(
    0, color="gray", linestyle=":", label="SI = 0"
)  # Referencia clara

# Rotar las etiquetas del eje X si son muy largas o numerosas
plt.xticks(rotation=45, ha="right")

# Organizar la leyenda fuera o dentro del gráfico de forma limpia
plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", title="Phase"
)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()