import pandas as pd

# 1. Leer el archivo CSV/TXT original (con delimitador ';')
df = pd.read_csv('5C_nuevoScript_datos_completos.txt', sep=';')

# 2. Hacer el pivot para que 'Archivo' sea el índice y los 'Elemento' sean columnas, 
# tomando los valores de 'wt.%'
tabla = df.pivot(index='Archivo', columns='Elemento', values='wt.%')

# 3. Ordenar las columnas en un orden específico si se desea (ej: Na, Mg, Al, Si, Ca, Fe)
columnas_orden = ['Na', 'Mg', 'Al', 'Si', 'Ca', 'Fe']
# Mantener solo las columnas existentes que coincidan con la lista
columnas_existentes = [col for col in columnas_orden if col in tabla.columns]
tabla = tabla[columnas_existentes]

# 4. Calcular la columna Total
tabla['Total'] = tabla.sum(axis=1)

# 5. Restablecer el índice para que 'Archivo' vuelva a ser una columna normal
tabla = tabla.reset_index()

# 6. Guardar el resultado en un archivo Excel o CSV nuevo
tabla.to_excel('resultado_formateado.xlsx', index=False)
# O si prefieres en formato CSV/TXT:
# tabla.to_csv('resultado_formateado.csv', sep=';', index=False)

print("¡Procesamiento completado con éxito!")
print(tabla.head())