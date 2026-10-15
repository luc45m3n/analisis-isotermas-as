import csv
import pandas as pd
import re

def generar_genesis_vs_hyperspy(
    genesis_path='genesis.csv', 
    hyperspy_path='hyperspy.txt', 
    cuentas_path='CuentasInsuficientes.txt', 
    output_path='GenesisVsHyperspy.csv'
):
    # 1. Cargar lista de espectros/archivos a excluir por cuentas insuficientes
    excluidos = set()
    try:
        with open(cuentas_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'(spectrum\d+_\d+)', line)
                if match:
                    excluidos.add(match.group(1))
    except FileNotFoundError:
        print(f"Advertencia: No se encontró {cuentas_path}. Se omiten exclusiones previas.")

    # 2. Leer genesis.csv de forma robusta con la librería estándar csv
    rows = []
    with open(genesis_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append([cell.strip() for cell in row])

    if len(rows) < 3:
        raise ValueError("El archivo genesis.csv no contiene suficientes filas de datos.")

    header_row_0 = rows[0]
    header_row_1 = rows[1]

    # Ubicar dónde inicia la sección "Weight %"
    start_wt = None
    for idx, cell in enumerate(header_row_0):
        if 'Weight %' in cell:
            start_wt = idx
            break

    if start_wt is None:
        raise ValueError("No se encontró la sección 'Weight %' en el encabezado de genesis.csv")

    # Mapear los elementos dentro de la sección Weight %
    gen_element_cols = {}
    for idx in range(start_wt, len(header_row_1)):
        elem_raw = header_row_1[idx]
        if elem_raw == 'Total' or not elem_raw:
            if elem_raw == 'Total':
                break
            continue
        
        # Limpiar nombre del elemento (ej. 'NaK' -> 'Na', 'SiK' -> 'Si', 'K K' -> 'K')
        elem_clean = re.sub(r'K$', '', elem_raw).strip()
        elem_clean = re.sub(r'\s+', '', elem_clean)
        gen_element_cols[elem_clean] = idx

    # Extraer registros de Genesis
    genesis_records = []
    for row_vals in rows[2:]:
        if not row_vals or not row_vals[0]:
            continue
            
        sample_name = row_vals[0]
        # Extraer ID base (e.g., spectrum20240215150333346_0)
        spec_match = re.search(r'(spectrum\d+_\d+)', sample_name)
        spec_id = spec_match.group(1) if spec_match else sample_name

        for elem, col_idx in gen_element_cols.items():
            wt_val = 0.0
            if col_idx < len(row_vals):
                try:
                    wt_val = float(row_vals[col_idx])
                except ValueError:
                    wt_val = 0.0

            genesis_records.append({
                'Archivo': sample_name,
                'Spec_ID': spec_id,
                'Elemento': elem,
                'Genesis_wt.': wt_val
            })

    df_genesis = pd.DataFrame(genesis_records)

    # 3. Leer hyperspy.txt
    df_hyperspy = pd.read_csv(hyperspy_path, sep=';', skipinitialspace=True)
    df_hyperspy.rename(columns={'wt.%': 'HyperSpy_wt.', 'Archivo': 'Archivo_HS'}, inplace=True)

    df_hyperspy['Spec_ID'] = df_hyperspy['Archivo_HS'].apply(
        lambda x: re.search(r'(spectrum\d+_\d+)', str(x)).group(1) if re.search(r'(spectrum\d+_\d+)', str(x)) else str(x)
    )

    # 4. Cruzar y unificar la lista completa de espectros y elementos
    all_specs = sorted(list(set(df_genesis['Spec_ID']).union(set(df_hyperspy['Spec_ID']))))
    final_rows = []

    for spec in all_specs:
        sub_gen = df_genesis[df_genesis['Spec_ID'] == spec]
        sub_hs = df_hyperspy[df_hyperspy['Spec_ID'] == spec]

        # Tomar la etiqueta de archivo preferentemente desde Genesis, de lo contrario desde HyperSpy
        if not sub_gen.empty:
            archivo_label = sub_gen['Archivo'].iloc[0]
        else:
            archivo_label = sub_hs['Archivo_HS'].iloc[0]

        elems_gen = set(sub_gen['Elemento'])
        elems_hs = set(sub_hs['Elemento'])
        all_elems = sorted(list(elems_gen.union(elems_hs)))

        is_excluido = 'X' if spec in excluidos else ''

        for elem in all_elems:
            gen_match = sub_gen[sub_gen['Elemento'] == elem]
            val_gen = gen_match['Genesis_wt.'].iloc[0] if not gen_match.empty else 0.0

            hs_match = sub_hs[sub_hs['Elemento'] == elem]
            val_hs = hs_match['HyperSpy_wt.'].iloc[0] if not hs_match.empty else 0.0

            # Diferencia relativa/absoluta
            mod_d_wt = round(val_hs - val_gen, 2)

            final_rows.append({
                'Archivo': archivo_label,
                'Elemento': elem,
                'HyperSpy_wt.': round(val_hs, 2),
                'Genesis_wt.': round(val_gen, 2),
                'mod(D_wt)': mod_d_wt,
                'Eliminado': is_excluido
            })

    # 5. Generar archivo de salida separado por ';'
    df_output = pd.DataFrame(final_rows)
    df_output.to_csv(output_path, sep=';', index=False, encoding='utf-8')
    print(f"Archivo generado correctamente: {output_path}")

if __name__ == '__main__':
    generar_genesis_vs_hyperspy()