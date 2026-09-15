#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de isotermas de adsorción con ODR
Usa promedios de réplicas con error de la media (σ/√n)
Modelos: Langmuir, Freundlich, Sips, Langmuir con offset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.stats import t

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. MODELOS DE ISOTERMA
# ══════════════════════════════════════════════════════════════════════════════

def langmuir(Ce, Qmax, KL):
    return (Qmax * KL * Ce) / (1 + KL * Ce)

def freundlich(Ce, Kf, n):
    return Kf * np.power(Ce, 1.0 / n)

def sips(Ce, Qmax, KL, n):
    return (Qmax * np.power(KL * Ce, n)) / (1 + np.power(KL * Ce, n))

def langmuir_offset(Ce, Q0, Qmax, KL):
    """Langmuir con offset (sitio de altísima afinidad + sitio convencional)"""
    return Q0 + (Qmax * KL * Ce) / (1 + KL * Ce)

# Wrappers para ODR (firma: beta, x)
def langmuir_odr(beta, x):         return langmuir(x, *beta)
def freundlich_odr(beta, x):       return freundlich(x, *beta)
def sips_odr(beta, x):             return sips(x, *beta)
def langmuir_offset_odr(beta, x):  return langmuir_offset(x, *beta)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA Y PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_y_preparar_promedios(csv_path, tolerancia_co=0.01):
    """
    Carga CSV, agrupa réplicas por Co similar,
    calcula promedios y error de la media (σ/√n).
    """
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Corregir typos comunes
    df.rename(columns={
        'simga_co': 'sigma_co', 'simga_ce': 'sigma_ce',
        'simga_v': 'sigma_v', 'simga_m': 'sigma_m', 'simga_qe': 'sigma_qe'
    }, inplace=True)
    
    # Columna de condición (pH)
    if 'ph' in df.columns:
        df['condicion'] = df['ph'].apply(lambda x: f'pH_{x}')
    else:
        df['condicion'] = 'condicion_1'
    
    # Renombrar columnas clave
    df.rename(columns={'ce': 'Ce', 'qe': 'Qe', 'co': 'Co'}, inplace=True)
    
    # Filtrar puntos válidos (excluir blancos)
    df = df[(df['Ce'] > 0) & (df['Qe'] > 0)].copy()
    
    # AGRUPAR RÉPLICAS POR Co SIMILAR
    df_sorted = df.sort_values(['condicion', 'Co']).reset_index(drop=True)
    
    grupos = []
    for cond in df_sorted['condicion'].unique():
        sub = df_sorted[df_sorted['condicion'] == cond].copy()
        sub = sub.sort_values('Co').reset_index(drop=True)
        
        grupo_id = 0
        co_anterior = None
        
        for idx, row in sub.iterrows():
            if co_anterior is None or abs(row['Co'] - co_anterior) / co_anterior > tolerancia_co:
                grupo_id += 1
                co_anterior = row['Co']
            sub.at[idx, 'grupo'] = grupo_id
        
        grupos.append(sub)
    
    df_agrupado = pd.concat(grupos, ignore_index=True)
    
    # CALCULAR PROMEDIOS Y DESVÍO ESTÁNDAR POR GRUPO
    df_stats = df_agrupado.groupby(['condicion', 'grupo']).agg({
        'Co': 'mean',
        'Ce': ['mean', 'std', 'count'],
        'Qe': ['mean', 'std']
    }).reset_index()
    
    df_stats.columns = ['condicion', 'grupo', 'Co', 'Ce_mean', 'Ce_std', 'Ce_count', 'Qe_mean', 'Qe_std']
    
    # ERROR DE LA MEDIA: σ_media = σ / √n
    #df_stats['sigma_Ce'] = df_stats['Ce_std'] / np.sqrt(df_stats['Ce_count'])
    #df_stats['sigma_Qe'] = df_stats['Qe_std'] / np.sqrt(df_stats['Ce_count'])
    df_stats['sigma_Ce'] = df_stats['Ce_std'] 
    df_stats['sigma_Qe'] = df_stats['Qe_std']
    # CORREGIR SIGMA=0 o NaN (réplicas idénticas o punto único)
    mask_ce = df_stats['sigma_Ce'].isna() | (df_stats['sigma_Ce'] == 0)
    df_stats.loc[mask_ce, 'sigma_Ce'] = df_stats.loc[mask_ce, 'Ce_mean'] * 0.01
    
    mask_qe = df_stats['sigma_Qe'].isna() | (df_stats['sigma_Qe'] == 0)
    df_stats.loc[mask_qe, 'sigma_Qe'] = df_stats.loc[mask_qe, 'Qe_mean'] * 0.05
    
    print(f"✅ Datos preparados: {len(df_stats)} puntos (promedios de réplicas)")
    for cond in df_stats['condicion'].unique():
        n = len(df_stats[df_stats['condicion'] == cond])
        print(f"   {cond}: {n} puntos")
    
    return df_stats

# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAFICAR DATOS
# ══════════════════════════════════════════════════════════════════════════════

def graficar_datos(df_stats):
    """Grafica los promedios con barras de error (error de la media)"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    condiciones = df_stats['condicion'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(condiciones)))
    markers = ['o', 's', '^', 'D']
    
    for i, cond in enumerate(condiciones):
        sub = df_stats[df_stats['condicion'] == cond]
        ax.errorbar(sub['Ce_mean'], sub['Qe_mean'], 
                    xerr=sub['sigma_Ce'], yerr=sub['sigma_Qe'],
                    fmt=markers[i % len(markers)], markersize=8, capsize=4, capthick=1.5,
                    color=colors[i], ecolor=colors[i],
                    label=f'{cond} (n={int(sub["Ce_count"].iloc[0])} réplicas)', 
                    linewidth=1, elinewidth=1, alpha=0.8)
    
    ax.set_xlabel(r'Concentración en equilibrio, $C_e$ ($\mu g/L$)', fontsize=11)
    ax.set_ylabel('Capacidad de adsorción, $q_e$ (mg/g)', fontsize=11)
    ax.set_title('Isotermas de Adsorción (promedios ± error de la media)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('datos_promedios.png', dpi=300, bbox_inches='tight')
    print("\n📊 Gráfico guardado: datos_promedios.png")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 4. ESTADÍSTICOS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_estadisticos(y_exp, y_pred, sigma_y, n_params):
    """Calcula R², R²_adj, RMSE, χ²/dof y AICc"""
    n = len(y_exp)
    dof = n - n_params
    
    ss_res = np.sum((y_exp - y_pred)**2)
    ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    r2_adj = 1 - (ss_res / dof) / (ss_tot / (n - 1)) if dof > 0 else np.nan
    rmse = np.sqrt(ss_res / n)
    
    chi2 = np.sum(((y_exp - y_pred) / sigma_y)**2)
    chi2_red = chi2 / dof if dof > 0 else np.nan
    
    log_lik = -0.5 * chi2
    aic = 2 * n_params - 2 * log_lik
    aicc = aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1) if n > n_params + 1 else np.nan
    
    return r2, r2_adj, rmse, chi2_red, aicc

# ══════════════════════════════════════════════════════════════════════════════
# 5. AJUSTE DE MODELOS CON ODR
# ══════════════════════════════════════════════════════════════════════════════

def ajustar_modelos(df_stats):
    """Ajusta los 4 modelos con ODR sobre los promedios"""
    condiciones = df_stats['condicion'].unique()
    
    modelos = {
        'Langmuir': {
            'func_odr': langmuir_odr,
            'func': langmuir,
            'beta0': [80, 0.01],
            'params': ['Qmax', 'KL'],
            'n_params': 2
        },
        'Freundlich': {
            'func_odr': freundlich_odr,
            'func': freundlich,
            'beta0': [10.0, 2.0],
            'params': ['Kf', 'n'],
            'n_params': 2
        },
        'Sips': {
            'func_odr': sips_odr,
            'func': sips,
            'beta0': [80, 0.01, 0.6],
            'params': ['Qmax', 'KL', 'n'],
            'n_params': 3
        },
        'Langmuir_offset': {
            'func_odr': langmuir_offset_odr,
            'func': langmuir_offset,
            'beta0': [7, 50, 0.01],
            'params': ['Q0', 'Qmax', 'KL'],
            'n_params': 3
        }
    }
    
    resultados_por_condicion = {}
    
    for cond in condiciones:
        sub = df_stats[df_stats['condicion'] == cond]
        Ce = sub['Ce_mean'].values
        Qe = sub['Qe_mean'].values
        sCe = sub['sigma_Ce'].values
        sQe = sub['sigma_Qe'].values
        
        print(f"\n Ajustando modelos para {cond} ({len(Ce)} puntos)...")
        print("-" * 70)
        
        resultados = {}
        for nombre, config in modelos.items():
            try:
                # Ajustar beta0 basado en datos observados
                beta0_ajustado = config['beta0'].copy()
                if 'Qmax' in config['params']:
                    idx = config['params'].index('Qmax')
                    beta0_ajustado[idx] = max(Qe) * 1.2
                if 'Q0' in config['params']:
                    idx = config['params'].index('Q0')
                    beta0_ajustado[idx] = max(Qe) * 0.1
                
                data = RealData(x=Ce, y=Qe, sx=sCe, sy=sQe)
                model = Model(config['func_odr'])
                odr = ODR(data, model, beta0=beta0_ajustado, maxit=10000)
                result = odr.run()
                
                popt = result.beta
                perr = result.sd_beta
                Qe_pred = config['func'](Ce, *popt)
                
                r2, r2_adj, rmse, chi2_red, aicc = calcular_estadisticos(Qe, Qe_pred, sQe, config['n_params'])
                
                resultados[nombre] = {
                    'params': dict(zip(config['params'], popt)),
                    'params_err': dict(zip(config['params'], perr)),
                    'R2': r2,
                    'R2_adj': r2_adj,
                    'RMSE': rmse,
                    'chi2_red': chi2_red,
                    'AICc': aicc,
                    'func': config['func'],
                    'popt': popt,
                    'success': np.all(np.isfinite(popt)) and np.all(np.isfinite(perr))
                }
                
                estado = "✓" if resultados[nombre]['success'] else "⚠"
                print(f"{estado} {nombre:18s}: R²={r2:.4f}, χ²/dof={chi2_red:.3f}, AICc={aicc:.2f}")
                
            except Exception as e:
                print(f"✗ {nombre:18s}: Error - {e}")
        
        print("-" * 70)
        resultados_por_condicion[cond] = resultados
    
    return resultados_por_condicion

# ══════════════════════════════════════════════════════════════════════════════
# 6. IMPRIMIR PARÁMETROS CON IC 95%
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_parametros_ajustados(resultados_por_condicion):
    """Imprime parámetros con incertidumbres e intervalo de confianza 95%"""
    for cond, resultados in resultados_por_condicion.items():
        print(f"\n{'='*80}")
        print(f"PARÁMETROS AJUSTADOS - {cond}")
        print(f"{'='*80}")
        
        for nombre, res in resultados.items():
            if not res['success']:
                print(f"\n  {nombre}: ❌ No convergió")
                continue
            
            n = len(res['popt']) * 2  # aproximación de grados de libertad
            p = len(res['popt'])
            dof = max(n - p - 1, 1)
            t_val = t.ppf(0.975, dof)
            
            print(f"\n  {nombre}:")
            print("-" * 60)
            for param, val in res['params'].items():
                err = res['params_err'][param]
                ci_inf = val - t_val * err
                ci_sup = val + t_val * err
                print(f"    {param:<10}: {val:.4g} ± {err:.4g}  [IC 95%: {ci_inf:.4g}, {ci_sup:.4g}]")
            
            print(f"\n    Estadísticos:")
            print(f"      R²     = {res['R2']:.4f}")
            print(f"      R²_adj = {res['R2_adj']:.4f}")
            print(f"      χ²/dof = {res['chi2_red']:.3f}")
            print(f"      AICc   = {res['AICc']:.2f}")
            print(f"      RMSE   = {res['RMSE']:.4f}")
        
        print(f"\n{'='*80}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_tabla_comparativa(resultados_por_condicion):
    for cond, resultados in resultados_por_condicion.items():
        print(f"\n📊 TABLA COMPARATIVA - {cond}")
        print("=" * 95)
        print(f"{'Modelo':<18} {'R²':<10} {'R²_adj':<10} {'RMSE':<10} {'χ²/dof':<10} {'AICc':<12} {'ΔAICc':<10}")
        print("-" * 95)
        
        aicc_values = [r['AICc'] for r in resultados.values() if not np.isnan(r['AICc'])]
        if not aicc_values:
            print("❌ Ningún modelo convergió")
            continue
        
        mejor_aicc = min(aicc_values)
        
        for nombre, res in resultados.items():
            delta_aicc = res['AICc'] - mejor_aicc if not np.isnan(res['AICc']) else np.nan
            print(f"{nombre:<18} "
                  f"{res['R2']:<10.4f} "
                  f"{res['R2_adj']:<10.4f} "
                  f"{res['RMSE']:<10.4f} "
                  f"{res['chi2_red']:<10.3f} "
                  f"{res['AICc']:<12.2f} "
                  f"{delta_aicc:<10.2f}")
        
        print("=" * 95)

def seleccionar_mejores_modelos(resultados_por_condicion):
    mejores = {}
    
    for cond, resultados in resultados_por_condicion.items():
        modelos_validos = {k: v for k, v in resultados.items() if not np.isnan(v['AICc']) and v['success']}
        
        if not modelos_validos:
            print(f"❌ Ningún modelo convergió para {cond}")
            continue
        
        mejor = min(modelos_validos, key=lambda k: modelos_validos[k]['AICc'])
        mejores[cond] = mejor
        
        print(f"\n🏆 {cond} - MEJOR MODELO: {mejor}")
        print(f"   AICc = {resultados[mejor]['AICc']:.2f}")
        print(f"   R²_adj = {resultados[mejor]['R2_adj']:.4f}")
        print(f"   χ²/dof = {resultados[mejor]['chi2_red']:.3f}")
    
    return mejores

# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAFICAR AJUSTES
# ══════════════════════════════════════════════════════════════════════════════

def graficar_ajustes(df_stats, resultados_por_condicion, mejores_modelos):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    condiciones = df_stats['condicion'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(condiciones)))
    markers = ['o', 's', '^', 'D']
    
    # Graficar datos experimentales
    for i, cond in enumerate(condiciones):
        sub = df_stats[df_stats['condicion'] == cond]
        ax.errorbar(sub['Ce_mean'], sub['Qe_mean'], 
                    xerr=sub['sigma_Ce'], yerr=sub['sigma_Qe'],
                    fmt=markers[i % len(markers)], markersize=8, capsize=4, capthick=1.5,
                    color=colors[i], ecolor=colors[i],
                    label=f'{cond} (datos)', linewidth=1, elinewidth=1, alpha=0.7, zorder=5)
    
    # Graficar curvas ajustadas
    Ce_smooth = np.linspace(df_stats['Ce_mean'].min() * 0.95, df_stats['Ce_mean'].max() * 1.05, 300)
    
    for i, cond in enumerate(condiciones):
        if cond in mejores_modelos:
            mejor = mejores_modelos[cond]
            res = resultados_por_condicion[cond][mejor]
            Qe_pred = res['func'](Ce_smooth, *res['popt'])
            
            ax.plot(Ce_smooth, Qe_pred, 
                    color=colors[i], linewidth=2.5, linestyle='-',
                    label=f'{cond} - {mejor}', zorder=4)
    
    ax.set_xlabel(r'Concentración en equilibrio, $C_e$ ($\mu g/L$)', fontsize=11)
    ax.set_ylabel('Capacidad de adsorción, $q_e$ (mg/g)', fontsize=11)
    ax.set_title('Isotermas de Adsorción - Ajuste ODR (promedios)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ajustes_promedios.png', dpi=300, bbox_inches='tight')
    print("\n📊 Gráfico guardado: ajustes_promedios.png")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 9. EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CSV_PATH = 'C://Users//Lucas//Documents//CNEA//Latex//Informe de Avance 2026-2027//python//datos_isotermas.csv'  # ← AJUSTA ESTA RUTA
    
    print("=" * 80)
    print("  ANÁLISIS DE ISOTERMAS DE ADSORCIÓN CON ODR")
    print("  Promedios de réplicas con error de la media (σ/√n)")
    print("=" * 80)
    
    # Paso 1: Cargar y preparar datos
    print("\n[1/6] Cargando datos...")
    df_stats = cargar_y_preparar_promedios(CSV_PATH, tolerancia_co=0.01)
    
    # Paso 2: Graficar datos
    print("\n[2/6] Graficando datos...")
    graficar_datos(df_stats)
    
    # Paso 3: Ajustar modelos
    print("\n[3/6] Ajustando modelos con ODR...")
    resultados_por_condicion = ajustar_modelos(df_stats)
    
    if not resultados_por_condicion:
        print("❌ No se pudo ajustar ningún modelo")
        exit(1)
    
    # Paso 4: Imprimir parámetros
    print("\n[4/6] Imprimiendo parámetros ajustados...")
    imprimir_parametros_ajustados(resultados_por_condicion)
    
    # Paso 5: Tabla comparativa
    print("\n[5/6] Generando tabla comparativa...")
    imprimir_tabla_comparativa(resultados_por_condicion)
    mejores_modelos = seleccionar_mejores_modelos(resultados_por_condicion)
    
    if not mejores_modelos:
        print("❌ No se pudo seleccionar el mejor modelo")
        exit(1)
    
    # Paso 6: Graficar ajustes
    print("\n[6/6] Graficando ajustes finales...")
    graficar_ajustes(df_stats, resultados_por_condicion, mejores_modelos)
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)