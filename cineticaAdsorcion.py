#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Análisis completo de cinética de adsorción As(V) con datos reales."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.stats import t as student_t
from scipy.optimize import curve_fit

# ═════════════ 0. CONFIGURACIÓN ═════════════
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 11, "axes.titlesize": 12})

# ═════════════ 1. MODELOS CINÉTICOS ═════════════
def pfo(t, qe, k1):
    """Pseudo-primer orden (Lagergren 1898)."""
    return qe * (1.0 - np.exp(-k1 * t))

def pso(t, qe, k2):
    """Pseudo-segundo orden (Ho & McKay 1999)."""
    return (qe**2 * k2 * t) / (1.0 + qe * k2 * t)

def elovich(t, alpha, beta):
    """Elovich (Elovich & Zhabrova 1940)."""
    t_safe = np.where(t <= 0, 1e-12, t)
    return (1.0/beta) * np.log(alpha * beta) + (1.0/beta) * np.log(t_safe)

def ipd(t, kid, C):
    """Difusión intrapartícula (Weber & Morris 1963)."""
    return kid * np.sqrt(t) + C

def avrami(t, qe, kAv, n):
    """Avrami fraccional (Lima et al. 2009)."""
    return qe * (1.0 - np.exp(-((kAv * t)**n)))

# Wrappers para ODR
def pfo_odr(b, x):      return pfo(x, *b)
def pso_odr(b, x):      return pso(x, *b)
def elovich_odr(b, x):  return elovich(x, *b)
def ipd_odr(b, x):      return ipd(x, *b)
def avrami_odr(b, x):   return avrami(x, *b)

MODELOS_CINETICA = {
    'PFO':     dict(func=pfo,     odr=pfo_odr,     beta0=[50, 0.01],      params=['qe', 'k1']),
    'PSO':     dict(func=pso,     odr=pso_odr,     beta0=[50, 0.001],     params=['qe', 'k2']),
    'Elovich': dict(func=elovich, odr=elovich_odr, beta0=[1.0, 0.5],      params=['alpha', 'beta']),
    'IPD':     dict(func=ipd,     odr=ipd_odr,     beta0=[1.0, 5.0],      params=['kid', 'C']),
    'Avrami':  dict(func=avrami,  odr=avrami_odr,  beta0=[50, 0.01, 1.0], params=['qe', 'kAv', 'n']),
}

# ═════════════ 2. UTILIDADES ═════════════
def interpretar_cv(cv):
    """Clasifica el Coeficiente de Variación."""
    if np.isnan(cv): return "N/A"
    if cv < 5:    return "Excelente"
    if cv < 10:   return "Bueno"
    if cv < 20:   return "Aceptable"
    if cv < 30:   return "Alto"
    return "Muy alto"

def calcular_estadisticos(y, yp, sy, k):
    """Calcula R², R²_adj, RMSE, χ²/dof, AICc."""
    n, dof = len(y), len(y) - k
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - y.mean())**2)
    chi2 = np.sum(((y - yp)/sy)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    r2a = 1 - (ss_res/dof)/(ss_tot/(n-1)) if dof > 0 else np.nan
    rmse = np.sqrt(ss_res/n)
    chi2r = chi2/dof if dof > 0 else np.nan
    aic = 2*k + chi2
    aicc = aic + 2*k*(k+1)/(n-k-1) if n > k+1 else np.nan
    return r2, r2a, rmse, chi2r, aicc

# ═════════════ 3. CARGA DE DATOS ═════════════
def cargar_cinetica_real(csv_path):
    """
    Carga el CSV con formato real:
    condicion, Muestras, t[min.], peso1, peso2, peso3, masa[mg], vol_L, 
    Ct1, Ct2, Ct3, Co1, Co2, Co3
    
    Maneja valores NaN en Ct2, Ct3, Co2, Co3 para DE y DM.
    """
    df = pd.read_csv(csv_path, sep=";")
    
    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace('[mg]', '').replace('[min]', '').replace('.', '') for c in df.columns]
    
    print(f"Columnas detectadas: {list(df.columns)}")
    print(f"Filas cargadas: {len(df)}")
    
    # Verificar columnas obligatorias
    cols_req = ['condicion', 't', 'masa', 'vol_l', 'ct1', 'co1']
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"❌ Faltan columnas: {faltantes}")
    
    # Calcular promedios de Co y Ct (manejando NaN)
    df['co_prom'] = df[['co1', 'co2', 'co3']].mean(axis=1, skipna=True)
    df['ct_prom'] = df[['ct1', 'ct2', 'ct3']].mean(axis=1, skipna=True)
    
    # Calcular qt con masa individual
    df['qt'] = (df['co_prom'] - df['ct_prom']) * df['vol_l'] / df['masa']
    
    # CV intra-muestra (FIASS) - solo para DI que tiene 3 mediciones
    df['ct_cv_fiass'] = df[['ct1', 'ct2', 'ct3']].std(axis=1, ddof=1, skipna=True) / df['ct_prom'] * 100
    df['co_cv_fiass'] = df[['co1', 'co2', 'co3']].std(axis=1, ddof=1, skipna=True) / df['co_prom'] * 100
    
    # Filtrar qt negativos (ruido analítico en tiempos muy cortos)
    qt_neg = (df['qt'] < 0).sum()
    if qt_neg > 0:
        print(f"⚠ {qt_neg} valores de qt < 0 detectados (ruido analítico)")
    
    print(f"\n✓ Condiciones: {sorted(df['condicion'].unique())}")
    print(f"✓ Tiempos: {sorted(df['t'].unique())} min")
    print(f"✓ Rango de masa: {df['masa'].min():.4f} – {df['masa'].max():.4f} mg")
    print(f"✓ Rango de qt: {df['qt'].min():.4f} – {df['qt'].max():.4f} µg/mg")
    
    return df

# ═════════════ 4. AGRUPACIÓN POR TIEMPO ════════════
def agrupar_cinetica(df):
    """Agrupa por (condición, tiempo) y calcula estadísticas."""
    grupos = []
    
    for (cond, t), sub in df.groupby(['condicion', 't']):
        n = len(sub)
        qt_mean = sub['qt'].mean()
        qt_std = sub['qt'].std(ddof=1) if n > 1 else np.nan
        qt_cv = (qt_std / qt_mean * 100) if (n > 1 and qt_mean != 0) else np.nan
        
        # IC 95%
        if n > 1:
            tc = student_t.ppf(0.975, n - 1)
            qt_ci95 = tc * qt_std / np.sqrt(n)
        else:
            qt_ci95 = np.nan
        
        # CV FIASS promedio
        ct_cv_fiass_mean = sub['ct_cv_fiass'].mean()
        
        grupos.append({
            'condicion': cond,
            't': t,
            'n_muestras': n,
            'co_mean': sub['co_prom'].mean(),
            'qt_mean': qt_mean,
            'qt_std': qt_std,
            'qt_cv': qt_cv,
            'qt_ci95': qt_ci95,
            'ct_cv_fiass_mean': ct_cv_fiass_mean,
        })
    
    st = pd.DataFrame(grupos)
    
    # Piso para ODR
    st['sy'] = np.maximum(st['qt_std'].fillna(0.0), 
                          0.01 * np.abs(st['qt_mean']) + 1e-6)
    st['sx'] = 0.0
    
    return st

def reportar_estadisticas(st):
    """Reporte detallado."""
    print(f"\n{'='*100}")
    print("📈 ESTADÍSTICAS CINÉTICAS")
    print(f"{'='*100}")
    
    for cond in sorted(st['condicion'].unique()):
        s = st[st['condicion'] == cond]
        print(f"\n🔬 {cond}")
        print(f"  {'t(min)':>7} │ {'n':>2} │ {'C0':>10} │ {'qt_mean':>10} {'qt_std':>9} "
              f"{'CV(%)':>14} {'IC95':>9} │ {'CV_FIASS':>10}")
        print("  " + "─"*95)
        
        for _, r in s.iterrows():
            cv_txt = (f"{r['qt_cv']:>6.2f}% [{interpretar_cv(r['qt_cv']):>9}]"
                      if np.isfinite(r['qt_cv']) else "     n/a (n=1)   ")
            
            print(f"  {r['t']:>7.0f} │ {r['n_muestras']:>2} │ {r['co_mean']:>10.2f} │ "
                  f"{r['qt_mean']:>10.4f} "
                  f"{(r['qt_std'] if np.isfinite(r['qt_std']) else np.nan):>9.4f} "
                  f"{cv_txt} "
                  f"{(r['qt_ci95'] if np.isfinite(r['qt_ci95']) else np.nan):>9.4f} │ "
                  f"{r['ct_cv_fiass_mean']:>9.2f}%")
        
        cvp = np.nanmean(s['qt_cv'])
        print(f"  → CV inter-muestra promedio: {cvp:.2f}% [{interpretar_cv(cvp)}]")

# ═════════════ 5. AJUSTE DE MODELOS ════════════
def valido_fisico_cinetica(nombre, p):
    """Validación física de parámetros."""
    if not np.all(np.isfinite(p)): return False
    if nombre in ('PFO', 'PSO', 'Avrami'):
        return p[0] > 0 and p[1] > 0
    if nombre == 'Elovich':
        return p[0] > 0 and p[1] > 0
    if nombre == 'IPD':
        return p[0] > 0
    return True

def estimar_p0_pso(t, qt):
    """Estima qe y k2 para PSO usando la forma linealizada t/qt vs t."""
    # Evitar división por cero o valores negativos tempranos
    mask = qt > 0
    if mask.sum() < 3: return [qt.max(), 0.001]
    
    t_pos, qt_pos = t[mask], qt[mask]
    # t/qt = 1/(k2*qe^2) + (1/qe)*t
    # y = b + m*x  => m = 1/qe, b = 1/(k2*qe^2)
    m, b = np.polyfit(t_pos, t_pos/qt_pos, 1)
    
    if m <= 0: return [qt.max(), 0.001] # Fallback si la pendiente es negativa (ruido)
    
    qe_est = 1.0 / m
    k2_est = 1.0 / (b * qe_est**2) if b > 0 else 0.001
    
    # Limitar qe_est para que no explote si b es muy pequeño
    qe_est = max(qe_est, qt.max()) 
    
    return [qe_est, k2_est]

def estimar_p0_pfo(t, qt):
    """Estima qe y k1 para PFO."""
    qe_est = qt.max() * 1.1
    # ln(qe - qt) = ln(qe) - k1*t
    mask = (qe_est - qt) > 0
    if mask.sum() < 3: return [qe_est, 0.01]
    
    ln_diff = np.log(qe_est - qt[mask])
    m, b = np.polyfit(t[mask], ln_diff, 1)
    k1_est = -m if m < 0 else 0.01
    return [qe_est, k1_est]

def ajustar_modelos_cinetica(df_stats):
    """Ajusta modelos cinéticos usando curve_fit ponderado (más robusto que ODR para cinética)."""
    resultados = {}
    
    for cond, sub in df_stats.groupby('condicion'):
        print(f"\n{'='*80}")
        print(f"📐 AJUSTE CINÉTICO: {cond}")
        print(f"{'='*80}")
        
        t_vals = sub['t'].values.astype(float)
        qt_vals = sub['qt_mean'].values.astype(float)
        sy = sub['sy'].values.astype(float)
        
        # Normalizar tiempo a horas para mejorar condición numérica (opcional pero recomendado)
        # Si no querés cambiar la unidad en el gráfico, usá t_vals directamente, 
        # pero ajustá los beta0 en consecuencia. Aquí usamos minutos directos.
        
        resultados[cond] = {}
        
        for nombre, cfg in MODELOS_CINETICA.items():
            k = len(cfg['params'])
            if len(t_vals) < k + 2:
                print(f"⊘ {nombre:10s}: pocos puntos"); continue
            
            # Filtrar t>0 para Elovich
            if nombre == 'Elovich':
                mask = t_vals > 0
                t_fit, qt_fit, sy_fit = t_vals[mask], qt_vals[mask], sy[mask]
            else:
                t_fit, qt_fit, sy_fit = t_vals, qt_vals, sy
            
            # Estimaciones iniciales inteligentes
            if nombre == 'PSO':
                beta0 = estimar_p0_pso(t_fit, qt_fit)
            elif nombre == 'PFO':
                beta0 = estimar_p0_pfo(t_fit, qt_fit)
            elif nombre == 'Avrami':
                beta0 = [qt_fit.max() * 1.1, 0.01, 1.0]
            elif nombre == 'IPD':
                # kid ~ (qt_max - qt_min)/sqrt(t_max)
                beta0 = [(qt_fit.max() - qt_fit.min()) / np.sqrt(t_fit.max() + 1e-9), qt_fit.min()]
            else:
                beta0 = cfg['beta0']
            
            try:
                # curve_fit con sigma (pesos inversos a la varianza)
                # absolute_sigma=True para que los errores de parámetros sean correctos
                popt, pcov = curve_fit(
                    cfg['func'], t_fit, qt_fit, 
                    p0=beta0, 
                    sigma=sy_fit, 
                    absolute_sigma=True,
                    maxfev=10000
                )
                
                perr = np.sqrt(np.diag(pcov))
                yp = cfg['func'](t_fit, *popt)
                
                # Estadísticos
                r2, r2a, rmse, chi2r, aicc = calcular_estadisticos(qt_fit, yp, sy_fit, k)
                valido = valido_fisico_cinetica(nombre, popt)
                
                # Guardar resultados
                rd = dict(
                    params=dict(zip(cfg['params'], popt)),
                    params_err=dict(zip(cfg['params'], perr)),
                    R2=r2, R2_adj=r2a, RMSE=rmse, chi2_red=chi2r, AICc=aicc,
                    func=cfg['func'], popt=popt, n_points=len(t_fit),
                    valido=valido, success=True,
                    t_fit=t_fit, qt_fit=qt_fit, yp=yp, sy_fit=sy_fit
                )
                resultados[cond][nombre] = rd
                
                v = "✓" if (valido and r2 > 0.7) else "⚠"
                print(f"{v} {nombre:10s}: R²={r2:.4f}, χ²/dof={chi2r:.3f}, AICc={aicc:.2f}")
                
            except RuntimeError as e:
                print(f"✗ {nombre:10s}: no convergió ({str(e)[:50]})")
            except Exception as e:
                print(f"✗ {nombre:10s}: error ({str(e)[:50]})")
    
    return resultados

# ═════════════ 6. SELECCIÓN Y REPORTES ═════════════
def seleccionar_mejor_modelo(res_cin):
    """Selecciona el mejor modelo por AICc."""
    mejores = {}
    
    for cond, resd in res_cin.items():
        validos = {k: v for k, v in resd.items()
                   if v['success'] and v['valido'] and np.isfinite(v.get('AICc', np.nan))
                   and v['R2'] > 0.7}
        
        if not validos:
            print(f"❌ {cond}: sin modelos válidos"); continue
        
        orden = sorted(validos, key=lambda k: validos[k]['AICc'])
        amin = validos[orden[0]]['AICc']
        candidatos = [k for k in orden if validos[k]['AICc'] - amin < 2]
        mejor = min(candidatos, key=lambda k: len(validos[k]['popt']))
        
        mejores[cond] = mejor
        r = validos[mejor]
        print(f"\n {cond}: {mejor} (AICc={r['AICc']:.2f}, R²={r['R2']:.4f})")
    
    return mejores

def tabla_comparativa(res_cin):
    """Tabla comparativa de todos los modelos."""
    print(f"\n{'='*100}")
    print("📊 COMPARACIÓN DE MODELOS CINÉTICOS")
    print(f"{'='*100}")
    
    for cond, resd in res_cin.items():
        print(f"\n {cond}")
        print(f"{'Modelo':<12}{'R²':<9}{'R²_adj':<9}{'RMSE':<9}{'χ²/dof':<9}{'AICc':<10}{'ΔAICc':<9}{'ok'}")
        print("─" * 100)
        
        aics = [r['AICc'] for r in resd.values() if np.isfinite(r.get('AICc', np.nan))]
        if not aics: continue
        amin = min(aics)
        
        for nombre, r in sorted(resd.items(), key=lambda kv: kv[1].get('AICc', np.inf)):
            a = r.get('AICc', np.nan)
            d = a - amin if np.isfinite(a) else np.nan
            ok = "✓" if (r['success'] and r['valido'] and r['R2'] > 0.7) else ""
            print(f"{nombre:<12}{r['R2']:<9.4f}{r['R2_adj']:<9.4f}{r['RMSE']:<9.4f}"
                  f"{r['chi2_red']:<9.3f}{a:<10.2f}{d:<9.2f}{ok}")

def imprimir_parametros(res_cin, mejores):
    """Reporte de parámetros del modelo ganador."""
    print(f"\n{'='*100}")
    print("📋 PARÁMETROS CINÉTICOS (modelo ganador)")
    print(f"{'='*100}")
    
    for cond, mejor in mejores.items():
        r = res_cin[cond][mejor]
        n = r['n_points']
        k = len(r['popt'])
        tc = student_t.ppf(0.975, max(n - k - 1, 1))
        
        print(f"\n🔬 {cond} - {mejor}")
        print(f"  R² = {r['R2']:.4f}  |  R²_adj = {r['R2_adj']:.4f}")
        print(f"  RMSE = {r['RMSE']:.4f}  |  χ²/dof = {r['chi2_red']:.4f}")
        print(f"  AICc = {r['AICc']:.2f}")
        print(f"\n  Parámetros (IC 95% Wald):")
        for p in r['params']:
            v = r['params'][p]
            e = r['params_err'][p]
            lo, hi = v - tc*e, v + tc*e
            print(f"    {p:<8} = {v:.4g} ± {e:.4g}   [{lo:.4g}, {hi:.4g}]")

# ═════════════ 7. GRÁFICOS ═════════════
def graficar_cinetica_paper(df_stats, res_cin, mejores):
    """
    Gráfico principal tipo paper: solo el modelo ganador por condición,
    escala lineal de tiempo, datos con IC 95%.
    """
    condiciones_con_modelo = [c for c in sorted(res_cin.keys()) if c in mejores]
    
    if not condiciones_con_modelo:
        print("⚠ No hay modelos válidos para graficar")
        return
    
    n_cond = len(condiciones_con_modelo)
    fig, axes = plt.subplots(1, n_cond, figsize=(6*n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]
    
    colores = {'DI': '#1f77b4', 'DE': '#2ca02c', 'DM': '#d62728'}
    labels_cond = {'DI': 'DI (agua desionizada)', 
                   'DE': 'DE (NaCl, sin modificar)', 
                   'DM': 'DM (NaCl, modificado)'}
    
    for ax, cond in zip(axes, condiciones_con_modelo):
        sub = df_stats[df_stats['condicion'] == cond]
        color = colores.get(cond, 'gray')
        
        # Datos experimentales con IC 95%
        ax.errorbar(sub['t'], sub['qt_mean'], 
                    yerr=sub['qt_ci95'],
                    fmt='o', color=color, ms=8, capsize=4, 
                    label='Datos experimentales', zorder=5, alpha=0.9)
        
        # Modelo ganador (curve_fit no lineal)
        mejor = mejores[cond]
        r = res_cin[cond][mejor]
        t_curve = np.linspace(0, sub['t'].max() * 1.05, 500)
        y_curve = r['func'](t_curve, *r['popt'])
        
        # Construir etiqueta con parámetros
        if cond in bootstrap_results and mejor == 'Elovich':
            boot = bootstrap_results[cond]
            params_str = f"α={boot['alpha']['media']:.2f}, β={boot['beta']['media']:.2f}"
        else:
            params_str = ', '.join([f'{k}={v:.3g}' for k, v in r['params'].items()])
        ax.plot(t_curve, y_curve, '-', color=color, lw=2.5,
                label=f'{mejor}: R²={r["R2"]:.3f}\n({params_str})',
                alpha=0.85)
        ax.set_ylim(0, 3.0)
        ax.set_xlabel('Tiempo (min)', fontsize=12)
        ax.set_title(labels_cond.get(cond, cond), fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
        ax.grid(ls='--', alpha=0.4)
        
        # Eje X lineal con ticks en los tiempos experimentales
        ax.set_xticks(sorted(sub['t'].unique()))
        ax.set_xticklabels([f'{int(t)}' for t in sorted(sub['t'].unique())], 
                           rotation=45, ha='right', fontsize=9)
    
    axes[0].set_ylabel('$q_t$ (µg/mg)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cinetica_modelos_ganadores.png', dpi=300, bbox_inches='tight')
    print("📊 Guardado: cinetica_modelos_ganadores.png")
    plt.show()


def graficar_comparacion_DI_DM(df_stats, res_cin, mejores):
    """
    Gráfico comparativo DI vs DM en un solo panel (el más importante para el paper).
    Muestra cómo la matriz salina afecta la cinética.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colores = {'DI': '#1f77b4', 'DM': '#d62728'}
    markers = {'DI': 'o', 'DM': 's'}
    
    for cond in ['DI', 'DM']:
        if cond not in mejores:
            continue
        
        sub = df_stats[df_stats['condicion'] == cond]
        color = colores[cond]
        marker = markers[cond]
        
        # Datos
        ax.errorbar(sub['t'], sub['qt_mean'], 
                    yerr=sub['qt_ci95'],
                    fmt=marker, color=color, ms=8, capsize=4, 
                    label=f'{cond} datos', zorder=5, alpha=0.9)
        
        # Modelo ganador
        mejor = mejores[cond]
        r = res_cin[cond][mejor]
        t_curve = np.linspace(0, sub['t'].max() * 1.05, 500)
        y_curve = r['func'](t_curve, *r['popt'])
        ax.plot(t_curve, y_curve, '-', color=color, lw=2.5,
                label=f'{cond} – {mejor} (R²={r["R2"]:.3f})',
                alpha=0.85)
    
    ax.set_xlabel('Tiempo (min)', fontsize=13)
    ax.set_ylabel('$q_t$ (µg/mg)', fontsize=13)
    ax.set_title('Efecto de la matriz salina en la cinética de adsorción de As(V)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(ls='--', alpha=0.4)
    
    # Ticks en tiempos experimentales
    todos_tiempos = sorted(df_stats['t'].unique())
    ax.set_xticks(todos_tiempos)
    ax.set_xticklabels([f'{int(t)}' for t in todos_tiempos], rotation=45, ha='right')
    ax.set_ylim(0, 3.0)
    plt.tight_layout()
    plt.savefig('cinetica_DI_vs_DM.png', dpi=300, bbox_inches='tight')
    print(" Guardado: cinetica_DI_vs_DM.png")
    plt.show()


def graficar_DE_datos(df_stats):
    """
    Gráfico exclusivo de DE mostrando que NO hay adsorción significativa.
    Importante para justificar su descarte en el paper.
    """
    sub = df_stats[df_stats['condicion'] == 'DE']
    if len(sub) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(sub['t'], sub['qt_mean'], 
                yerr=sub['qt_ci95'],
                fmt='o', color='#2ca02c', ms=8, capsize=4, 
                label='DE (sin modificar)', zorder=5)
    
    # Línea de referencia en q=0
    ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
    # Línea en el promedio de DE
    ax.axhline(sub['qt_mean'].mean(), color='#2ca02c', ls='--', lw=1.5, 
               alpha=0.7, label=f'Promedio DE: {sub["qt_mean"].mean():.3f} µg/mg')
    
    ax.set_xlabel('Tiempo (min)', fontsize=12)
    ax.set_ylabel('$q_t$ (µg/mg)', fontsize=12)
    ax.set_title('Adsorbente sin modificar (DE): ausencia de adsorción', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(ls='--', alpha=0.4)
    ax.set_xticks(sorted(sub['t'].unique()))
    ax.set_xticklabels([f'{int(t)}' for t in sorted(sub['t'].unique())], 
                       rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('cinetica_DE_sin_adsorcion.png', dpi=300, bbox_inches='tight')
    print("📊 Guardado: cinetica_DE_sin_adsorcion.png")
    plt.show()

def validar_parametros_fisicos(nombre, params, params_err, n, k):
    """
    Validación menos estricta: solo descarta si el VALOR PUNTUAL es negativo.
    El IC amplio se reporta pero no descalifica el modelo.
    """
    for p, v in params.items():
        if v <= 0 and p in ['alpha', 'beta', 'qe', 'k1', 'k2', 'kAv', 'n', 'kid']:
            return False
    return True


def bootstrap_elovich(df_stats, cond, popt, B=1000, seed=42):
    """
    Bootstrap paramétrico para Elovich.
    Genera IC empíricos sin asumir normalidad.
    """
    from scipy.optimize import curve_fit
    
    sub = df_stats[df_stats['condicion'] == cond]
    mask = sub['t'].values > 0
    t = sub['t'].values[mask]
    qt = sub['qt_mean'].values[mask]
    sy = sub['sy'].values[mask]
    
    rng = np.random.default_rng(seed)
    muestras_alpha = []
    muestras_beta = []
    
    for _ in range(B):
        # Perturbar qt con su error experimental
        qt_b = qt + rng.normal(0, sy)
        try:
            popt_b, _ = curve_fit(
                elovich, t, qt_b, 
                p0=popt, 
                sigma=sy, 
                absolute_sigma=True, 
                maxfev=5000
            )
            # Solo aceptar si ambos parámetros son positivos
            if popt_b[0] > 0 and popt_b[1] > 0:
                muestras_alpha.append(popt_b[0])
                muestras_beta.append(popt_b[1])
        except:
            continue
    
    if len(muestras_alpha) < 100:
        print(f"⚠ Bootstrap para {cond}: solo {len(muestras_alpha)}/{B} convergencias")
        return None
    
    alpha_arr = np.array(muestras_alpha)
    beta_arr = np.array(muestras_beta)
    
    return {
        'alpha': {
            'valor': np.median(alpha_arr),
            'IC_95': (np.percentile(alpha_arr, 2.5), np.percentile(alpha_arr, 97.5)),
            'media': np.mean(alpha_arr),
            'std': np.std(alpha_arr)
        },
        'beta': {
            'valor': np.median(beta_arr),
            'IC_95': (np.percentile(beta_arr, 2.5), np.percentile(beta_arr, 97.5)),
            'media': np.mean(beta_arr),
            'std': np.std(beta_arr)
        },
        'tasa_convergencia': len(muestras_alpha) / B
    }


def seleccionar_mejor_modelo(res_cin):
    """Selecciona el mejor modelo por AICc, validando parámetros físicos."""
    mejores = {}
    
    for cond, resd in res_cin.items():
        # Filtrar modelos válidos (R² > 0.7, parámetros físicos correctos)
        validos = {}
        for k, v in resd.items():
            if not (v['success'] and v['valido'] and np.isfinite(v.get('AICc', np.nan)) and v['R2'] > 0.7):
                continue
            
            # Validar que los parámetros sean físicamente aceptables
            if validar_parametros_fisicos(k, v['params'], v['params_err'], v['n_points'], len(v['popt'])):
                validos[k] = v
        
        if not validos:
            print(f" {cond}: sin modelos válidos (parámetros no físicos)"); continue
        
        orden = sorted(validos, key=lambda k: validos[k]['AICc'])
        amin = validos[orden[0]]['AICc']
        candidatos = [k for k in orden if validos[k]['AICc'] - amin < 2]
        mejor = min(candidatos, key=lambda k: len(validos[k]['popt']))
        
        mejores[cond] = mejor
        r = validos[mejor]
        print(f"\n {cond}: {mejor} (AICc={r['AICc']:.2f}, R²={r['R2']:.4f})")
    
    return mejores
# ═════════════ 8. MAIN ═════════════
if __name__ == "__main__":
    CSV_PATH = "DatosCineticas1.csv"
    
    
    print("="*100)
    print("ANÁLISIS DE CINÉTICA DE ADSORCIÓN As(V)")
    print("="*100)
    
    df_raw = cargar_cinetica_real(CSV_PATH)
    df_stats = agrupar_cinetica(df_raw)
    reportar_estadisticas(df_stats)
    
    res_cin = ajustar_modelos_cinetica(df_stats)
    if not res_cin:
        raise SystemExit(" Sin ajustes convergidos")
    
    tabla_comparativa(res_cin)
    
    # NUEVO: selecciona modelos con parámetros físicamente válidos
    mejores = seleccionar_mejor_modelo(res_cin)
    
    # Bootstrap para los modelos ganadores
    print("\n" + "="*80)
    print("BOOTSTRAP DE PARÁMETROS (IC empíricos)")
    print("="*80)
    
    bootstrap_results = {}
    for cond, mejor in mejores.items():
        if mejor == 'Elovich':
            popt = res_cin[cond][mejor]['popt']
            boot = bootstrap_elovich(df_stats, cond, popt, B=1000)
            if boot:
                bootstrap_results[cond] = boot
                print(f"\n🔬 {cond} - Elovich (Bootstrap, B=1000):")
                print(f"  Tasa de convergencia: {boot['tasa_convergencia']:.1%}")
                print(f"  α = {boot['alpha']['media']:.3f} ± {boot['alpha']['std']:.3f}")
                print(f"     IC 95% bootstrap: [{boot['alpha']['IC_95'][0]:.3f}, {boot['alpha']['IC_95'][1]:.3f}]")
                print(f"  β = {boot['beta']['media']:.3f} ± {boot['beta']['std']:.3f}")
                print(f"     IC 95% bootstrap: [{boot['beta']['IC_95'][0]:.3f}, {boot['beta']['IC_95'][1]:.3f}]")
                
                # Validación física
                if boot['alpha']['IC_95'][0] <= 0:
                    print(f"  ⚠ El IC bootstrap de α incluye valores ≤ 0")
                    print(f"    → Elovich NO es físicamente válido para {cond}")
                else:
                    print(f"  ✓ El IC bootstrap de α es estrictamente positivo")
            else:
                print(f"\n⚠ {cond}: Bootstrap falló (pocas convergencias)")
    
    # Decisión final basada en bootstrap
    print("\n" + "="*80)
    print("DECISIÓN FINAL DE MODELOS")
    print("="*80)
    
    modelos_finales = {}
    for cond, mejor in mejores.items():
        if cond in bootstrap_results:
            boot = bootstrap_results[cond]
            if boot['alpha']['IC_95'][0] > 0:
                modelos_finales[cond] = mejor
                print(f"  {cond}: {mejor} ✓ (IC bootstrap válido)")
            else:
                # Buscar alternativa
                alternativas = [m for m in ['Avrami', 'PSO', 'IPD', 'PFO'] 
                               if m in res_cin[cond] and res_cin[cond][m]['success']]
                if alternativas:
                    modelos_finales[cond] = alternativas[0]
                    print(f"  {cond}: {mejor} ✗ (IC bootstrap inválido) → Usando {alternativas[0]}")
                else:
                    print(f"  {cond}: Sin modelos válidos")
        else:
            modelos_finales[cond] = mejor
            print(f"  {cond}: {mejor} (sin bootstrap)")

    graficar_cinetica_paper(df_stats, res_cin, modelos_finales)
    graficar_comparacion_DI_DM(df_stats, res_cin, modelos_finales)
