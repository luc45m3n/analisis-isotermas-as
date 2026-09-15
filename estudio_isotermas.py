#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Análisis de isotermas As(V): ODR con réplicas, acotado de exponentes y selección por AICc."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.stats import t as student_t

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 11, "axes.titlesize": 12})

UNIDAD_CE = "mg/L"     # ← ajustá si es µg/L
CS_BET    = 2000.0     # ← solo si reincorporás BET con valor físico
MOSTRAR   = True       # False para batch sin ventanas

# ═════════════ 1. MODELOS ═════════════
def langmuir(Ce, Qmax, KL):       return (Qmax*KL*Ce)/(1 + KL*Ce)
def freundlich(Ce, Kf, n):         return Kf*np.power(Ce, 1.0/n)
def sips(Ce, Qmax, KL, n):         return (Qmax*KL*np.power(Ce, n))/(1 + KL*np.power(Ce, n))
def langmuir_freundlich_hybrid(Ce, Qmax, KL, Kf, m):
    return (Qmax*KL*Ce)/(1 + KL*Ce) + Kf*np.power(Ce, 1/m)

def khan(Ce, Qmax, KL, m):         return (Qmax*KL*Ce)/np.power(1 + KL*Ce, m)
def halsey(Ce, Kh, m):             return (1/m)*np.log(Kh) - (1/m)*np.log(Ce)
def double_langmuir_ord(Ce, Q1, K1, Q2, t):
    """Dos sitios con K2 = K1·r (r ∈ (0,1)): rompe la simetría de intercambio."""
    K2 = K1*acotar(t, 0.0, 1.0)
    return (Q1*K1*Ce)/(1 + K1*Ce) + (Q2*K2*Ce)/(1 + K2*Ce)

# --- Redlich–Peterson EXACTA de la fuente bibliográfica, β acotado a (0,1) ---
def acotar(t, a, b):    return a + (b - a)/(1.0 + np.exp(-t))
def desacotar(p, a, b): return np.log((p - a)/(b - p))
def redlich_peterson(Ce, KRP, aRP, t):
    beta = acotar(t, 0.0, 1.0)
    return (KRP*Ce)/(1.0 + aRP*np.power(Ce, beta))

def langmuir_odr(b, x):            return langmuir(x, *b)
def freundlich_odr(b, x):          return freundlich(x, *b)
def sips_odr(b, x):                return sips(x, *b)
def lf_hybrid_odr(b, x):           return langmuir_freundlich_hybrid(x, *b)
def khan_odr(b, x):                return khan(x, *b)
def halsey_odr(b, x):              return halsey(x, *b)
def double_langmuir_ord_odr(b, x): return double_langmuir_ord(x, *b)
def redlich_peterson_odr(b, x):    return redlich_peterson(x, *b)

MODELOS = {
 'Langmuir':          dict(func=langmuir,        odr=langmuir_odr,        beta0=[80, 0.01],            params=['Qmax', 'KL']),
 'Freundlich':        dict(func=freundlich,      odr=freundlich_odr,      beta0=[10, 2.0],             params=['Kf', 'n']),
 'Sips':              dict(func=sips,            odr=sips_odr,            beta0=[80, 0.01, 1.0],       params=['Qmax', 'KL', 'n']),
 'Double_Langmuir_ord': dict(func=double_langmuir_ord, odr=double_langmuir_ord_odr,
                             beta0=[40, 0.01, 25, desacotar(0.5, 0, 1)],   params=['Q1', 'K1', 'Q2', 't'],
                             acotado=('t', 0.0, 1.0, 'r')),
 'LF_hybrid':         dict(func=langmuir_freundlich_hybrid, odr=lf_hybrid_odr, beta0=[40, 0.01, 5, 0.5], params=['Qmax', 'KL', 'Kf', 'm']),
 'Khan':              dict(func=khan,            odr=khan_odr,            beta0=[80, 0.01, 1.0],       params=['Qmax', 'KL', 'm']),
 'halsey':            dict(func=halsey,          odr=halsey_odr,          beta0=[0.01, 1.0],           params=['Kh', 'm']),
 'Redlich_Peterson':  dict(func=redlich_peterson, odr=redlich_peterson_odr,
                             beta0=[10, 0.01, desacotar(0.5, 0, 1)],       params=['K_RP', 'a_RP', 't'],
                             acotado=('t', 0.0, 1.0)),
}

# ═════════════ 1b. COEFICIENTE DE VARIACIÓN ═════════════
def calcular_cv(valores):
    """Calcula CV (%) ignorando NaN. Si solo hay 1 valor, retorna 0."""
    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]
    if len(valores) < 2:
        return 0.0
    mu = np.mean(valores)
    if mu == 0:
        return np.nan
    return (np.std(valores, ddof=1) / mu) * 100.0

def interpretar_cv(cv):
    """Clasifica el CV según criterios analíticos comunes."""
    if np.isnan(cv): return "N/A"
    if cv < 5:    return "Excelente"
    if cv < 10:   return "Bueno"
    if cv < 20:   return "Aceptable"
    if cv < 30:   return "Alto"
    return "Muy alto"
# Fuera de la batería (sin Cs físico / inestable): bet, Fritz_Schlunder.
# Reincorporar solo con Cs real y forma (Qm*Kf*Ce)/(1+Kf*Ce**m), respectivamente.

def valido_fisico(nombre, p):
    if not np.all(np.isfinite(p)): return False
    if nombre == 'Double_Langmuir_ord': return p[0] > 0 and p[1] > 0 and p[2] > 0
    if nombre == 'halsey':  return p[1] > 0
    return p[0] > 0 and p[1] > 0

def aceptable(r):
    return (r['success'] and r['valido'] and r['R2'] > 0.5
            and np.isfinite(r['chi2_red']) and r['chi2_red'] < 20)

def bootstrap_parametros(df_cond, cfg, popt, B=500, seed=42, parametrico=True):
    """IC bootstrap de los parámetros de un modelo ya ajustado."""
    rng  = np.random.default_rng(seed)
    Ce   = df_cond.Ce_raw.values
    Qe   = df_cond.Qe_raw.values
    sCe  = np.maximum(df_cond.Ce_std.values, 0.01*np.abs(Ce) + 1e-6)
    sQe  = np.maximum(df_cond.Qe_std.values, 0.05*np.abs(Qe) + 1e-6)
    yp   = cfg['func'](Ce, *popt)                 # predicción del ajuste original
    grupos = np.unique(df_cond['grupo'].values)

    muestras = []
    for _ in range(B):
        if parametrico:
            # errores tal como los asume ODR; opcionalmente perturbá Ce también:
            Ce_b = Ce                            # o: Ce + rng.normal(0.0, sCe)
            Qe_b = yp + rng.normal(0.0, sQe)
            sCe_b, sQe_b = sCe, sQe
        else:
            # cluster bootstrap: grupos (niveles de Ce) remuestreados con reposición
            gb  = rng.choice(grupos, size=len(grupos), replace=True)
            idx = np.concatenate([np.where(df_cond['grupo'].values == g)[0] for g in gb])
            Ce_b, Qe_b   = Ce[idx], Qe[idx]
            sCe_b, sQe_b = sCe[idx], sQe[idx]
        try:
            r = ODR(RealData(Ce_b, Qe_b, sx=sCe_b, sy=sQe_b),
                    Model(cfg['odr']), beta0=popt, maxit=3000).run()
            if r.info in (1, 2) and np.all(np.isfinite(r.beta)):
                muestras.append(r.beta)
        except Exception:
            continue

    arr = np.asarray(muestras)
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    return arr, lo, hi, len(arr)/B                # arr (B_ok, k), IC, tasa de convergencia

def agregar_bootstrap(res_por_cond, df_ajuste, mejores, B=500):
    for cond, resd in res_por_cond.items():
        if cond not in mejores: continue
        nombre = mejores[cond]                    # solo el ganador (ahorrás tiempo)
        r = resd[nombre]
        if not r['success']: continue
        cfg = MODELOS[nombre]
        sub = df_ajuste[df_ajuste.condicion == cond]
        arr, lo, hi, tasa = bootstrap_parametros(sub, cfg, r['popt'], B=B)
        r['boot_ci']   = dict(zip(cfg['params'], zip(lo, hi)))
        r['boot_tasa'] = tasa
        # cantidades derivadas, sin aproximación lineal:
        if 'm' in cfg['params']:
            inv = 1.0/arr[:, cfg['params'].index('m')]
            r['boot_ci']['1/m'] = (np.percentile(inv, 2.5), np.percentile(inv, 97.5))
        print(f"🔁 {cond} / {nombre}: {tasa:.0%} convergencia bootstrap")

# ═════════════ 2. CARGA Y AGRUPACIÓN ═════════════
def cargar_y_agrupar_replicas(csv_path, tolerancia_co=0.01):
    """
    Carga datos con estructura:
    - 3 condiciones (DI, NaCl, Natural)
    - 7 réplicas (falcones) por condición
    - 3 mediciones FIASS por réplica
    Total: 63 filas por condición × 3 condiciones = 189 filas (o 63 si solo hay 1 condición)
    """
    df = pd.read_csv(csv_path, sep=";", decimal=".",comment='#')
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={'simga_co': 'sigma_co', 'simga_ce': 'sigma_ce', 'simga_qe': 'sigma_qe',
                       'ce': 'Ce', 'qe': 'Qe', 'co': 'Co'}, inplace=True)
    
    # Detectar columna de condición
    if 'condicion' in df.columns:
        pass  # ya existe
    elif 'ph' in df.columns:
        df['condicion'] = df['ph'].apply(lambda x: f'pH_{x}')
    elif 'matriz' in df.columns:
        df['condicion'] = df['matriz']
    else:
        # Si no hay columna de condición, intentar inferirla del rango de filas
        # Asumir que cada 21 filas es una condición diferente
        n_filas = len(df)
        if n_filas == 63:
            df['condicion'] = ['DI']*21 + ['NaCl']*21 + ['Natural']*21
        elif n_filas == 21:
            df['condicion'] = ['condicion_1']*21
        else:
            df['condicion'] = 'condicion_1'
        print(f"⚠ Columna de condición no encontrada. Asignada automáticamente: {df['condicion'].unique()}")
    
    # Verificar columnas críticas
    if 'replica' not in df.columns:
        raise ValueError("❌ Columna 'replica' no encontrada")
    
    # Corregir medicion_fiass: debe ser 1, 2, 3 dentro de cada réplica
    # No un contador global
    df = df.sort_values(['condicion', 'replica', 'medicion_fiass']).reset_index(drop=True)
    
    # Recalcular medicion_fiass correctamente
    df['medicion_fiass_correcta'] = df.groupby(['condicion', 'replica']).cumcount() + 1
    
    # Filtrar valores positivos
    df = df[(df['Ce'] > 0) & (df['Qe'] > 0)].copy()
    
    print(f"\n📊 Datos cargados: {len(df)} mediciones FIASS totales")
    print(f"   Condiciones: {df['condicion'].unique()}")
    print(f"   Réplicas por condición: {df.groupby('condicion')['replica'].nunique().to_dict()}")
    print(f"   Mediciones FIASS por réplica: {df.groupby(['condicion', 'replica'])['medicion_fiass_correcta'].nunique().max()}")
    
    # Usar medicion_fiass_correcta en lugar de la original
    df['medicion_fiass'] = df['medicion_fiass_correcta']
    
    # ═══════ PASO 1: Promediar mediciones FIASS dentro de cada réplica ═══════
    df_replica = df.groupby(['condicion', 'Co', 'replica']).agg(
        Ce=('Ce', 'mean'),
        Qe=('Qe', 'mean'),
        Ce_cv_fiass=('Ce', lambda x: (np.std(x, ddof=1) / np.mean(x) * 100) if len(x) > 1 else 0),
        Qe_cv_fiass=('Qe', lambda x: (np.std(x, ddof=1) / np.mean(x) * 100) if len(x) > 1 else 0),
        Ce_n=('Ce', 'count'),
        Qe_n=('Qe', 'count'),
    ).reset_index()
    
    n_fiass = df.groupby(['condicion', 'replica'])['medicion_fiass'].nunique().max()
    print(f"\n✓ Promedio de {n_fiass} mediciones FIASS → {len(df_replica)} valores por réplica")
    
    # ═══════ PASO 2: Agrupar réplicas por concentración (tolerancia 1%) ═══════
    grupos = []
    for cond, sub in df_replica.groupby('condicion'):
        sub = sub.sort_values('Co').reset_index(drop=True)
        gid, co0 = 0, None
        for idx, row in sub.iterrows():
            if co0 is None or abs(row['Co'] - co0)/co0 > tolerancia_co:
                gid += 1
                co0 = row['Co']
            sub.at[idx, 'grupo'] = gid
        grupos.append(sub)
    
    df_agr = pd.concat(grupos, ignore_index=True)
    
    # ═══════ PASO 3: Estadísticos entre réplicas ═══════
    st = df_agr.groupby(['condicion', 'grupo']).agg(
        Co=('Co', 'mean'),
        Ce_mean=('Ce', 'mean'),
        Ce_std=('Ce', 'std'),
        Ce_cv=('Ce', lambda x: (np.std(x, ddof=1) / np.mean(x) * 100) if len(x) > 1 else 0),
        Ce_count=('Ce', 'count'),
        Qe_mean=('Qe', 'mean'),
        Qe_std=('Qe', 'std'),
        Qe_cv=('Qe', lambda x: (np.std(x, ddof=1) / np.mean(x) * 100) if len(x) > 1 else 0),
        Qe_count=('Qe', 'count'),
        Ce_cv_fiass_mean=('Ce_cv_fiass', 'mean'),
        Qe_cv_fiass_mean=('Qe_cv_fiass', 'mean'),
    ).reset_index()
    
    # Manejar casos con 1 sola réplica
    st.loc[st.Ce_count == 1, 'Ce_std'] = st.loc[st.Ce_count == 1, 'Ce_mean'] * 0.01
    st.loc[st.Ce_count == 1, 'Qe_std'] = st.loc[st.Ce_count == 1, 'Qe_mean'] * 0.05
    st.loc[st.Ce_count == 1, 'Ce_cv'] = 1.0
    st.loc[st.Ce_count == 1, 'Qe_cv'] = 5.0
    
    # Preparar para ODR
    df_ajuste = df_agr[['condicion', 'grupo', 'Co', 'Ce', 'Qe']].merge(
        st[['condicion', 'grupo', 'Ce_std', 'Qe_std']], on=['condicion', 'grupo'])
    df_ajuste.rename(columns={'Ce': 'Ce_raw', 'Qe': 'Qe_raw'}, inplace=True)
    
    # ═══════ REPORTE ═══════
    print(f"\n{'='*90}")
    print("📈 ANÁLISIS DE COEFICIENTES DE VARIACIÓN")
    print(f"{'='*90}")
    
    for cond in sorted(st.condicion.unique()):
        sub = st[st.condicion == cond]
        print(f"\n🔬 {cond}:")
        print(f"  {'Co':>8} │ {'Ce_mean':>10} {'CV_Ce(%)':>10} {'CV_FIASS':>10} │ "
              f"{'Qe_mean':>10} {'CV_Qe(%)':>10} {'CV_FIASS':>10}")
        print("  " + "─" * 85)
        
        for _, row in sub.iterrows():
            print(f"  {row['Co']:>8.4f} │ {row['Ce_mean']:>10.4f} "
                  f"{row['Ce_cv']:>9.2f}% [{interpretar_cv(row['Ce_cv']):>9}] "
                  f"{row['Ce_cv_fiass_mean']:>9.2f}% │ "
                  f"{row['Qe_mean']:>10.4f} "
                  f"{row['Qe_cv']:>9.2f}% [{interpretar_cv(row['Qe_cv']):>9}] "
                  f"{row['Qe_cv_fiass_mean']:>9.2f}%")
    
    # Resumen global
    cv_fiass_prom = st[['Ce_cv_fiass_mean', 'Qe_cv_fiass_mean']].mean()
    cv_exp_prom = st[['Ce_cv', 'Qe_cv']].mean()
    
    print(f"\n{'='*90}")
    print("📊 RESUMEN GLOBAL")
    print(f"{'='*90}")
    print(f"  CV intra-réplica (FIASS) promedio:")
    print(f"    Ce: {cv_fiass_prom['Ce_cv_fiass_mean']:.2f}%  |  Qe: {cv_fiass_prom['Qe_cv_fiass_mean']:.2f}%")
    print(f"  CV inter-réplica (experimental) promedio:")
    print(f"    Ce: {cv_exp_prom['Ce_cv']:.2f}%  |  Qe: {cv_exp_prom['Qe_cv']:.2f}%")
    print(f"\n✅ {len(st)} puntos únicos; {len(df_ajuste)} réplicas para ajuste ODR")
    print(f"   Distribución: {df_ajuste.condicion.value_counts().to_dict()}")
    
    return st, df_ajuste

def interpretar_cv(cv):
    """Clasifica el CV según criterios analíticos comunes."""
    if np.isnan(cv): return "N/A"
    if cv < 5:    return "Excelente"
    if cv < 10:   return "Bueno"
    if cv < 20:   return "Aceptable"
    if cv < 30:   return "Alto"
    return "Muy alto"
# ═════════════ 3. ESTADÍSTICOS ═════════════
def calcular_estadisticos(y, yp, sy, k):
    n, dof = len(y), len(y) - k
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - y.mean())**2)
    chi2   = np.sum(((y - yp)/sy)**2)
    r2     = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    r2a    = 1 - (ss_res/dof)/(ss_tot/(n-1)) if dof > 0 else np.nan
    rmse   = np.sqrt(ss_res/n)
    chi2r  = chi2/dof if dof > 0 else np.nan
    aic    = 2*k + chi2
    aicc   = aic + 2*k*(k+1)/(n-k-1) if n > k+1 else np.nan
    return r2, r2a, rmse, chi2r, aicc

# ═════════════ 4. AJUSTE ODR POR CONDICIÓN ═════════════
def ajustar_modelos_por_condicion(df_ajuste, seed=2026):
    rng = np.random.default_rng(seed)
    out_por_cond = {}
    for cond, sub in df_ajuste.groupby('condicion'):
        Ce, Qe   = sub.Ce_raw.values, sub.Qe_raw.values
        sCe_safe = np.maximum(sub.Ce_std.values, 0.01*np.abs(Ce) + 1e-6)
        sQe_safe = np.maximum(sub.Qe_std.values, 0.05*np.abs(Qe) + 1e-6)
        data = RealData(Ce, Qe, sx=sCe_safe, sy=sQe_safe)
        print(f"\n📐 {cond}: {len(Ce)} réplicas")
        resultados = {}
        for nombre, cfg in MODELOS.items():
            k = len(cfg['params'])
            if len(Ce) < k + 2:
                print(f"⊘ {nombre:20s}: pocos puntos"); continue
            beta0 = cfg['beta0'].copy()
            if 'Qmax' in cfg['params']:
                beta0[cfg['params'].index('Qmax')] = 1.2*Qe.max()
            if 'Q1' in cfg['params']:
                p = cfg['params']; cm = np.median(Ce) or 1.0
                beta0[p.index('Q1')] = Qe.max();  beta0[p.index('K1')] = 2.0/cm
                beta0[p.index('Q2')] = 0.5*Qe.max(); beta0[p.index('t')] = 0.0
           
            res_ok = None
            for intento in range(5):
                b0 = beta0 if intento == 0 else \
                     [b + rng.uniform(-.5, .5)*max(abs(b), 1e-3) for b in beta0]
                try:
                    r = ODR(data, Model(cfg['odr']), beta0=b0, maxit=10000).run()
                    if np.all(np.isfinite(r.beta)) and r.info in (1, 2):
                        res_ok = r; break
                except Exception:
                    continue
            if res_ok is None:
                print(f"✗ {nombre:20s}: no convergió"); continue

            popt, perr = res_ok.beta, res_ok.sd_beta
            yp = cfg['func'](Ce, *popt)
            r2, r2a, rmse, chi2r, aicc = calcular_estadisticos(Qe, yp, sQe_safe, k)
            rd = dict(params=dict(zip(cfg['params'], popt)),
                      params_err=dict(zip(cfg['params'], perr)),
                      R2=r2, R2_adj=r2a, RMSE=rmse, chi2_red=chi2r, AICc=aicc,
                      func=cfg['func'], popt=popt, n_points=len(Ce),
                      valido=valido_fisico(nombre, popt),
                      success=np.all(np.isfinite(popt)) and np.all(np.isfinite(perr)))

            if 'acotado' in cfg:
                pn, a, b = cfg['acotado'][:3]
                rep = cfg['acotado'][3] if len(cfg['acotado']) > 3 else 'beta'
                i = cfg['params'].index(pn)
                tv, sd = popt[i], perr[i]; bv = acotar(tv, a, b)
                tc = student_t.ppf(0.975, max(len(Ce)-k, 1))
                for d in (rd['params'], rd['params_err']): d.pop(pn)
                rd['params'][rep]     = bv
                rd['params_err'][rep] = sd*bv*(1-bv)
                rd['ci_map'] = {rep: (acotar(tv-tc*sd, a, b), acotar(tv+tc*sd, a, b))}
                rd['report_names'] = [n if n != pn else rep for n in cfg['params']]
                rd['report_vals']  = [bv if n == pn else v for n, v in zip(cfg['params'], popt)]
            else:
                rd['ci_map'] = {}
                rd['report_names'] = list(cfg['params'])
                rd['report_vals']  = list(popt)
            if nombre == 'Double_Langmuir_ord':
                rd['report_names'].append('K2')
                rd['report_vals'].append(rd['report_vals'][1]*rd['params']['r'])

            resultados[nombre] = rd
            v = "✓" if aceptable(rd) else "⚠"
            print(f"{v} {nombre:20s}: R²={r2:.4f}, χ²/dof={chi2r:.3f}, AICc={aicc:.2f}")
        out_por_cond[cond] = resultados
    return out_por_cond

# ═════════════ 5. REPORTES ═════════════
def imprimir_parametros_ajustados(res_por_cond):
    for cond, resd in res_por_cond.items():
        print(f"\n{'='*80}\nPARÁMETROS - {cond}\n{'='*80}")
        for nombre, r in resd.items():
            if not r['success']: continue
            tc = student_t.ppf(0.975, max(r['n_points']-len(r['popt'])-1, 1))
            print(f"\n {nombre} {'(aceptable)' if aceptable(r) else '(NO aceptable)'}")
            for p, v in r['params'].items():
                if p in r.get('ci_map', {}):
                    lo, hi = r['ci_map'][p]
                    print(f"  {p:<10}: {v:.4g}  [Wald IC 95%: {lo:.4g}, {hi:.4g}]")
                else:
                    e = r['params_err'][p]
                    print(f"  {p:<10}: {v:.4g} ± {e:.4g}  [Wald IC 95%: {v-tc*e:.4g}, {v+tc*e:.4g}]")
                if p in r.get('boot_ci', {}):
                    lo, hi = r['boot_ci'][p]
                    print(f"  {'':<10}  [boot IC 95%: {lo:.4g}, {hi:.4g}]")
            # cantidad derivada, invariante al parametrizado:
            if '1/m' in r.get('boot_ci', {}):
                lo, hi = r['boot_ci']['1/m']
                print(f"  {'1/m':<10}: {1.0/r['params']['m']:.4g}  [boot IC 95%: {lo:.4g}, {hi:.4g}]")

def imprimir_tabla_por_condicion(res_por_cond):
    for cond, resd in res_por_cond.items():
        print(f"\n📊 {cond}")
        print(f"{'Modelo':<20}{'R²':<9}{'R²_adj':<9}{'RMSE':<9}{'χ²/dof':<9}{'AICc':<10}{'ΔAICc':<9}{'ok'}")
        vals = [r['AICc'] for r in resd.values() if np.isfinite(r.get('AICc', np.nan))]
        if not vals: print("❌ nada convergió"); continue
        amin = min(vals)
        for nombre, r in sorted(resd.items(), key=lambda kv: kv[1].get('AICc', np.inf)):
            a = r.get('AICc', np.nan)
            d = a - amin if np.isfinite(a) else np.nan
            print(f"{nombre:<20}{r['R2']:<9.4f}{r['R2_adj']:<9.4f}{r['RMSE']:<9.4f}"
                  f"{r['chi2_red']:<9.3f}{a:<10.2f}{d:<9.2f}{'✓' if aceptable(r) else '✗'}")

def seleccionar_mejores_modelos(res_por_cond, modelo_fijo=None):
    """
    Selecciona el mejor modelo para la primera condición (DI) y lo fuerza para las demás.
    
    Args:
        res_por_cond: diccionario con resultados por condición
        modelo_fijo: nombre del modelo a usar para todas las condiciones. 
                     Si es None, selecciona automáticamente el mejor para DI.
    """
    mejores = {}
    
    # Ordenar condiciones: DI primero, luego NaCl, luego Natural
    cond_ordenadas = sorted(res_por_cond.keys(), 
                           key=lambda x: (0 if 'DI' in x or 'di' in x else 
                                         1 if 'NaCl' in x or 'nacl' in x else 2))
    
    for i, cond in enumerate(cond_ordenadas):
        resd = res_por_cond[cond]
        
        if i == 0 or modelo_fijo is None:
            # Para DI (o si no hay modelo fijo), seleccionar el mejor
            val = {k: v for k, v in resd.items() 
                   if v['success'] and v['valido'] and np.isfinite(v.get('AICc', np.nan))}
            
            if not val:
                print(f"❌ {cond}: sin modelos válidos")
                continue
            
            # Ordenar por AICc
            orden = sorted(val, key=lambda k: val[k]['AICc'])
            amin = val[orden[0]]['AICc']
            
            # Seleccionar el mejor (AICc más bajo, preferir modelos más simples si ΔAICc < 2)
            candidatos = [k for k in orden if val[k]['AICc'] - amin < 2]
            mejor = min(candidatos, key=lambda k: len(val[k]['popt']))
            
            if i == 0:
                modelo_fijo = mejor  # Guardar para usar en las otras condiciones
                print(f"\n Modelo seleccionado para todas las condiciones: {mejor}")
                print(f"   (AICc={val[mejor]['AICc']:.2f}, χ²/dof={val[mejor]['chi2_red']:.3f}, R²={val[mejor]['R2']:.4f})")
        else:
            # Para las otras condiciones, usar el modelo fijo
            mejor = modelo_fijo
        
        if mejor not in resd or not resd[mejor]['success']:
            print(f" {cond}: {mejor} no está disponible o no convergió")
            # Buscar alternativa
            val = {k: v for k, v in resd.items() 
                   if v['success'] and v['valido'] and np.isfinite(v.get('AICc', np.nan))}
            if val:
                mejor = min(val, key=lambda k: val[k]['AICc'])
                print(f"   → Usando alternativa: {mejor}")
            else:
                print(f"   ❌ Sin modelos disponibles para {cond}")
                continue
        
        mejores[cond] = mejor
        r = resd[mejor]
        print(f"   {cond}: {mejor} (AICc={r['AICc']:.2f}, χ²/dof={r['chi2_red']:.3f}, R²={r['R2']:.4f})")
    
    return mejores

# ═════════════ 6. GRÁFICOS ═════════════
def guardar_fig(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"📊 Guardado: {fname}")
    if MOSTRAR: plt.show()
    plt.close('all')

def trazar_datos(ax, df_stats, colors, markers):
    for i, cond in enumerate(sorted(df_stats.condicion.unique())):
        s = df_stats[df_stats.condicion == cond]
        ax.errorbar(s.Ce_mean, s.Qe_mean, xerr=s.Ce_std, yerr=s.Qe_std,
                    fmt=markers[i % 4], ms=7, capsize=4, color=colors[i],
                    label=f'{cond} (datos)', zorder=5, alpha=0.8)

def graficar_datos(df_stats):
    fig, ax = plt.subplots(figsize=(10, 7))
    trazar_datos(ax, df_stats, plt.cm.tab10(np.linspace(0, 1, 4)), ['o', 's', '^', 'D'])
    ax.set_xlabel(f'$C_e$ ({UNIDAD_CE})'); ax.set_ylabel('$q_e$ (mg/g)')
    ax.legend(); ax.grid(ls='--', alpha=.5)
    guardar_fig('isotermas_desvio_real.png')

def graficar_ajustes_por_condicion(df_stats, res_por_cond, mejores):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    trazar_datos(ax, df_stats, colors, ['o', 's', '^', 'D'])
    Cs = np.linspace(df_stats.Ce_mean.min()*0.95, df_stats.Ce_mean.max()*1.05, 300)
    for i, cond in enumerate(sorted(df_stats.condicion.unique())):
        if cond not in mejores: continue
        r = res_por_cond[cond][mejores[cond]]
        ax.plot(Cs, r['func'](Cs, *r['popt']), color=colors[i], lw=2.5,
                label=f'{cond} – {mejores[cond]}')
    ax.set_xlabel(f'$C_e$ ({UNIDAD_CE})'); ax.set_ylabel('$q_e$ (mg/g)')
    ax.legend(fontsize=9, ncol=2); ax.grid(ls='--', alpha=.5)
    guardar_fig('ajustes_desvio_real.png')

def graficar_modelo(df_stats, res_por_cond, modelo):
    for cond, resd in res_por_cond.items():
        if modelo not in resd or not resd[modelo]['success']: continue
        r = resd[modelo]
        fig, ax = plt.subplots(figsize=(10, 7))
        trazar_datos(ax, df_stats, ['gray']*4, ['o', 's', '^', 'D'])
        Cs = np.linspace(0, df_stats.Ce_mean.max()*1.1, 300)
        lbl = ', '.join(f'${n}$={v:.3g}' for n, v in zip(r['report_names'], r['report_vals']))
        ax.plot(Cs, r['func'](Cs, *r['popt']), 'r-', lw=2.5, label=f'{modelo}: {lbl}')
        ax.set_xlabel(f'$C_e$ ({UNIDAD_CE})'); ax.set_ylabel('$q_e$ (mg/g)')
        ax.legend(); ax.grid(ls='--', alpha=.5)
        guardar_fig(f'{modelo.lower()}_{cond.lower()}.png')

def graficar_descomposicion(df_stats, res_por_cond, mejores):
    """Descompone el ganador en sus dos términos (DL_ord o LF_hybrid)."""
    for cond, mej in mejores.items():
        if mej not in ('Double_Langmuir_ord', 'LF_hybrid'): continue
        p = res_por_cond[cond][mej]['popt']
        Cs = np.linspace(0, df_stats.Ce_mean.max()*1.1, 300)
        if mej == 'Double_Langmuir_ord':
            Q1, K1, Q2, t = p; K2 = K1*acotar(t, 0, 1)
            s1, s2 = (Q1*K1*Cs)/(1+K1*Cs), (Q2*K2*Cs)/(1+K2*Cs)
            l1, l2 = f'sitio 1 (Q={Q1:.1f}, K={K1:.3f})', f'sitio 2 (Q={Q2:.1f}, K={K2:.4f})'
        else:
            Qmax, KL, Kf, m = p
            s1, s2 = (Qmax*KL*Cs)/(1+KL*Cs), Kf*np.power(Cs, m)
            l1, l2 = f'monocapa (Q={Qmax:.1f}, K={KL:.3f})', f'multicapa (Kf={Kf:.2f}, m={m:.2f})'
        fig, ax = plt.subplots(figsize=(10, 7))
        trazar_datos(ax, df_stats, ['gray']*4, ['o', 's', '^', 'D'])
        ax.plot(Cs, s1+s2, 'r-', lw=2.5, label='total')
        ax.plot(Cs, s1, 'b--', lw=1.5, label=l1)
        ax.plot(Cs, s2, 'g--', lw=1.5, label=l2)
        ax.set_xlabel(f'$C_e$ ({UNIDAD_CE})'); ax.set_ylabel('$q_e$ (mg/g)')
        ax.set_title(f'{cond} – descomposición {mej}')
        ax.legend(); ax.grid(ls='--', alpha=.5)
        guardar_fig(f'descomposicion_{cond.lower()}.png')

def graficar_cv(df_stats):
    """Grafica CV intra-réplica (FIASS) e inter-réplica (experimental)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    markers = ['o', 's', '^', 'D']
    
    # Panel 1: CV inter-réplica de Ce
    ax = axes[0, 0]
    for i, cond in enumerate(sorted(df_stats.condicion.unique())):
        s = df_stats[df_stats.condicion == cond]
        ax.plot(s.Co, s.Ce_cv, markers[i % 4], color=colors[i],
                ms=8, label=cond, alpha=0.8)
    ax.axhline(5,  color='green',  ls='--', alpha=0.5, label='Excelente (5%)')
    ax.axhline(10, color='orange', ls='--', alpha=0.5, label='Bueno (10%)')
    ax.axhline(20, color='red',    ls='--', alpha=0.5, label='Aceptable (20%)')
    ax.set_xlabel('$C_0$ (mg/L)')
    ax.set_ylabel('CV inter-réplica (%)')
    ax.set_title('Variabilidad entre falcones (Ce)')
    ax.set_yscale('log')
    ax.grid(ls='--', alpha=0.4)
    ax.legend(fontsize=8)
    
    # Panel 2: CV inter-réplica de Qe
    ax = axes[0, 1]
    for i, cond in enumerate(sorted(df_stats.condicion.unique())):
        s = df_stats[df_stats.condicion == cond]
        ax.plot(s.Co, s.Qe_cv, markers[i % 4], color=colors[i],
                ms=8, label=cond, alpha=0.8)
    ax.axhline(5,  color='green',  ls='--', alpha=0.5)
    ax.axhline(10, color='orange', ls='--', alpha=0.5)
    ax.axhline(20, color='red',    ls='--', alpha=0.5)
    ax.set_xlabel('$C_0$ (mg/L)')
    ax.set_ylabel('CV inter-réplica (%)')
    ax.set_title('Variabilidad entre falcones (Qe)')
    ax.set_yscale('log')
    ax.grid(ls='--', alpha=0.4)
    
    # Panel 3: CV intra-réplica (FIASS) de Ce
    ax = axes[1, 0]
    for i, cond in enumerate(sorted(df_stats.condicion.unique())):
        s = df_stats[df_stats.condicion == cond]
        ax.plot(s.Co, s.Ce_cv_fiass_mean, markers[i % 4], color=colors[i],
                ms=8, label=cond, alpha=0.8)
    ax.axhline(5,  color='green',  ls='--', alpha=0.5)
    ax.set_xlabel('$C_0$ (mg/L)')
    ax.set_ylabel('CV intra-réplica FIASS (%)')
    ax.set_title('Precisión analítica FIASS (Ce)')
    ax.set_yscale('log')
    ax.grid(ls='--', alpha=0.4)
    
    # Panel 4: CV intra-réplica (FIASS) de Qe
    ax = axes[1, 1]
    for i, cond in enumerate(sorted(df_stats.condicion.unique())):
        s = df_stats[df_stats.condicion == cond]
        ax.plot(s.Co, s.Qe_cv_fiass_mean, markers[i % 4], color=colors[i],
                ms=8, label=cond, alpha=0.8)
    ax.axhline(5,  color='green',  ls='--', alpha=0.5)
    ax.set_xlabel('$C_0$ (mg/L)')
    ax.set_ylabel('CV intra-réplica FIASS (%)')
    ax.set_title('Precisión analítica FIASS (Qe)')
    ax.set_yscale('log')
    ax.grid(ls='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('analisis_cv_completo.png', dpi=300, bbox_inches='tight')
    print("📊 Guardado: analisis_cv_completo.png")
    if MOSTRAR: plt.show()
    plt.close('all')

def comparar_parametros_entre_condiciones(res_por_cond, mejores):
    """Compara los parámetros del mismo modelo entre diferentes condiciones."""
    print(f"\n{'='*100}")
    print("📊 COMPARACIÓN DE PARÁMETROS ENTRE CONDICIONES")
    print(f"{'='*100}")
    
    # Obtener todos los parámetros únicos
    todos_params = set()
    for cond, nombre in mejores.items():
        if nombre in res_por_cond[cond]:
            todos_params.update(res_por_cond[cond][nombre]['params'].keys())
    
    todos_params = sorted(list(todos_params))
    
    # Imprimir tabla comparativa
    condiciones = sorted(mejores.keys())
    header = f"{'Parámetro':<15}" + "".join([f"{cond:>20}" for cond in condiciones])
    print(f"\n{header}")
    print("─" * len(header))
    
    for param in todos_params:
        fila = f"{param:<15}"
        for cond in condiciones:
            nombre = mejores[cond]
            if nombre in res_por_cond[cond] and param in res_por_cond[cond][nombre]['params']:
                val = res_por_cond[cond][nombre]['params'][param]
                err = res_por_cond[cond][nombre]['params_err'].get(param, 0)
                fila += f"{val:.4f} ± {err:.4f}  "
            else:
                fila += "N/A".rjust(20)
        print(fila)
    
    # Análisis de tendencias
    print(f"\n📈 ANÁLISIS DE TENDENCIAS:")
    print(f"{'─'*100}")
    
    for param in todos_params:
        valores = []
        for cond in condiciones:
            nombre = mejores[cond]
            if nombre in res_por_cond[cond] and param in res_por_cond[cond][nombre]['params']:
                val = res_por_cond[cond][nombre]['params'][param]
                valores.append((cond, val))
        
        if len(valores) >= 2:
            # Calcular cambio relativo
            v0 = valores[0][1]
            if v0 != 0:
                cambios = [(v[0], (v[1] - v0) / v0 * 100) for v in valores[1:]]
                print(f"\n{param}:")
                print(f"  Valor base ({valores[0][0]}): {v0:.4f}")
                for cond, cambio in cambios:
                    direccion = "↓" if cambio < 0 else "↑"
                    print(f"  {cond}: {direccion} {cambio:+.1f}%")
# ═════════════ 7. MAIN ═════════════
if __name__ == "__main__":
    CSV_PATH = r"DatosIsoterma-DI.csv"
    
    # Cargar datos
    df_stats, df_ajuste = cargar_y_agrupar_replicas(CSV_PATH)
    #graficar_cv(df_stats)
    #graficar_datos(df_stats)
    
    # Ajustar modelos
    res_por_cond = ajustar_modelos_por_condicion(df_ajuste)
    if not res_por_cond:
        raise SystemExit("❌ sin ajustes")
    
    # Seleccionar mejores modelos (automáticamente elige el mejor para DI y lo fuerza para las demás)
    mejores = seleccionar_mejores_modelos(res_por_cond, modelo_fijo=None)
    if not mejores:
        raise SystemExit("❌ sin modelo aceptable")
    
    # Agregar bootstrap
    agregar_bootstrap(res_por_cond, df_ajuste, mejores, B=500)
    
    # Imprimir resultados
    imprimir_parametros_ajustados(res_por_cond)
    imprimir_tabla_por_condicion(res_por_cond)
    
    # NUEVO: Comparar parámetros entre condiciones
    comparar_parametros_entre_condiciones(res_por_cond, mejores)
    
    # Graficar
    graficar_ajustes_por_condicion(df_stats, res_por_cond, mejores)
    graficar_descomposicion(df_stats, res_por_cond, mejores)
    
    # Graficar el modelo específico (cambiar según el modelo seleccionado)
    modelo_principal = list(mejores.values())[0]  # Usar el modelo de la primera condición
    graficar_modelo(df_stats, res_por_cond, modelo_principal)
    
    print("\n✅ ANÁLISIS COMPLETADO")


    R = 8.314  # J/(mol·K)
    T = 298.15  # K

    KL_DI = 0.02771  # L/µg
    KL_NaCl = 0.05756
    KL_Natural = 0.001769

    # Convertir a L/mol (masa molar As = 74.92 g/mol)
    factor = 74.92 * 1e6  # µg/g → g/mol

    KL_DI_mol = KL_DI * factor
    KL_NaCl_mol = KL_NaCl * factor
    KL_Natural_mol = KL_Natural * factor

    # Energía libre de adsorción
    dG_DI = -R * T * np.log(KL_DI_mol)
    dG_NaCl = -R * T * np.log(KL_NaCl_mol)
    dG_Natural = -R * T * np.log(KL_Natural_mol)

    print(f"ΔG° adsorción:")
    print(f"  DI:      {dG_DI/1000:.2f} kJ/mol")
    print(f"  NaCl:    {dG_NaCl/1000:.2f} kJ/mol")
    print(f"  Natural: {dG_Natural/1000:.2f} kJ/mol")

    from scipy.optimize import fsolve
    import numpy as np

    def sips(Ce, Qmax, KL, n):
        return (Qmax * KL * Ce**n) / (1 + KL * Ce**n)

    # Dosis efectiva de Fe (normalizada)
    dosis_Fe = 0.004368  # g/L (equivalente a 4.368 mg/L)
    Co = 10.0            # µg/L (límite OMS)

    print(f"{'='*70}")
    print(f"REMOCIÓN DE As(V) A 10 µg/L - Dosis efectiva de Fe: {dosis_Fe*1000:.3f} mg/L")
    print(f"{'='*70}")

    resultados = []
    for cond, Qmax, KL, n in [
        ('DI',      103.3,   0.02771,  0.5238),
        ('NaCl',     59.3,   0.05756,  0.6193),
        ('Natural',  87.6,   0.001769, 0.8629)
    ]:
        # Capacidad máxima teórica del sistema
        q_max_sistema = Qmax * dosis_Fe  # µg/L

        # Balance de masa: (Co - Ce)/dosis = Qmax*KL*Ce^n/(1+KL*Ce^n)
        def balance(Ce):
            if Ce <= 0:
                return Co / dosis_Fe  # valor grande positivo
            q_balance = (Co - Ce) / dosis_Fe
            q_sips = sips(Ce, Qmax, KL, n)
            return q_balance - q_sips

        # Resolver (Ce debe estar entre 0 y Co)
        Ce_eq = fsolve(balance, Co * 0.9)[0]  # valor inicial cercano a Co
        Ce_eq = max(0, min(Ce_eq, Co))  # acotar

        q_ads = (Co - Ce_eq) / dosis_Fe  # µg/g(Fe) adsorbido real
        remocion = (Co - Ce_eq) / Co * 100

        resultados.append({
            'cond': cond,
            'Qmax': Qmax,
            'KL': KL,
            'n': n,
            'q_max_sistema': q_max_sistema,
            'Ce_eq': Ce_eq,
            'q_ads': q_ads,
            'remocion': remocion
        })

        print(f"\n{cond}:")
        print(f"  Capacidad máx. teórica del sistema: {q_max_sistema:.3f} µg/L")
        print(f"  Ce en equilibrio:                   {Ce_eq:.3f} µg/L")
        print(f"  q adsorbido real:                   {q_ads:.3f} µg/g(Fe)")
        print(f"  % Remoción:                         {remocion:.2f}%")

    # Resumen en tabla
    print(f"\n{'='*70}")
    print(f"{'Condición':<10} {'Qmax(µg/g)':<12} {'KL(L/µg)':<10} {'q_max_sis':<10} {'Ce(µg/L)':<10} {'Remoción':<10}")
    print(f"{'-'*70}")
    for r in resultados:
        print(f"{r['cond']:<10} {r['Qmax']:<12.1f} {r['KL']:<10.4f} {r['q_max_sistema']:<10.3f} {r['Ce_eq']:<10.3f} {r['remocion']:<10.2f}%")