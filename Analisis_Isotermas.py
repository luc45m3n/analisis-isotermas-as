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
def gab(Ce, Qm, C, K):
    Ce_p = np.maximum(Ce, 0.0); eps = 1e-8
    d1 = np.where(1 - K*Ce_p <= eps, eps, 1 - K*Ce_p)
    d2 = np.where(1 - K*Ce_p + C*K*Ce_p <= eps, eps, 1 - K*Ce_p + C*K*Ce_p)
    return (Qm*C*K*Ce_p)/(d1*d2)
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
def gab_odr(b, x):                 return gab(x, *b)
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
 'GAB':               dict(func=gab,             odr=gab_odr,             beta0=[40, 100, 0.01],       params=['Qm', 'C', 'K']),
 'Khan':              dict(func=khan,            odr=khan_odr,            beta0=[80, 0.01, 1.0],       params=['Qmax', 'KL', 'm']),
 'halsey':            dict(func=halsey,          odr=halsey_odr,          beta0=[0.01, 1.0],           params=['Kh', 'm']),
 'Redlich_Peterson':  dict(func=redlich_peterson, odr=redlich_peterson_odr,
                             beta0=[10, 0.01, desacotar(0.5, 0, 1)],       params=['K_RP', 'a_RP', 't'],
                             acotado=('t', 0.0, 1.0)),
}
# Fuera de la batería (sin Cs físico / inestable): bet, Fritz_Schlunder.
# Reincorporar solo con Cs real y forma (Qm*Kf*Ce)/(1+Kf*Ce**m), respectivamente.

def valido_fisico(nombre, p):
    if not np.all(np.isfinite(p)): return False
    if nombre == 'Double_Langmuir_ord': return p[0] > 0 and p[1] > 0 and p[2] > 0
    if nombre == 'halsey':  return p[1] > 0
    if nombre == 'GAB':     return np.all(p > 0)
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
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={'simga_co': 'sigma_co', 'simga_ce': 'sigma_ce', 'simga_qe': 'sigma_qe',
                       'ce': 'Ce', 'qe': 'Qe', 'co': 'Co'}, inplace=True)
    df['condicion'] = df['ph'].apply(lambda x: f'pH_{x}') if 'ph' in df.columns else 'condicion_1'
    df = df[(df['Ce'] > 0) & (df['Qe'] > 0)].copy().sort_values(['condicion', 'Co']).reset_index(drop=True)

    grupos = []
    for cond, sub in df.groupby('condicion'):
        sub = sub.sort_values('Co').reset_index(drop=True)
        gid, co0 = 0, None
        for idx, row in sub.iterrows():
            if co0 is None or abs(row['Co'] - co0)/co0 > tolerancia_co:
                gid += 1; co0 = row['Co']
            sub.at[idx, 'grupo'] = gid
        grupos.append(sub)
    df_agr = pd.concat(grupos, ignore_index=True)

    st = df_agr.groupby(['condicion', 'grupo']).agg(
        Co=('Co', 'mean'), Ce_mean=('Ce', 'mean'), Ce_std=('Ce', 'std'),
        Ce_count=('Ce', 'count'), Qe_mean=('Qe', 'mean'), Qe_std=('Qe', 'std')).reset_index()
    st.loc[st.Ce_count == 1, 'Ce_std'] = st.loc[st.Ce_count == 1, 'Ce_mean']*0.01
    st.loc[st.Ce_count == 1, 'Qe_std'] = st.loc[st.Ce_count == 1, 'Qe_mean']*0.05

    df_ajuste = df_agr[['condicion', 'grupo', 'Co', 'Ce', 'Qe']].merge(
        st[['condicion', 'grupo', 'Ce_std', 'Qe_std']], on=['condicion', 'grupo'])
    df_ajuste.rename(columns={'Ce': 'Ce_raw', 'Qe': 'Qe_raw'}, inplace=True)
    print(f"✅ {len(st)} puntos únicos; {len(df_ajuste)} réplicas "
          f"{df_ajuste.condicion.value_counts().to_dict()}")
    return st, df_ajuste

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
            if nombre == 'GAB': beta0 = [40.0, 100.0, 0.8/Ce.max()]

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

def seleccionar_mejores_modelos(res_por_cond):
    mejores = {}
    for cond, resd in res_por_cond.items():
        val = {k: v for k, v in resd.items() if aceptable(v) and np.isfinite(v['AICc'])}
        if not val: print(f"❌ {cond}: sin modelos aceptables"); continue
        orden = sorted(val, key=lambda k: val[k]['AICc'])
        amin  = val[orden[0]]['AICc']
        mejor = min([k for k in orden if val[k]['AICc'] - amin < 2],
                    key=lambda k: len(val[k]['popt']))
        mejores[cond] = mejor
        print(f"\n🏆 {cond}: {mejor}  (AICc={val[mejor]['AICc']:.2f}, χ²/dof={val[mejor]['chi2_red']:.3f})")
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

# ═════════════ 7. MAIN ═════════════
if __name__ == "__main__":
    CSV_PATH = r"C:\Users\Lucas\Documents\CNEA\Latex\Informe de Avance 2026-2027\python\datos_isotermas.csv"

    df_stats, df_ajuste = cargar_y_agrupar_replicas(CSV_PATH)
    graficar_datos(df_stats)
    res_por_cond = ajustar_modelos_por_condicion(df_ajuste)
    if not res_por_cond: raise SystemExit("❌ sin ajustes")

    mejores = seleccionar_mejores_modelos(res_por_cond)          # antes de imprimir
    if not mejores: raise SystemExit("❌ sin modelo aceptable")

    agregar_bootstrap(res_por_cond, df_ajuste, mejores, B=500)   # ahora sí, antes del reporte

    imprimir_parametros_ajustados(res_por_cond)
    imprimir_tabla_por_condicion(res_por_cond)
    graficar_ajustes_por_condicion(df_stats, res_por_cond, mejores)
    graficar_descomposicion(df_stats, res_por_cond, mejores)
    graficar_modelo(df_stats, res_por_cond, 'LF_hybrid')
    print("\n✅ ANÁLISIS COMPLETADO")

    
    