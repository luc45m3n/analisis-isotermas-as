#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isotermas de adsorción As(V) — versión refactorizada.

Qué hace
--------
1. Lee el CSV (mismo formato que la versión anterior), promedia las mediciones
   FIASS dentro de cada falcón y agrupa los falcones por nivel de Co.
2. Estima la incertidumbre de cada punto con un modelo de varianza agrupado
   (SD = a·media^b, con corrección de sesgo del log) o con la SD cruda del grupo.
3. Ajusta cada modelo por ODR (errores en Ce y Qe) con multi-arranque.
   Los parámetros se optimizan en escala log (positivos) o logit (acotados),
   de modo que los intervalos de Wald siempre respetan los límites físicos.
4. Compara modelos con AICc, QAICc (corrige sobredispersión), pesos de Akaike
   y test de falta de ajuste contra error puro (aprox.).
5. IC de parámetros: Wald (todos los modelos) y bootstrap (candidatos).
   IC de magnitudes derivadas (q a Ce límite, Ce de equilibrio, % remoción,
   dosis requerida, ΔG°, K2, 1/n, ...) por bootstrap o simulación MVN.
6. Exporta tablas (CSV/XLSX) y figuras (ajustes con banda, descomposición,
   residuos estandarizados, modelo de varianza).

Uso
---
    python estudio_isotermas_v2.py DatosIsoterma-DI.csv
    python estudio_isotermas_v2.py datos.csv --boot 2000 --no-mostrar
    python estudio_isotermas_v2.py datos.csv --varianza grupo --seleccion independiente
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.optimize import brentq
from scipy.special import digamma, expit
from scipy.stats import f as f_dist
from scipy.stats import t as student_t

# ═════════════════════════════════════════════════════════════════════════════
# 0. CONSTANTES Y CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════
R_GAS = 8.314      # J/(mol·K)
T_K = 298.15       # K
M_AS = 74.92       # g/mol (As total)
UNIDAD_CE = "µg/L"
UNIDAD_QE = "µg/mg"   # numéricamente = mg/g

DELTA_PARSIMONIA = 2.0   # entre modelos con ΔQAICc < 2 se elige el de menos parámetros
LIM_TH = 25.0            # |θ| mayor => parámetro pegado a un límite (no identificable)

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 11, "axes.titlesize": 12})


@dataclass
class Config:
    csv_path: str = "DatosIsoterma-DI.csv"
    outdir: str = "resultados_isotermas"
    # --- datos ---
    tolerancia_co: float = 0.05          # agrupación de réplicas por Co (5 %)
    modo_varianza: str = "suavizado"     # 'suavizado' | 'grupo'
    # --- ajuste ---
    modelos: tuple = ("Langmuir", "Freundlich", "Sips", "Double_Langmuir",
                      "LF_hybrid", "Khan", "Redlich_Peterson")   # + "Halsey" si querés
    n_starts: int = 8
    r2_min: float = 0.5
    # --- selección ---
    modo_seleccion: str = "compartido"   # 'compartido' (ganador de la 1ª cond.) | 'independiente'
    modelo_fijo: Optional[str] = None
    # --- incertidumbre ---
    nivel: float = 0.95
    boot_B: int = 1000
    boot_modo: str = "estratificado"     # 'estratificado' | 'parametrico'
    boot_cuales: str = "candidatos"      # 'ninguno' | 'ganador' | 'candidatos' | 'todos'
    delta_candidatos: float = 2.0
    n_mvn: int = 2000
    # --- magnitudes derivadas ---
    ce_limite: float = 10.0              # µg/L (límite OMS)
    co_tratar: float = 10.0              # µg/L, Co del agua a tratar
    dosis_mg_L: Optional[float] = None   # None => se deduce del CSV (masa_mg / v_ml)
    # --- salida ---
    barras: str = "sd"                   # 'sd' | 'ic95' (solo visualización)
    mostrar: bool = True
    seed: int = 2026


# ═════════════════════════════════════════════════════════════════════════════
# 1. MODELOS Y PARAMETRIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════
def langmuir(Ce, Qmax, KL):
    return (Qmax * KL * Ce) / (1 + KL * Ce)


def freundlich(Ce, Kf, n):
    return Kf * np.power(Ce, 1.0 / n)


def sips(Ce, Qmax, KL, n):
    return (Qmax * KL * np.power(Ce, n)) / (1 + KL * np.power(Ce, n))


def double_langmuir(Ce, Q1, K1, Q2, r):
    """Dos sitios con K2 = r·K1 (0<r<1): rompe la simetría de intercambio."""
    K2 = K1 * r
    return (Q1 * K1 * Ce) / (1 + K1 * Ce) + (Q2 * K2 * Ce) / (1 + K2 * Ce)


def lf_hybrid(Ce, Qmax, KL, Kf, m):
    return (Qmax * KL * Ce) / (1 + KL * Ce) + Kf * np.power(Ce, 1.0 / m)


def khan(Ce, Qmax, KL, m):
    return (Qmax * KL * Ce) / np.power(1 + KL * Ce, m)


def halsey(Ce, Kh, m):
    # Decreciente en Ce: solo tiene sentido con Ce < Kh. Desactivado por defecto.
    return (1.0 / m) * np.log(Kh) - (1.0 / m) * np.log(Ce)


def redlich_peterson(Ce, K_RP, a_RP, beta):
    return (K_RP * Ce) / (1.0 + a_RP * np.power(Ce, beta))


@dataclass(frozen=True)
class Par:
    """Parámetro con su transformación: log (positivo) o logit (acotado a (a, b))."""
    nombre: str
    lim: Optional[tuple] = None

    def a_nat(self, th):
        th = np.asarray(th, float)
        if self.lim is None:
            return np.exp(th)
        a, b = self.lim
        return a + (b - a) * expit(th)

    def a_th(self, p):
        p = np.asarray(p, float)
        if self.lim is None:
            return np.log(p)
        a, b = self.lim
        u = (p - a) / (b - a)
        return np.log(u / (1 - u))

    def dp_dth(self, th):
        th = np.asarray(th, float)
        if self.lim is None:
            return np.exp(th)
        a, b = self.lim
        s = expit(th)
        return (b - a) * s * (1 - s)


@dataclass
class ModeloSpec:
    nombre: str
    func: Callable
    pars: list
    p0: Callable                                   # (Ce, Qe) -> valores iniciales naturales
    derivadas: dict = field(default_factory=dict)  # nombre -> f(dict de parámetros)
    afinidad: Optional[tuple] = None               # (param K, param exponente | None) para ΔG°
    componentes: Optional[Callable] = None         # (Ce, p) -> [(etiqueta, y), ...]

    @property
    def nombres(self):
        return [p.nombre for p in self.pars]

    @property
    def k(self):
        return len(self.pars)

    def a_nat(self, th):
        return np.array([p.a_nat(th[i]) for i, p in enumerate(self.pars)])

    def a_nat_mat(self, TH):
        return np.column_stack([p.a_nat(TH[:, i]) for i, p in enumerate(self.pars)])

    def a_th(self, p):
        return np.array([par.a_th(p[i]) for i, par in enumerate(self.pars)])

    def odr(self, th, x):
        """Función con la firma que espera scipy.odr: f(beta, x)."""
        return self.func(x, *self.a_nat(th))


def _escalas(Ce, Qe):
    return float(np.max(Qe)), float(np.median(Ce)), float(np.max(Ce))


def construir_modelos() -> dict:
    P = Par
    M = {}

    M["Langmuir"] = ModeloSpec(
        "Langmuir", langmuir, [P("Qmax"), P("KL")],
        lambda Ce, Qe: [1.2 * _escalas(Ce, Qe)[0], 1 / _escalas(Ce, Qe)[1]],
        afinidad=("KL", None))

    M["Freundlich"] = ModeloSpec(
        "Freundlich", freundlich, [P("Kf"), P("n")],
        lambda Ce, Qe: [np.median(Qe) / _escalas(Ce, Qe)[1] ** 0.5, 2.0],
        derivadas={"1/n": lambda d: 1.0 / d["n"]})

    M["Sips"] = ModeloSpec(
        "Sips", sips, [P("Qmax"), P("KL"), P("n")],
        lambda Ce, Qe: [1.2 * _escalas(Ce, Qe)[0], 1 / _escalas(Ce, Qe)[1], 1.0],
        afinidad=("KL", "n"))

    M["Double_Langmuir"] = ModeloSpec(
        "Double_Langmuir", double_langmuir,
        [P("Q1"), P("K1"), P("Q2"), P("r", (0.0, 1.0))],
        lambda Ce, Qe: [_escalas(Ce, Qe)[0], 2 / _escalas(Ce, Qe)[1],
                        0.5 * _escalas(Ce, Qe)[0], 0.1],
        derivadas={"K2": lambda d: d["K1"] * d["r"], "Q1+Q2": lambda d: d["Q1"] + d["Q2"]},
        afinidad=("K1", None),
        componentes=lambda Ce, p: [
            (f"sitio 1 (Q1={p[0]:.3g}, K1={p[1]:.3g})", langmuir(Ce, p[0], p[1])),
            (f"sitio 2 (Q2={p[2]:.3g}, K2={p[1] * p[3]:.3g})", langmuir(Ce, p[2], p[1] * p[3]))])

    M["LF_hybrid"] = ModeloSpec(
        "LF_hybrid", lf_hybrid, [P("Qmax"), P("KL"), P("Kf"), P("m")],
        lambda Ce, Qe: [0.7 * _escalas(Ce, Qe)[0], 1 / _escalas(Ce, Qe)[1],
                        0.3 * _escalas(Ce, Qe)[0] / _escalas(Ce, Qe)[2] ** 0.5, 2.0],
        derivadas={"1/m": lambda d: 1.0 / d["m"]},
        afinidad=("KL", None),
        componentes=lambda Ce, p: [
            (f"monocapa (Qmax={p[0]:.3g}, KL={p[1]:.3g})", langmuir(Ce, p[0], p[1])),
            (f"término Freundlich (Kf={p[2]:.3g}, 1/m={1 / p[3]:.3g})",
             p[2] * np.power(Ce, 1.0 / p[3]))])

    M["Khan"] = ModeloSpec(
        "Khan", khan, [P("Qmax"), P("KL"), P("m")],
        lambda Ce, Qe: [1.2 * _escalas(Ce, Qe)[0], 1 / _escalas(Ce, Qe)[1], 1.0],
        afinidad=("KL", None))

    M["Halsey"] = ModeloSpec(
        "Halsey", halsey, [P("Kh"), P("m")],
        lambda Ce, Qe: [10 * _escalas(Ce, Qe)[2], 1.0])

    M["Redlich_Peterson"] = ModeloSpec(
        "Redlich_Peterson", redlich_peterson,
        [P("K_RP"), P("a_RP"), P("beta", (0.0, 1.0))],
        lambda Ce, Qe: [2 * _escalas(Ce, Qe)[0] / _escalas(Ce, Qe)[1],
                        _escalas(Ce, Qe)[1] ** -0.5, 0.5])
    return M


MODELOS = construir_modelos()


# ═════════════════════════════════════════════════════════════════════════════
# 2. CARGA, AGRUPACIÓN Y MODELO DE VARIANZA
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Datos:
    stats: pd.DataFrame     # una fila por (condición, nivel de Co): medias, SD, CV
    ajuste: pd.DataFrame    # una fila por falcón: condicion, grupo, Ce, Qe, sx, sy
    var_info: dict          # (condicion, 'Ce'|'Qe') -> (a, b) del modelo de varianza
    dosis_mg_L: float


def _cv(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.mean(x) == 0:
        return 0.0
    return float(np.std(x, ddof=1) / np.mean(x) * 100)


def _agrupar_por_co(co, tol):
    orden = np.argsort(co)
    gid = np.zeros(len(co), int)
    g, ref = 0, None
    for i in orden:
        if ref is None or abs(co[i] - ref) / max(abs(ref), 1e-12) > tol:
            g += 1
            ref = co[i]
        gid[i] = g
    return gid


def ajustar_varianza(media, sd, n):
    """
    log σ = a + b·log(media), ponderado por grados de libertad (n-1).
    Corrige el sesgo de E[ln s]: E[ln s] = ln σ + ½[ψ(ν/2) − ln(ν/2)]
    (con n=3 el sesgo es ≈ −0.29, o sea las SD crudas subestiman σ en ~25 %).
    Devuelve (a, b) o None si hay menos de 4 grupos con n ≥ 2.
    """
    media, sd, n = (np.asarray(v, float) for v in (media, sd, n))
    ok = (n >= 2) & np.isfinite(sd) & (sd > 0) & (media > 0)
    if ok.sum() < 4:
        return None
    nu = n[ok] - 1
    x = np.log(media[ok])
    y = np.log(sd[ok]) - 0.5 * (digamma(nu / 2) - np.log(nu / 2))
    xm, ym = np.average(x, weights=nu), np.average(y, weights=nu)
    sxx = np.sum(nu * (x - xm) ** 2)
    b = np.sum(nu * (x - xm) * (y - ym)) / sxx if sxx > 1e-12 else 1.0
    b = float(np.clip(b, 0.0, 2.0))
    a = float(np.average(y - b * x, weights=nu))
    return a, b


def _piso_sd(media):
    """Piso único y relativo, usado en el ajuste Y en el bootstrap (consistencia)."""
    return np.maximum(1e-3 * np.abs(media), 1e-9)


def cargar_datos(cfg: Config) -> Datos:
    df = pd.read_csv(cfg.csv_path, sep=";", decimal=".", comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Columnas detectadas: {list(df.columns)}")

    falt = [c for c in ("co", "ce", "v_ml", "masa_mg", "replica") if c not in df.columns]
    if falt:
        raise ValueError(f"Faltan columnas en el CSV: {falt}")

    # Qe = (Co − Ce)·V/m   [µg/L · L / mg = µg/mg ≡ mg/g]
    df["qe"] = (df["co"] - df["ce"]) * (df["v_ml"] / 1000.0) / df["masa_mg"]
    dosis = float(np.median(df["masa_mg"] / (df["v_ml"] / 1000.0)))   # mg/L
    df = df.rename(columns={"co": "Co", "ce": "Ce", "qe": "Qe"})

    # --- condición ---
    if "condicion" not in df.columns:
        if "ph" in df.columns:
            df["condicion"] = df["ph"].apply(lambda x: f"pH_{x}")
        elif "matriz" in df.columns:
            df["condicion"] = df["matriz"]
        elif len(df) == 63:
            df["condicion"] = ["DI"] * 21 + ["NaCl"] * 21 + ["Natural"] * 21
            print("AVISO: sin columna de condición; se asignó DI/NaCl/Natural por posición (63 filas).")
        else:
            df["condicion"] = "condicion_1"
            print("AVISO: sin columna de condición; se asignó 'condicion_1'.")

    n0 = len(df)
    df = df[(df["Ce"] > 0) & (df["Qe"] > 0)].copy()
    if len(df) < n0:
        print(f"AVISO: {n0 - len(df)} mediciones con Ce<=0 o Qe<=0 descartadas "
              f"(censurar valores ≤0 sesga Qe hacia arriba a Ce bajos).")
    print(f"Datos: {len(df)} mediciones FIASS | condiciones: {list(df['condicion'].unique())}")
    print(f"Masa: {df['masa_mg'].min():.3f}-{df['masa_mg'].max():.3f} mg | "
          f"V: {df['v_ml'].unique()} mL | dosis deducida: {dosis:.2f} mg/L")

    # --- paso 1: promediar FIASS dentro de cada falcón ---
    rep = (df.groupby(["condicion", "replica", "Co"], as_index=False)
             .agg(Ce=("Ce", "mean"), Qe=("Qe", "mean"),
                  Ce_cv_fiass=("Ce", _cv), Qe_cv_fiass=("Qe", _cv)))

    # --- paso 2: agrupar falcones por nivel de Co ---
    rep["grupo"] = 0
    for cond, idx in rep.groupby("condicion").groups.items():
        rep.loc[idx, "grupo"] = _agrupar_por_co(rep.loc[idx, "Co"].to_numpy(), cfg.tolerancia_co)

    # --- paso 3: estadísticos entre falcones ---
    st = (rep.groupby(["condicion", "grupo"])
             .agg(Co=("Co", "mean"), n=("Ce", "size"),
                  Ce_mean=("Ce", "mean"), Ce_std=("Ce", "std"),
                  Qe_mean=("Qe", "mean"), Qe_std=("Qe", "std"),
                  Ce_cv_fiass=("Ce_cv_fiass", "mean"), Qe_cv_fiass=("Qe_cv_fiass", "mean"))
             .reset_index())
    st["Ce_cv"] = st["Ce_std"] / st["Ce_mean"] * 100
    st["Qe_cv"] = st["Qe_std"] / st["Qe_mean"] * 100

    # --- paso 4: SD que se usará como peso ---
    var_info = {}
    for var, rel_fallback in (("Ce", 0.01), ("Qe", 0.05)):
        st[f"{var}_sd"] = np.nan
    for cond, idx in st.groupby("condicion").groups.items():
        sub = st.loc[idx]
        for var, rel_fallback in (("Ce", 0.01), ("Qe", 0.05)):
            media, sd_g = sub[f"{var}_mean"].to_numpy(), sub[f"{var}_std"].to_numpy()
            ab = ajustar_varianza(media, sd_g, sub["n"].to_numpy())
            var_info[(cond, var)] = ab
            sd_fit = np.exp(ab[0]) * media ** ab[1] if ab is not None else None
            if cfg.modo_varianza == "suavizado" and sd_fit is not None:
                sd = sd_fit
            else:
                sd = sd_g.copy()
                falta = ~np.isfinite(sd)
                sd[falta] = sd_fit[falta] if sd_fit is not None else rel_fallback * media[falta]
            st.loc[idx, f"{var}_sd"] = np.maximum(sd, _piso_sd(media))

    df_ajuste = (rep[["condicion", "grupo", "Ce", "Qe"]]
                 .merge(st[["condicion", "grupo", "Ce_sd", "Qe_sd"]], on=["condicion", "grupo"])
                 .rename(columns={"Ce_sd": "sx", "Qe_sd": "sy"}))

    _reporte_variabilidad(st, var_info, cfg)
    return Datos(st, df_ajuste, var_info, dosis)


def _reporte_variabilidad(st, var_info, cfg):
    print(f"\n{'═' * 90}\nVARIABILIDAD (modo de varianza: {cfg.modo_varianza})\n{'═' * 90}")
    for cond in sorted(st.condicion.unique()):
        s = st[st.condicion == cond]
        t = pd.DataFrame({
            "Co": s.Co, "n": s.n,
            "Ce_media": s.Ce_mean, "CV_Ce%": s.Ce_cv, "CV_FIASS_Ce%": s.Ce_cv_fiass,
            "Qe_media": s.Qe_mean, "CV_Qe%": s.Qe_cv, "CV_FIASS_Qe%": s.Qe_cv_fiass,
            "sx_usada": s.Ce_sd, "sy_usada": s.Qe_sd})
        print(f"\n{cond}")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        for var in ("Ce", "Qe"):
            ab = var_info.get((cond, var))
            if ab is None:
                print(f"  modelo de varianza {var}: no estimable (<4 grupos con n≥2)")
            else:
                print(f"  modelo de varianza {var}: SD = {np.exp(ab[0]):.4g}·media^{ab[1]:.2f}"
                      f"{'  (≈ CV constante)' if abs(ab[1] - 1) < 0.2 else ''}")
    pobre = st[st.n < 3]
    if len(pobre):
        print(f"\nAVISO: {len(pobre)} niveles de Co con n<3 falcones; su SD proviene del modelo de varianza.")


# ═════════════════════════════════════════════════════════════════════════════
# 3. AJUSTE ODR
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Bootstrap:
    P: np.ndarray        # (B_ok, k) parámetros en escala natural
    tasa: float          # fracción de remuestras que convergieron
    ci: dict             # nombre -> (lo, hi) percentil
    modo: str


@dataclass
class Ajuste:
    spec: ModeloSpec
    theta: np.ndarray
    sd_theta: np.ndarray
    cov_theta: np.ndarray
    popt: np.ndarray
    se: np.ndarray
    ci_wald: dict
    xplus: np.ndarray
    n: int
    chi2: float
    chi2_red: float
    r2: float
    r2_adj: float
    rmse: float
    aicc: float
    p_lof: float
    ident: bool
    n_starts_ok: int
    c_hat: float = np.nan
    qaicc: float = np.nan
    delta: float = np.nan
    peso: float = np.nan
    boot: Optional[Bootstrap] = None
    mvn: Optional[np.ndarray] = None

    @property
    def nombre(self):
        return self.spec.nombre

    @property
    def k(self):
        return self.spec.k

    @property
    def params(self):
        return dict(zip(self.spec.nombres, self.popt))


def _correr_odr(spec, Ce, Qe, sx, sy, th0, maxit):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return ODR(RealData(Ce, Qe, sx=sx, sy=sy), Model(spec.odr),
                   beta0=th0, maxit=maxit).run()


def _convergio(r):
    # ODRPACK: 1, 2 y 3 son convergencia (3 = ambos criterios); 4 = límite de iteraciones.
    return r.info in (1, 2, 3) and np.all(np.isfinite(r.beta))


def _test_falta_ajuste(y, yhat, sy, grupo, k):
    """F de falta de ajuste contra error puro (ponderado, residuos verticales: aproximado)."""
    w = 1.0 / sy ** 2
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_pe, df_pe = 0.0, 0
    for g in np.unique(grupo):
        ix = np.where(grupo == g)[0]
        if len(ix) > 1:
            ss_pe += np.sum(w[ix] * (y[ix] - np.mean(y[ix])) ** 2)
            df_pe += len(ix) - 1
    df_lof = (len(y) - k) - df_pe
    if df_pe <= 0 or df_lof <= 0 or ss_pe <= 0:
        return np.nan
    F = max((ss_res - ss_pe) / df_lof / (ss_pe / df_pe), 0.0)
    return float(f_dist.sf(F, df_lof, df_pe))


def ajustar_un_modelo(spec: ModeloSpec, d: pd.DataFrame, cfg: Config, rng) -> Optional[Ajuste]:
    Ce, Qe = d.Ce.to_numpy(), d.Qe.to_numpy()
    sx, sy, grupo = d.sx.to_numpy(), d.sy.to_numpy(), d.grupo.to_numpy()
    n, k = len(Ce), spec.k
    if n < k + 2:
        return None

    th_nom = spec.a_th(spec.p0(Ce, Qe))
    mejor, n_ok = None, 0
    for i in range(cfg.n_starts):
        th0 = th_nom if i == 0 else th_nom + rng.normal(0.0, 1.0, k)
        try:
            r = _correr_odr(spec, Ce, Qe, sx, sy, th0, maxit=5000)
        except Exception:
            continue
        if _convergio(r):
            n_ok += 1
            if mejor is None or r.sum_square < mejor.sum_square - 1e-12:
                mejor = r
    if mejor is None:
        return None

    th = np.asarray(mejor.beta, float)
    popt = spec.a_nat(th)
    sd_th = np.asarray(mejor.sd_beta, float)
    cov_th = np.asarray(mejor.cov_beta, float) * mejor.res_var
    se = np.array([abs(par.dp_dth(th[i])) * sd_th[i] for i, par in enumerate(spec.pars)])

    dof = n - k
    tc = student_t.ppf(0.5 + cfg.nivel / 2, max(dof, 1))
    ci = {par.nombre: (float(par.a_nat(th[i] - tc * sd_th[i])), float(par.a_nat(th[i] + tc * sd_th[i])))
          for i, par in enumerate(spec.pars)}

    yhat = spec.func(Ce, *popt)
    ss_res, ss_tot = np.sum((Qe - yhat) ** 2), np.sum((Qe - Qe.mean()) ** 2)
    chi2 = float(mejor.sum_square)                       # objetivo ODR: incluye error en x e y
    aicc = chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.nan
    ident = bool(np.all(np.isfinite(sd_th)) and np.all(np.isfinite(cov_th))
                 and np.all(np.abs(th) < LIM_TH))

    return Ajuste(
        spec=spec, theta=th, sd_theta=sd_th, cov_theta=cov_th, popt=popt, se=se, ci_wald=ci,
        xplus=np.asarray(mejor.xplus, float), n=n, chi2=chi2, chi2_red=chi2 / dof,
        r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        r2_adj=1 - (ss_res / dof) / (ss_tot / (n - 1)) if dof > 0 and ss_tot > 0 else np.nan,
        rmse=float(np.sqrt(ss_res / n)), aicc=float(aicc),
        p_lof=_test_falta_ajuste(Qe, yhat, sy, grupo, k),
        ident=ident, n_starts_ok=n_ok)


def ajustar_todo(datos: Datos, cfg: Config, rng) -> dict:
    out = {}
    for cond, d in datos.ajuste.groupby("condicion"):
        print(f"\n{cond}: {len(d)} falcones, {d.grupo.nunique()} niveles de Co")
        res = {}
        for nombre in cfg.modelos:
            aj = ajustar_un_modelo(MODELOS[nombre], d, cfg, rng)
            if aj is None:
                print(f"  ✗ {nombre:<18} sin convergencia o pocos puntos")
                continue
            res[nombre] = aj
            print(f"  {'✓' if aj.ident else '⚠'} {nombre:<18} R²={aj.r2:.4f}  "
                  f"χ²/gl={aj.chi2_red:.2f}  AICc={aj.aicc:.2f}  "
                  f"({aj.n_starts_ok}/{cfg.n_starts} arranques convergieron)")
        evaluar_modelos(res)
        out[cond] = res
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. COMPARACIÓN Y SELECCIÓN DE MODELOS
# ═════════════════════════════════════════════════════════════════════════════
def evaluar_modelos(res: dict):
    """
    QAICc = χ²/ĉ + 2k + 2k(k+1)/(n−k−1), con ĉ = max(1, χ²_red del modelo más complejo).
    Si los pesos están bien especificados ĉ≈1 y QAICc = AICc. Si ĉ≫1, los ΔAICc crudos
    están inflados y favorecen modelos sobreparametrizados.
    """
    val = {n: a for n, a in res.items() if a.ident and np.isfinite(a.aicc)}
    if not val:
        return
    kmax = max(a.k for a in val.values())
    c_hat = max(1.0, min(a.chi2_red for a in val.values() if a.k == kmax))
    for a in val.values():
        a.c_hat = c_hat
        a.qaicc = a.chi2 / c_hat + 2 * a.k + 2 * a.k * (a.k + 1) / (a.n - a.k - 1)
    qmin = min(a.qaicc for a in val.values())
    for a in val.values():
        a.delta = a.qaicc - qmin
    suma = sum(np.exp(-a.delta / 2) for a in val.values())
    for a in val.values():
        a.peso = np.exp(-a.delta / 2) / suma


def elegir_en_condicion(res: dict, cfg: Config) -> Optional[str]:
    cand = [a for a in res.values()
            if a.ident and np.isfinite(a.qaicc) and a.r2 >= cfg.r2_min]
    if not cand:
        return None
    qmin = min(a.qaicc for a in cand)
    cerca = [a for a in cand if a.qaicc - qmin < DELTA_PARSIMONIA]
    return min(cerca, key=lambda a: (a.k, a.qaicc)).nombre


def _orden_condiciones(conds):
    def clave(c):
        c0 = c.lower()
        return (0 if c0.startswith("di") else 1 if c0.startswith("nacl") else 2, c)
    return sorted(conds, key=clave)


def seleccionar_modelos(res_cond: dict, cfg: Config) -> dict:
    conds = _orden_condiciones(res_cond)
    mejores = {}
    modelo_ref = cfg.modelo_fijo
    if cfg.modo_seleccion == "compartido" and modelo_ref is None:
        modelo_ref = elegir_en_condicion(res_cond[conds[0]], cfg)
        print(f"\nModelo de referencia (condición '{conds[0]}'): {modelo_ref}")
    for c in conds:
        if cfg.modo_seleccion == "compartido" and modelo_ref is not None:
            m = modelo_ref
            if m not in res_cond[c] or not res_cond[c][m].ident:
                alt = elegir_en_condicion(res_cond[c], cfg)
                print(f"  {c}: {m} no disponible/identificable → alternativa: {alt}")
                m = alt
        else:
            m = elegir_en_condicion(res_cond[c], cfg)
        if m is None:
            print(f"  {c}: sin modelo aceptable")
            continue
        mejores[c] = m
        a = res_cond[c][m]
        print(f"  {c}: {m} (ΔQAICc={a.delta:.2f}, peso={a.peso:.2f}, R²={a.r2:.4f}, "
              f"χ²/gl={a.chi2_red:.2f}, p_LOF={a.p_lof:.3g})")
    return mejores


def candidatos(res: dict, mejor: Optional[str], cfg: Config) -> list:
    if cfg.boot_cuales == "ninguno":
        return []
    if cfg.boot_cuales == "ganador":
        return [mejor] if mejor else []
    if cfg.boot_cuales == "todos":
        return [n for n, a in res.items() if a.ident]
    lista = [n for n, a in res.items() if a.ident and np.isfinite(a.delta)
             and a.delta <= cfg.delta_candidatos]
    if mejor and mejor not in lista:
        lista.append(mejor)
    return lista


# ═════════════════════════════════════════════════════════════════════════════
# 5. INCERTIDUMBRE: BOOTSTRAP Y SIMULACIÓN MVN
# ═════════════════════════════════════════════════════════════════════════════
def bootstrap_ajuste(aj: Ajuste, d: pd.DataFrame, cfg: Config, rng) -> Bootstrap:
    """
    'estratificado': remuestrea falcones CON reposición dentro de cada nivel de Co
                     (no paramétrico; mantiene el diseño y los pesos de cada nivel).
    'parametrico'  : x* = x̂+ε_x, y* = f(x̂)+ε_y, con x̂ = x+δ del ODR y los mismos
                     sx, sy del ajuste (propaga también el error en Ce).
    """
    spec = aj.spec
    Ce, Qe = d.Ce.to_numpy(), d.Qe.to_numpy()
    sx, sy, grupo = d.sx.to_numpy(), d.sy.to_numpy(), d.grupo.to_numpy()
    idx_g = [np.where(grupo == g)[0] for g in np.unique(grupo)]
    xp = aj.xplus
    yp = spec.func(xp, *aj.popt)

    draws = []
    for _ in range(cfg.boot_B):
        if cfg.boot_modo == "parametrico":
            xb = np.maximum(xp + rng.normal(0.0, sx), 1e-9)
            yb = yp + rng.normal(0.0, sy)
            sxb, syb = sx, sy
        else:
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_g])
            xb, yb, sxb, syb = Ce[idx], Qe[idx], sx[idx], sy[idx]
        try:
            r = _correr_odr(spec, xb, yb, sxb, syb, aj.theta, maxit=500)
        except Exception:
            continue
        if _convergio(r) and np.all(np.abs(r.beta) < LIM_TH):
            draws.append(r.beta)

    if len(draws) < 20:
        return Bootstrap(np.empty((0, spec.k)), len(draws) / cfg.boot_B, {}, cfg.boot_modo)
    P = spec.a_nat_mat(np.asarray(draws))
    lo, hi = np.percentile(P, [50 * (1 - cfg.nivel), 50 * (1 + cfg.nivel)], axis=0)
    ci = {nom: (float(lo[j]), float(hi[j])) for j, nom in enumerate(spec.nombres)}
    return Bootstrap(P, len(draws) / cfg.boot_B, ci, cfg.boot_modo)


def muestras_incertidumbre(aj: Ajuste, cfg: Config, rng):
    """Devuelve (P natural (N,k), fuente). Usa bootstrap si existe; si no, simulación MVN (Wald)."""
    if aj.boot is not None and len(aj.boot.P) >= 100:
        return aj.boot.P, "bootstrap"
    if aj.mvn is None:
        with np.errstate(all="ignore"):
            th = rng.multivariate_normal(aj.theta, aj.cov_theta, size=cfg.n_mvn, check_valid="ignore")
            th = th[np.all(np.isfinite(th), axis=1) & np.all(np.abs(th) < LIM_TH, axis=1)]
            aj.mvn = aj.spec.a_nat_mat(th)
    return aj.mvn, "MVN (Wald)"


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAGNITUDES DERIVADAS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Ctx:
    ce_limite: float
    co_tratar: float
    dosis: float          # mg/L


def _ce_equilibrio(func, p, Co, dosis):
    """Resuelve (Co − Ce)/dosis = q(Ce) en (0, Co]."""
    def bal(Ce):
        return (Co - Ce) / dosis - func(Ce, *p)
    try:
        with np.errstate(all="ignore"):
            return brentq(bal, 1e-9, Co, xtol=1e-9)
    except (ValueError, RuntimeError):
        return np.nan


def calcular_derivadas(spec: ModeloSpec, P: np.ndarray, ctx: Ctx) -> dict:
    """P: (N,k) en escala natural. Devuelve {magnitud: array(N)}. Sirve para punto, bootstrap y MVN."""
    P = np.atleast_2d(P)
    Pd = {nom: P[:, j] for j, nom in enumerate(spec.nombres)}
    out = {nom: f(Pd) for nom, f in spec.derivadas.items()}
    with np.errstate(all="ignore"):
        q_lim = np.asarray(spec.func(ctx.ce_limite, *P.T), float)
        out[f"q(Ce={ctx.ce_limite:g}) [{UNIDAD_QE}]"] = q_lim
        ce_eq = np.array([_ce_equilibrio(spec.func, p, ctx.co_tratar, ctx.dosis) for p in P])
        out[f"Ce_eq(Co={ctx.co_tratar:g}) [{UNIDAD_CE}]"] = ce_eq
        out["remocion [%]"] = (ctx.co_tratar - ce_eq) / ctx.co_tratar * 100
        if ctx.co_tratar > ctx.ce_limite:
            out[f"dosis para Ce={ctx.ce_limite:g} [mg/L]"] = (ctx.co_tratar - ctx.ce_limite) / q_lim
        if spec.afinidad is not None:
            kn, en = spec.afinidad
            expo = Pd[en] if en else 1.0
            K_mol = Pd[kn] * (M_AS * 1e6) ** expo           # L/mol (Ce en µg/L)
            out["ΔG° [kJ/mol]"] = -R_GAS * T_K * np.log(K_mol) / 1000.0
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 7. TABLAS Y REPORTES
# ═════════════════════════════════════════════════════════════════════════════
def _fmt(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4g}"


def tabla_modelos(res: dict, elegido: Optional[str] = None) -> pd.DataFrame:
    filas = []
    orden = sorted(res.items(), key=lambda kv: kv[1].qaicc if np.isfinite(kv[1].qaicc) else np.inf)
    for n, a in orden:
        filas.append({"modelo": n + (" ◀" if n == elegido else ""), "k": a.k,
                      "R2": a.r2, "R2_adj": a.r2_adj, "RMSE": a.rmse, "chi2_red": a.chi2_red,
                      "AICc": a.aicc, "QAICc": a.qaicc, "dQAICc": a.delta,
                      "w_Akaike": a.peso, "p_LOF": a.p_lof, "identificable": "sí" if a.ident else "no"})
    return pd.DataFrame(filas)


def tabla_parametros(res_cond: dict) -> pd.DataFrame:
    filas = []
    for cond, res in res_cond.items():
        for nombre, a in res.items():
            for j, p in enumerate(a.spec.nombres):
                blo, bhi = a.boot.ci.get(p, (np.nan, np.nan)) if a.boot else (np.nan, np.nan)
                filas.append({"condicion": cond, "modelo": nombre, "parametro": p,
                              "valor": a.popt[j], "SE": a.se[j],
                              "Wald_lo": a.ci_wald[p][0], "Wald_hi": a.ci_wald[p][1],
                              "boot_lo": blo, "boot_hi": bhi,
                              "boot_conv": a.boot.tasa if a.boot else np.nan})
    return pd.DataFrame(filas)


def tabla_derivadas(res_cond: dict, listas: dict, ctx: Ctx, cfg: Config, rng) -> pd.DataFrame:
    filas = []
    lo_q, hi_q = 50 * (1 - cfg.nivel), 50 * (1 + cfg.nivel)
    for cond, res in res_cond.items():
        for nombre in listas.get(cond, []):
            a = res[nombre]
            P, fuente = muestras_incertidumbre(a, cfg, rng)
            puntual = calcular_derivadas(a.spec, a.popt[None, :], ctx)
            dist = calcular_derivadas(a.spec, P, ctx)
            for mag, v in puntual.items():
                x = dist[mag][np.isfinite(dist[mag])]
                lo, hi = np.percentile(x, [lo_q, hi_q]) if len(x) > 20 else (np.nan, np.nan)
                filas.append({"condicion": cond, "modelo": nombre, "magnitud": mag,
                              "estimado": float(v[0]), "lo": lo, "hi": hi, "metodo": fuente})
    return pd.DataFrame(filas)


def comparar_condiciones(res_cond: dict, mejores: dict) -> pd.DataFrame:
    """
    Compara parámetros del MISMO modelo entre condiciones (ajustes independientes).
    z = (θ₂ − θ₁)/√(sd₁² + sd₂²) en escala log/logit; p-valor con t de Student.
    Sin corrección por comparaciones múltiples (usar Holm/Bonferroni si hay muchas).
    """
    filas = []
    por_modelo = {}
    for c, m in mejores.items():
        por_modelo.setdefault(m, []).append(c)
    for m, cs in por_modelo.items():
        if len(cs) < 2:
            continue
        ref = cs[0]
        a0 = res_cond[ref][m]
        for c in cs[1:]:
            a1 = res_cond[c][m]
            gl = max((a0.n - a0.k) + (a1.n - a1.k), 1)
            for j, p in enumerate(a0.spec.nombres):
                z = (a1.theta[j] - a0.theta[j]) / np.hypot(a0.sd_theta[j], a1.sd_theta[j])
                filas.append({"modelo": m, "parametro": p, "cond_ref": ref, "valor_ref": a0.popt[j],
                              "condicion": c, "valor": a1.popt[j],
                              "cambio_%": (a1.popt[j] - a0.popt[j]) / a0.popt[j] * 100,
                              "z": z, "p_valor": 2 * student_t.sf(abs(z), gl)})
    return pd.DataFrame(filas)


def _imprimir(titulo, df, **kw):
    print(f"\n{'═' * 100}\n{titulo}\n{'═' * 100}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}", **kw) if len(df) else "(vacío)")


def reporte_consola(res_cond, mejores, df_par, df_der, df_cmp, cfg):
    for cond, res in res_cond.items():
        _imprimir(f"{cond}: comparación de modelos "
                  f"(ĉ={next((a.c_hat for a in res.values() if np.isfinite(a.c_hat)), np.nan):.2f}; "
                  f"QAICc = AICc si ĉ=1)", tabla_modelos(res, mejores.get(cond)))
    # parámetros de los ganadores, con IC lado a lado
    filas = []
    for cond, m in mejores.items():
        sub = df_par[(df_par.condicion == cond) & (df_par.modelo == m)]
        for _, r in sub.iterrows():
            filas.append({"condicion": cond, "modelo": m, "parametro": r.parametro,
                          "valor ± SE": f"{_fmt(r.valor)} ± {_fmt(r.SE)}",
                          f"IC{cfg.nivel * 100:.0f}% Wald": f"[{_fmt(r.Wald_lo)}, {_fmt(r.Wald_hi)}]",
                          f"IC{cfg.nivel * 100:.0f}% bootstrap":
                              f"[{_fmt(r.boot_lo)}, {_fmt(r.boot_hi)}]" if np.isfinite(r.boot_lo) else "—"})
    _imprimir("Parámetros del modelo seleccionado", pd.DataFrame(filas))

    if len(df_der):
        tabla = df_der.assign(valor=lambda t: [f"{_fmt(e)} [{_fmt(l)}, {_fmt(h)}]"
                                               for e, l, h in zip(t.estimado, t.lo, t.hi)])
        for cond in df_der.condicion.unique():
            piv = (tabla[tabla.condicion == cond]
                   .pivot(index="modelo", columns="magnitud", values="valor").fillna("—"))
            fuente = tabla[tabla.condicion == cond].metodo.unique()
            print(f"\n{'═' * 100}\n{cond}: magnitudes derivadas por modelo candidato "
                  f"(estimado [IC{cfg.nivel * 100:.0f}%]; método: {', '.join(fuente)})\n{'═' * 100}")
            print(piv.to_string())
    if len(df_cmp):
        _imprimir("Comparación de parámetros entre condiciones (mismo modelo)", df_cmp)


def exportar(cfg, res_cond, mejores, df_par, df_der, df_cmp):
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tm = pd.concat([tabla_modelos(r, mejores.get(c)).assign(condicion=c) for c, r in res_cond.items()],
                   ignore_index=True)
    tm = tm[["condicion"] + [c for c in tm.columns if c != "condicion"]]
    tablas = {"modelos": tm, "parametros": df_par, "derivadas": df_der, "entre_condiciones": df_cmp}
    for nombre, df in tablas.items():
        df.to_csv(out / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    # correlación entre parámetros (escala log/logit) del ganador: diagnóstico de identificabilidad
    for c, m in mejores.items():
        a = res_cond[c][m]
        corr = a.cov_theta / np.outer(a.sd_theta, a.sd_theta)
        pd.DataFrame(corr, index=a.spec.nombres, columns=a.spec.nombres).to_csv(
            out / f"correlacion_{c}_{m}.csv", encoding="utf-8-sig")
        print(f"\nCorrelación entre parámetros, {c} / {m}:")
        print(pd.DataFrame(corr, index=a.spec.nombres, columns=a.spec.nombres)
              .to_string(float_format=lambda v: f"{v:6.2f}"))
    try:
        with pd.ExcelWriter(out / "resultados.xlsx") as xw:
            for nombre, df in tablas.items():
                df.to_excel(xw, sheet_name=nombre, index=False)
        print(f"\nTablas exportadas en {out}/ (CSV + resultados.xlsx)")
    except ImportError:
        print(f"\nTablas exportadas en {out}/ (CSV; instalá openpyxl para obtener también el XLSX)")


# ═════════════════════════════════════════════════════════════════════════════
# 8. GRÁFICOS
# ═════════════════════════════════════════════════════════════════════════════
MARCADORES = ["o", "s", "^", "D", "v", "P"]


def _colores(n):
    return plt.cm.tab10(np.linspace(0, 1, max(n, 4)))


def _guardar(fig, nombre, cfg):
    fig.tight_layout()
    ruta = Path(cfg.outdir) / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"  figura: {ruta}")
    if cfg.mostrar:
        plt.show()
    plt.close(fig)


def _barras(s: pd.DataFrame, cfg: Config):
    if cfg.barras == "ic95":
        tc = student_t.ppf(0.975, np.maximum(s.n.to_numpy() - 1, 1))
        f = np.where(s.n.to_numpy() > 1, tc / np.sqrt(s.n.to_numpy()), 0.0)
        return (s.Ce_std.fillna(0) * f).to_numpy(), (s.Qe_std.fillna(0) * f).to_numpy()
    return s.Ce_std.fillna(0).to_numpy(), s.Qe_std.fillna(0).to_numpy()


def _trazar_datos(ax, datos: Datos, cfg, color=None):
    conds = sorted(datos.stats.condicion.unique())
    cols = _colores(len(conds))
    et = "IC 95 % de la media" if cfg.barras == "ic95" else "±1 SD entre falcones"
    for i, c in enumerate(conds):
        s = datos.stats[datos.stats.condicion == c]
        xe, ye = _barras(s, cfg)
        ax.errorbar(s.Ce_mean, s.Qe_mean, xerr=xe, yerr=ye, fmt=MARCADORES[i % 6], ms=7, capsize=4,
                    color=color or cols[i], alpha=0.85, zorder=5, label=f"{c} (datos, {et})")
    return conds, cols


def _etiquetas(ax):
    ax.set_xlabel(f"$C_e$ ({UNIDAD_CE})")
    ax.set_ylabel(f"$q_e$ ({UNIDAD_QE})")
    ax.grid(ls="--", alpha=0.5)


def _banda(aj, Cs, cfg, rng):
    P, fuente = muestras_incertidumbre(aj, cfg, rng)
    with np.errstate(all="ignore"):
        Y = aj.spec.func(Cs[:, None], *[P[:, j][None, :] for j in range(aj.k)])
    lo, hi = np.nanpercentile(Y, [50 * (1 - cfg.nivel), 50 * (1 + cfg.nivel)], axis=1)
    return lo, hi, fuente


def graficar_datos(datos, cfg):
    fig, ax = plt.subplots(figsize=(10, 7))
    _trazar_datos(ax, datos, cfg)
    _etiquetas(ax)
    ax.legend()
    _guardar(fig, "datos.png", cfg)


def graficar_ajustes(datos, res_cond, mejores, cfg, rng):
    fig, ax = plt.subplots(figsize=(12, 8))
    conds, cols = _trazar_datos(ax, datos, cfg)
    Cs = np.linspace(1e-6, datos.stats.Ce_mean.max() * 1.1, 300)
    for i, c in enumerate(conds):
        if c not in mejores:
            continue
        a = res_cond[c][mejores[c]]
        with np.errstate(all="ignore"):
            ax.plot(Cs, a.spec.func(Cs, *a.popt), color=cols[i], lw=2.5, label=f"{c} – {a.nombre}")
        lo, hi, fuente = _banda(a, Cs, cfg, rng)
        ax.fill_between(Cs, lo, hi, color=cols[i], alpha=0.15,
                        label=f"banda {cfg.nivel * 100:.0f}% ({fuente})" if i == 0 else None)
    _etiquetas(ax)
    ax.legend(fontsize=8, ncol=2)
    _guardar(fig, "ajustes.png", cfg)


def graficar_descomposicion(datos, res_cond, mejores, cfg):
    for c, m in mejores.items():
        a = res_cond[c][m]
        if a.spec.componentes is None:
            continue
        Cs = np.linspace(1e-6, datos.stats.Ce_mean.max() * 1.1, 300)
        fig, ax = plt.subplots(figsize=(10, 7))
        _trazar_datos(ax, datos[:0] if False else datos, cfg, color="gray")
        ax.plot(Cs, a.spec.func(Cs, *a.popt), "r-", lw=2.5, label="total")
        for (lbl, y), est in zip(a.spec.componentes(Cs, a.popt), ["b--", "g--"]):
            ax.plot(Cs, y, est, lw=1.6, label=lbl)
        ax.set_title(f"{c} – descomposición {m}")
        _etiquetas(ax)
        ax.legend(fontsize=8)
        _guardar(fig, f"descomposicion_{c.lower()}.png", cfg)


def graficar_residuos(datos, res_cond, mejores, cfg):
    """Residuos estandarizados con varianza efectiva: (y − f(x)) / √(sy² + (f'(x)·sx)²)."""
    if not mejores:
        return
    fig, axes = plt.subplots(1, len(mejores), figsize=(5.5 * len(mejores), 5), squeeze=False)
    for ax, (c, m) in zip(axes[0], mejores.items()):
        a = res_cond[c][m]
        d = datos.ajuste[datos.ajuste.condicion == c]
        x, y, sx, sy = d.Ce.to_numpy(), d.Qe.to_numpy(), d.sx.to_numpy(), d.sy.to_numpy()
        h = 1e-4 * np.maximum(x, 1e-6)
        with np.errstate(all="ignore"):
            yhat = a.spec.func(x, *a.popt)
            dfdx = (a.spec.func(x + h, *a.popt) - a.spec.func(x - h, *a.popt)) / (2 * h)
        z = (y - yhat) / np.sqrt(sy ** 2 + (dfdx * sx) ** 2)
        ax.scatter(yhat, z, alpha=0.75)
        for yy, ls in ((0, "-"), (2, "--"), (-2, "--")):
            ax.axhline(yy, color="red", ls=ls, lw=1)
        ax.set_xlabel(f"$q_e$ predicho ({UNIDAD_QE})")
        ax.set_ylabel("residuo estandarizado")
        ax.set_title(f"{c} – {m}")
        ax.grid(ls="--", alpha=0.5)
    _guardar(fig, "residuos.png", cfg)


def graficar_varianza(datos, cfg):
    conds = sorted(datos.stats.condicion.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cols = _colores(len(conds))
    for ax, var, un in zip(axes, ("Ce", "Qe"), (UNIDAD_CE, UNIDAD_QE)):
        for i, c in enumerate(conds):
            s = datos.stats[(datos.stats.condicion == c) & (datos.stats.n >= 2)]
            ax.loglog(s[f"{var}_mean"], s[f"{var}_std"], MARCADORES[i % 6], color=cols[i],
                      label=f"{c}: SD de cada nivel")
            ab = datos.var_info.get((c, var))
            if ab is not None:
                xs = np.geomspace(s[f"{var}_mean"].min(), s[f"{var}_mean"].max(), 50)
                ax.loglog(xs, np.exp(ab[0]) * xs ** ab[1], "-", color=cols[i],
                          label=f"{c}: SD = {np.exp(ab[0]):.3g}·media^{ab[1]:.2f}")
        ax.set_xlabel(f"media de {var} ({un})")
        ax.set_ylabel(f"SD entre falcones ({un})")
        ax.set_title(f"Modelo de varianza – {var}")
        ax.grid(ls="--", alpha=0.4, which="both")
        ax.legend(fontsize=7)
    _guardar(fig, "modelo_varianza.png", cfg)


# ═════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═════════════════════════════════════════════════════════════════════════════
def ejecutar(cfg: Config):
    if not cfg.mostrar:
        plt.switch_backend("Agg")
    rng = np.random.default_rng(cfg.seed)
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)

    datos = cargar_datos(cfg)
    res_cond = ajustar_todo(datos, cfg, rng)
    if not any(res_cond.values()):
        raise SystemExit("Ningún modelo convergió.")

    print(f"\n{'═' * 100}\nSELECCIÓN DE MODELOS\n{'═' * 100}")
    mejores = seleccionar_modelos(res_cond, cfg)
    if not mejores:
        raise SystemExit("Ningún modelo aceptable.")

    # --- bootstrap solo de los candidatos ---
    listas = {c: candidatos(res_cond[c], mejores.get(c), cfg) for c in res_cond}
    for c, lista in listas.items():
        for nombre in lista:
            a = res_cond[c][nombre]
            a.boot = bootstrap_ajuste(a, datos.ajuste[datos.ajuste.condicion == c], cfg, rng)
            print(f"  bootstrap {c}/{nombre}: {a.boot.tasa:.0%} de {cfg.boot_B} remuestras convergieron")

    dosis = cfg.dosis_mg_L or datos.dosis_mg_L
    print(f"\nDosis usada: {dosis:.2f} mg/L | Co a tratar: {cfg.co_tratar:g} {UNIDAD_CE} | "
          f"límite: {cfg.ce_limite:g} {UNIDAD_CE}")
    ctx = Ctx(cfg.ce_limite, cfg.co_tratar, dosis)
    # derivadas: candidatos si hay bootstrap; si no, al menos el ganador (IC por MVN)
    listas_der = {c: (l if l else [mejores[c]] if c in mejores else []) for c, l in listas.items()}

    df_par = tabla_parametros(res_cond)
    df_der = tabla_derivadas(res_cond, listas_der, ctx, cfg, rng)
    df_cmp = comparar_condiciones(res_cond, mejores)

    reporte_consola(res_cond, mejores, df_par, df_der, df_cmp, cfg)
    exportar(cfg, res_cond, mejores, df_par, df_der, df_cmp)

    print("\nFiguras:")
    graficar_datos(datos, cfg)
    graficar_varianza(datos, cfg)
    graficar_ajustes(datos, res_cond, mejores, cfg, rng)
    graficar_descomposicion(datos, res_cond, mejores, cfg)
    graficar_residuos(datos, res_cond, mejores, cfg)
    print("\nANÁLISIS COMPLETADO")
    return datos, res_cond, mejores


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=Config.csv_path)
    p.add_argument("--outdir", default=Config.outdir)
    p.add_argument("--boot", type=int, default=Config.boot_B, help="remuestras bootstrap (0 = sin bootstrap)")
    p.add_argument("--boot-modo", choices=["estratificado", "parametrico"], default=Config.boot_modo)
    p.add_argument("--boot-cuales", choices=["ninguno", "ganador", "candidatos", "todos"],
                   default=Config.boot_cuales)
    p.add_argument("--varianza", choices=["suavizado", "grupo"], default=Config.modo_varianza)
    p.add_argument("--seleccion", choices=["compartido", "independiente"], default=Config.modo_seleccion)
    p.add_argument("--modelo-fijo", default=None)
    p.add_argument("--modelos", nargs="+", default=list(Config.modelos), choices=list(MODELOS))
    p.add_argument("--barras", choices=["sd", "ic95"], default=Config.barras)
    p.add_argument("--co-tratar", type=float, default=Config.co_tratar)
    p.add_argument("--dosis", type=float, default=None, help="mg/L; por defecto se deduce del CSV")
    p.add_argument("--no-mostrar", action="store_true", help="no abrir ventanas de gráficos")
    a = p.parse_args()
    return Config(csv_path=a.csv, outdir=a.outdir, boot_B=a.boot, boot_modo=a.boot_modo,
                  boot_cuales="ninguno" if a.boot == 0 else a.boot_cuales,
                  modo_varianza=a.varianza, modo_seleccion=a.seleccion, modelo_fijo=a.modelo_fijo,
                  modelos=tuple(a.modelos), barras=a.barras, co_tratar=a.co_tratar,
                  dosis_mg_L=a.dosis, mostrar=not a.no_mostrar)


if __name__ == "__main__":
    ejecutar(_parse_args())