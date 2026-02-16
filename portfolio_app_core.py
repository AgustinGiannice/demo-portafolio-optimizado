## Imports


# portfolio_optimizer_final.py
# Requisitos: numpy, pandas, cvxpy, scikit-learn
# (NO uses "pip install ..." dentro del .py)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

import matplotlib.pyplot as plt

"""##Carga CSV + imputación/alineación (LINEAL)"""

# ============================================================
# Data Prep: Carga CSV + Alineación + Imputación/Interpolación
# ============================================================

import pandas as pd
import numpy as np

def load_prices_csv(csv_path="precios.csv"):
    """
    Lee el CSV de precios (creado previamente desde Yahoo Finance).
    Espera: index = fechas, columns = tickers, valores = precios.
    """
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Asegurar orden temporal y que el índice sea datetime
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)

    # Convertir a float por si quedó como object/string
    prices = prices.apply(pd.to_numeric, errors="coerce")

    return prices


def align_and_impute_prices(prices, freq="B", ffill_limit=5, interp_limit=3):
    """
    - Restringe a ventana de solapamiento común entre tickers
    - Reindexa a calendario business days (L-V)
    - Imputa faltantes (ffill limitado + interpolación temporal limitada)
    Retorna: prices_final, missing_report (antes/después)
    """
    prices = prices.copy()

    # Ventana común (evita NaNs grandes al inicio/fin por activos con historia corta)
    first_valid = prices.apply(lambda s: s.first_valid_index())
    last_valid  = prices.apply(lambda s: s.last_valid_index())
    start_common = first_valid.max()
    end_common   = last_valid.min()

    if pd.isna(start_common) or pd.isna(end_common) or start_common >= end_common:
        raise ValueError("No hay solapamiento temporal suficiente entre los tickers para alinear series.")

    prices = prices.loc[start_common:end_common]

    # Calendario común (L-V; incluye feriados como días hábiles -> ahí aparecen NaNs y se imputan)
    idx = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq=freq)
    prices = prices.reindex(idx)

    missing_before = prices.isna().mean().sort_values(ascending=False)

    # 1) forward fill limitado (huecos cortos como feriados)
    prices_ff = prices.ffill(limit=ffill_limit)

    # 2) interpolación temporal limitada (por si quedan huecos internos)
    prices_int = prices_ff.interpolate(method="time", limit=interp_limit)

    # 3) limpieza final (si queda algo al inicio, backfill + ffill)
    prices_final = prices_int.bfill().ffill()

    missing_after = prices_final.isna().mean().sort_values(ascending=False)

    report = pd.DataFrame({
        "missing_before": missing_before,
        "missing_after": missing_after
    })

    # Si aún quedan NaNs, eliminamos filas problemáticas (conservador)
    if prices_final.isna().any().any():
        prices_final = prices_final.dropna(axis=0, how="any")

    return prices_final, report

"""## optimizador"""

# ============================================================
# 0) Utilidades de escala (ANNUAL <-> DAILY)
# ============================================================

def annual_return_to_daily(r_annual: float, periods_per_year: int = 252) -> float:
    """Convierte retorno anual (decimal) -> daily (decimal)."""
    return (1.0 + r_annual) ** (1.0 / periods_per_year) - 1.0


def annual_vol_to_daily(vol_annual: float, periods_per_year: int = 252) -> float:
    """Convierte volatilidad anual (decimal) -> daily (decimal)."""
    return vol_annual / np.sqrt(periods_per_year)


def daily_return_to_annual(r_daily: float, periods_per_year: int = 252) -> float:
    return (1.0 + r_daily) ** periods_per_year - 1.0


def daily_vol_to_annual(vol_daily: float, periods_per_year: int = 252) -> float:
    return vol_daily * np.sqrt(periods_per_year)


def make_psd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Simetriza y agrega jitter a la diagonal para estabilidad numérica."""
    S = 0.5 * (S + S.T)
    return S + eps * np.eye(S.shape[0])


# ============================================================
# 1) VaR / RoI (CONSISTENTES con tu pipeline: Close-to-Close)
# ============================================================

def VaR_from_prices(price_series: pd.Series, alpha: float = 0.05, as_positive_loss: bool = True) -> float:
    """
    VaR histórico (close-to-close daily). Usa cola izquierda (alpha=0.05 típico).
    Devuelve:
      - as_positive_loss=True: VaR como pérdida positiva (ej 0.02)
      - as_positive_loss=False: cuantil (ej -0.02)
    """
    r = price_series.pct_change().dropna()
    q = float(r.quantile(alpha))
    return float(-q) if as_positive_loss else q


def RoI_from_prices(price_series: pd.Series, days: int = 365) -> float:
    """
    RoI promedio a horizonte 'days' usando close-to-close:
      RoI = mean( P_t / P_{t-days} - 1 )
    OJO: esto es para reporting/inputs, NO mezclar con Σ daily dentro del optimizador.
    """
    r_h = price_series.pct_change(periods=days).dropna()
    return float(r_h.mean())


# ============================================================
# 2) Pipeline mínimo (si ya lo tenés, podés saltearlo)
# ============================================================

def prices_to_returns(prices_aligned: pd.DataFrame) -> pd.DataFrame:
    """Retornos daily close-to-close."""
    return prices_aligned.pct_change().dropna()


def temporal_split(returns: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal por ratio (sin shuffle)."""
    n = len(returns)
    cut = int(np.floor(train_ratio * n))
    return returns.iloc[:cut].copy(), returns.iloc[cut:].copy()


# ============================================================
# 3) Estimación Σ (GANADOR: Ledoit–Wolf)
# ============================================================

def get_sigma_hat_shrink_lw(returns_train: pd.DataFrame) -> np.ndarray:
    """
    Ledoit–Wolf sobre retornos daily.
    Output: Σ (N x N) daily PSD.
    """
    lw = LedoitWolf().fit(returns_train.values)
    return make_psd(lw.covariance_)


# ============================================================
# 4) Optimización FINAL (convexa, sin binarias, daily)
# ============================================================

@dataclass
class OptimizationResult:
    status: str
    tickers: List[str]
    weights: Optional[np.ndarray]
    amounts: Optional[np.ndarray]
    # métricas
    ret_daily: Optional[float]
    vol_daily: Optional[float]
    ret_annual: Optional[float]
    vol_annual: Optional[float]
    # slacks
    slack_risk: Optional[float]
    slack_return: Optional[float]
    # debug/checks
    checks: Dict[str, float]


def _diver_to_wmax(
    diver_pct: float,
    n_assets: int,
    k_min: int = 5,
    k_max: int = 25,
    curve: float = 1.5
) -> Tuple[int, float]:
    """
    ✅ NUEVO (IMPORTANTE):
    diver 0..100 controla k en un rango razonable [k_min, k_max] (NO depende de N enorme).

    - diver=0  -> k≈k_min (pocos activos)
    - diver=100-> k≈k_max (más activos)
    - curve>1  -> crecimiento más “suave” (diver alto recién empuja fuerte cerca del final)

    wmax = 1/k.
    """
    diver = float(np.clip(diver_pct / 100.0, 0.0, 1.0))
    diver = diver ** float(curve)

    k_max_eff = int(min(max(1, k_max), n_assets))
    k_min_eff = int(min(max(1, k_min), k_max_eff))

    k = int(round(k_min_eff + diver * (k_max_eff - k_min_eff)))
    k = max(1, min(n_assets, k))

    wmax = 1.0 / k
    return k, wmax


def _sqrt_psd_matrix(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Devuelve L tal que aprox S = L.T @ L (via eigen)."""
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 0.0, None)
    L = (np.sqrt(vals + eps)[:, None] * vecs.T)
    return L


def _pick_conic_solver() -> str:
    installed = set(cp.installed_solvers())
    for s in ["CLARABEL", "SCS", "ECOS"]:
        if s in installed:
            return s
    raise RuntimeError(
        f"No hay solvers cónicos instalados. Instalados: {sorted(installed)}. "
        f"Instalá alguno: pip install ecos  (o) pip install scs  (o) pip install clarabel"
    )


def optimize_portfolio(
    returns_train: pd.DataFrame,
    tickers: List[str],
    budget: float,
    mxr: float,
    exr: float,
    ponder: float = 50.0,
    diver: float = 50.0,
    inputs_are_percent: bool = True,
    inputs_are_annual: bool = True,
    periods_per_year: int = 252,
    risk_slack_penalty: float = 5_000.0,
    ret_slack_penalty: float = 5_000.0,
    # ✅ NUEVO: regularización hacia equiponderado (mejora estabilidad OOS)
    gamma_eq: float = 2.0,
    # ✅ NUEVO: control de rango de holdings para diver
    k_min: int = 5,
    k_max: int = 25,
    diver_curve: float = 1.5,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Problema convexo SOCP:

    Variables:
      w >= 0, sum(w)=1
      s1>=0 (slack riesgo), s2>=0 (slack retorno)

    Restricciones:
      w <= wmax (diversificación convexa por max-weight)
      mu'w >= r_min - s2
      ||L w||_2 <= sigma_max + s1   (forma DCP-correcta)

    Objetivo:
      maximizar retorno - lam*var - gamma_eq*||w-w_eq||^2 - penalizaciones de slacks
    """
    if not set(tickers).issubset(set(returns_train.columns)):
        missing = sorted(set(tickers) - set(returns_train.columns))
        raise ValueError(f"Tickers faltantes en returns_train: {missing}")

    R = returns_train[tickers].dropna()
    if R.shape[0] < 60:
        raise ValueError("Muy pocos datos en returns_train para estimar Σ y μ de forma estable.")

    mu = R.mean().values  # daily
    Sigma = get_sigma_hat_shrink_lw(R)  # daily LW
    Sigma = make_psd(Sigma)

    N = len(tickers)
    k, wmax = _diver_to_wmax(diver, N, k_min=k_min, k_max=k_max, curve=diver_curve)

    mxr_val = float(mxr)
    exr_val = float(exr)
    if inputs_are_percent:
        mxr_val /= 100.0
        exr_val /= 100.0

    # Targets en DAILY dentro del optimizador (si UI es anual)
    if inputs_are_annual:
        sigma_max = annual_vol_to_daily(mxr_val, periods_per_year)
        r_min = annual_return_to_daily(exr_val, periods_per_year)
    else:
        sigma_max = mxr_val
        r_min = exr_val

    lam = float(np.clip(ponder / 100.0, 0.0, 1.0))

    w = cp.Variable(N)
    s1 = cp.Variable(nonneg=True)
    s2 = cp.Variable(nonneg=True)

    # Forma SOCP DCP-correcta
    L = _sqrt_psd_matrix(np.asarray(Sigma))
    risk = cp.norm(L @ w, 2)
    var = cp.sum_squares(L @ w)

    constraints = [
        w >= 0,
        cp.sum(w) == 1,
        w <= wmax,
        mu @ w >= r_min - s2,
        risk <= sigma_max + s1,
    ]

    # Normalización de escalas
    var_scale = float(np.mean(np.diag(Sigma))) if float(np.mean(np.diag(Sigma))) > 0 else 1.0
    ret_scale = float(np.mean(np.abs(mu))) if float(np.mean(np.abs(mu))) > 0 else 1.0

    # Regularización hacia equal-weight (mejora OOS)
    w_eq = np.ones(N) / N

    objective = cp.Maximize(
        (mu @ w) / ret_scale
        - lam * (var / var_scale)
        - float(gamma_eq) * cp.sum_squares(w - w_eq)
        - risk_slack_penalty * s1
        - ret_slack_penalty * s2
    )

    prob = cp.Problem(objective, constraints)

    if solver is None:
        solver = _pick_conic_solver()

    prob.solve(solver=solver, verbose=verbose)

    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        return OptimizationResult(
            status=status,
            tickers=tickers,
            weights=None,
            amounts=None,
            ret_daily=None,
            vol_daily=None,
            ret_annual=None,
            vol_annual=None,
            slack_risk=None,
            slack_return=None,
            checks={"k_target": float(k), "wmax": float(wmax)},
        )

    wv = np.asarray(w.value).reshape(-1)
    wv = np.maximum(wv, 0.0)
    wv = wv / wv.sum()

    ret_d = float(mu @ wv)
    var_d = float(np.sum((L @ wv) ** 2))
    vol_d = float(np.sqrt(max(var_d, 0.0)))

    ret_a = daily_return_to_annual(ret_d, periods_per_year)
    vol_a = daily_vol_to_annual(vol_d, periods_per_year)

    s1v = float(s1.value) if s1.value is not None else 0.0
    s2v = float(s2.value) if s2.value is not None else 0.0

    amounts = wv * float(budget)

    # Métricas de diversificación útiles
    n_eff = float(1.0 / np.sum(wv ** 2))  # número efectivo
    n_active = float(np.sum(wv > 1e-4))   # pesos “no despreciables”

    checks = {
        "solver_used": solver,
        "k_target": float(k),
        "wmax": float(wmax),
        "sum_w": float(wv.sum()),
        "min_w": float(wv.min()),
        "max_w": float(wv.max()),
        "n_eff": n_eff,
        "n_active": n_active,
        "ret_daily": ret_d,
        "vol_daily": vol_d,
        "ret_target_daily": float(r_min),
        "vol_target_daily": float(sigma_max),
        "slack_risk": s1v,
        "slack_return": s2v,
        "risk_ok_soft": float(vol_d <= sigma_max + s1v + 1e-6),
        "ret_ok_soft": float(ret_d + 1e-6 >= r_min - s2v),
        "gamma_eq": float(gamma_eq),
        "diver_curve": float(diver_curve),
        "k_min": float(k_min),
        "k_max": float(k_max),
    }

    return OptimizationResult(
        status=status,
        tickers=tickers,
        weights=wv,
        amounts=amounts,
        ret_daily=ret_d,
        vol_daily=vol_d,
        ret_annual=ret_a,
        vol_annual=vol_a,
        slack_risk=s1v,
        slack_return=s2v,
        checks=checks,
    )


# ============================================================
# 5) Validación OOS (Sharpe vs equal-weight y benchmark opcional)
# ============================================================

def portfolio_returns(returns: pd.DataFrame, tickers: List[str], weights: np.ndarray) -> pd.Series:
    R = returns[tickers].dropna()
    w = np.asarray(weights).reshape(-1)
    return pd.Series(R.values @ w, index=R.index, name="portfolio")


def sharpe_daily(r: pd.Series, eps: float = 1e-12) -> float:
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    return mu / (sd + eps)


def validate_oos(
    returns_test: pd.DataFrame,
    tickers: List[str],
    weights: np.ndarray,
    benchmark_ticker: Optional[str] = "^GSPC",
) -> Dict[str, float]:
    pr = portfolio_returns(returns_test, tickers, weights)
    sr_p = sharpe_daily(pr)

    w_eq = np.ones(len(tickers)) / len(tickers)
    pr_eq = portfolio_returns(returns_test, tickers, w_eq)
    sr_eq = sharpe_daily(pr_eq)

    out = {"sharpe_portfolio_daily": sr_p, "sharpe_equal_daily": sr_eq}

    if benchmark_ticker is not None and benchmark_ticker in returns_test.columns:
        sr_b = sharpe_daily(returns_test[benchmark_ticker].dropna())
        out["sharpe_benchmark_daily"] = sr_b

    return out

"""## visualizacion"""

# -------------------------
# Helpers (métricas simples, cliente-friendly)
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def _equity_curve(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return (1.0 + r).cumprod()

def _max_drawdown_from_values(v: pd.Series) -> float:
    v = v.dropna()
    if len(v) == 0:
        return np.nan
    dd = v / v.cummax() - 1.0
    return float(dd.min())

def _cagr_from_returns(r: pd.Series, ppy: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    eq = _equity_curve(r)
    years = len(r) / ppy
    if years <= 0:
        return np.nan
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)

def _var_cvar_loss(r: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    VaR/CVaR como pérdida positiva:
    - VaR 5%: “1 de cada 20 días peores, podrías perder ~X% o más”
    - CVaR 5%: “promedio de esos días muy malos”
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan
    q = float(r.quantile(alpha))  # típicamente negativo
    cvar = float(r[r <= q].mean()) if (r <= q).any() else np.nan
    return float(-q), float(-cvar)

def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"

def _round_amounts_to_budget(amounts: pd.Series, total_budget: float, round_to: int = 10) -> pd.Series:
    round_to = max(int(round_to), 1)
    a = (amounts / round_to).round() * round_to
    a = a.astype(float)

    diff = float(total_budget - a.sum())
    if len(a) > 0:
        a.iloc[-1] = float(a.iloc[-1] + diff)

    if (a < 0).any():
        a[a < 0] = 0.0
        diff2 = float(total_budget - a.sum())
        a.iloc[0] = float(a.iloc[0] + diff2)

    return a


# -------------------------
# Dashboard final (simple)
# -------------------------
def run_portfolio_app(
    prices_aligned: pd.DataFrame,
    budget: float,
    max_risk_pct: float,          # vol anual % (input del usuario)
    min_return_pct: float,        # retorno anual mínimo % (input del usuario)
    diversification_pct: float,   # diver 0..100
    risk_return_pct: float,       # ponder 0..100
    investment_days: int = 252,
    tickers: list[str] | None = None,
    benchmark_ticker: str = "^GSPC",
    train_ratio: float = 0.7,

    # knobs del optimizador
    gamma_eq: float = 0.0,
    k_min: int = 5,
    k_max: int = 10,
    diver_curve: float = 1.5,

    # UX
    active_weight_threshold: float = 0.001,
    cash_buffer_pct: float = 0.00,
    round_to_dollars: int = 10,

    # salida
    show_plots: bool = True,
    show_compare: bool = False,
    print_summary: bool = True,
):
    """
    Salida pensada para alguien no técnico:
    - Tabla de compra (qué comprar y cuánto)
    - “En 1 año podrías tener aprox: $X (Y%)” (CAGR OOS)
    - Drawdown (explicado)
    - Gráficos simples
    """

    if budget <= 0:
        raise ValueError("budget debe ser > 0")
    cash_buffer_pct = float(np.clip(cash_buffer_pct, 0.0, 0.20))

    investable_budget = float(budget * (1.0 - cash_buffer_pct))
    cash_reserved = float(budget - investable_budget)

    # returns + split
    returns = prices_to_returns(prices_aligned)
    returns_train, returns_test = temporal_split(returns, train_ratio=train_ratio)

    if tickers is None:
        tickers = [c for c in returns_train.columns if c != benchmark_ticker]
    else:
        tickers = [t for t in tickers if t in returns_train.columns and t != benchmark_ticker]

    if len(tickers) < 2:
        raise ValueError("Necesitás al menos 2 tickers válidos para armar el portafolio.")

    # Optimización (TRAIN)
    res = optimize_portfolio(
        returns_train=returns_train,
        tickers=tickers,
        budget=investable_budget,
        mxr=max_risk_pct,
        exr=min_return_pct,
        ponder=risk_return_pct,
        diver=diversification_pct,
        inputs_are_percent=True,
        inputs_are_annual=True,
        gamma_eq=gamma_eq,
        k_min=k_min,
        k_max=k_max,
        diver_curve=diver_curve,
    )
    if res.weights is None:
        raise RuntimeError(f"Optimización falló. Status={res.status}")

    # Evaluación: últimos N días del TEST
    test_slice = returns_test.copy()
    if investment_days is not None and investment_days > 0:
        test_slice = test_slice.iloc[-min(investment_days, len(test_slice)):]
    test_slice = test_slice.dropna(how="all")

    r_port = portfolio_returns(test_slice, tickers, res.weights)

    # (Opcional) comparativos
    w_eq = np.ones(len(tickers)) / len(tickers)
    r_eq = portfolio_returns(test_slice, tickers, w_eq)
    r_bench = test_slice[benchmark_ticker].dropna() if benchmark_ticker in test_slice.columns else None

    # Curva de valor en $
    v_port = investable_budget * _equity_curve(r_port) + cash_reserved
    v_eq   = investable_budget * _equity_curve(r_eq) + cash_reserved
    v_bench = (investable_budget * _equity_curve(r_bench) + cash_reserved) if r_bench is not None else None

    # Métricas simples
    cagr = _cagr_from_returns(r_port)
    expected_1y_value = investable_budget * (1.0 + (0.0 if np.isnan(cagr) else cagr)) + cash_reserved
    expected_1y_gain  = expected_1y_value - budget

    mdd = _max_drawdown_from_values(v_port)  # negativo
    varL, cvarL = _var_cvar_loss(r_port, 0.05)

    # Asignación
    alloc = (
        pd.DataFrame({"Ticker": tickers, "Peso": res.weights, "Monto ($)": res.amounts})
        .sort_values("Peso", ascending=False)
        .reset_index(drop=True)
    )
    alloc_active = alloc[alloc["Peso"] >= float(active_weight_threshold)].copy()
    n_holdings = int(len(alloc_active))

    # Lista de compra (redondeada, suma exacta)
    buy = alloc_active.copy()
    buy["Monto ($)"] = _round_amounts_to_budget(buy["Monto ($)"], investable_budget, round_to=round_to_dollars)
    buy["Peso (%)"] = (100 * buy["Monto ($)"] / investable_budget).round(2)
    buy = buy[["Ticker", "Peso (%)", "Monto ($)"]].sort_values("Monto ($)", ascending=False).reset_index(drop=True)

    # Tabla “resumen simple”
    summary_simple = pd.DataFrame({
        "Dato": [
            "Presupuesto total",
            "Monto a invertir",
            "Cash reservado",
            "Cantidad de acciones (donde realmente invertís)",
            "En 1 año podrías tener aprox (estimación)",
            "Ganancia/pérdida estimada a 1 año",
            "Peor caída desde un pico (Drawdown máx.)",
            "Día malo (VaR 5%)",
            "Promedio de días muy malos (CVaR 5%)",
        ],
        "Valor": [
            _fmt_money(budget),
            _fmt_money(investable_budget),
            _fmt_money(cash_reserved),
            f"{n_holdings}",
            f"{_fmt_money(expected_1y_value)} ({'' if np.isnan(cagr) else f'{cagr:.2%}'})",
            f"{_fmt_money(expected_1y_gain)}",
            "" if np.isnan(mdd) else f"{mdd:.2%}",
            "" if np.isnan(varL) else f"{varL:.2%}  (~{_fmt_money(investable_budget * varL)})",
            "" if np.isnan(cvarL) else f"{cvarL:.2%} (~{_fmt_money(investable_budget * cvarL)})",
        ]
    })

    if print_summary:
        print("✅ Recomendación lista")
        print(f"- Presupuesto: {_fmt_money(budget)} | Invertís: {_fmt_money(investable_budget)} | Acciones: {n_holdings}")
        if not np.isnan(cagr):
            print(f"- Estimación (según histórico OOS): en 1 año → {_fmt_money(expected_1y_value)}  ({cagr:.2%})")
        if not np.isnan(mdd):
            print(f"- Drawdown máx: {mdd:.2%} (peor caída desde un pico durante el período evaluado)")
        print("")

    # Mostrar tablas
    display(summary_simple)

    print("🧾 Lista de compra (cuánto invertir en cada empresa):")
    buy_show = buy.copy()
    buy_show["Monto ($)"] = buy_show["Monto ($)"].map(lambda x: _fmt_money(float(x)))
    display(buy_show)

    # -------------------------
    # Gráficos
    # -------------------------
    if show_plots:
        # 1) Evolución del valor del portfolio
        plt.figure(figsize=(12, 4))
        plt.plot(v_port.index, v_port.values, label="Tu portfolio", linewidth=2.2)
        plt.title("Evolución del valor del portfolio ($)")
        plt.ylabel("USD")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2) Drawdown
        dd = v_port / v_port.cummax() - 1.0
        plt.figure(figsize=(12, 3.5))
        plt.plot(dd.index, dd.values, linewidth=2)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Caídas temporales (Drawdown) — cuánto bajó desde el último máximo")
        plt.ylabel("Caída")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        # 3) Asignación recomendada en $
        # (TODAS las acciones donde ponés plata, no top 15)
        if len(buy) > 0:
            n = len(buy)
            fig_h = max(4.0, min(0.35 * n, 18.0))  # altura dinámica
            plt.figure(figsize=(12, fig_h))
            plt.barh(buy["Ticker"][::-1], buy["Monto ($)"][::-1])
            plt.title("Asignación recomendada ($) — todas las posiciones")
            plt.xlabel("USD")
            plt.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()

        # 4) Comparación (opcional)
        if show_compare and (v_bench is not None):
            base_port = 100 * (v_port / v_port.iloc[0])
            base_eq   = 100 * (v_eq / v_eq.iloc[0])
            base_b    = 100 * (v_bench / v_bench.iloc[0])

            plt.figure(figsize=(12, 4))
            plt.plot(base_port.index, base_port.values, label="Tu portfolio", linewidth=2.2)
            plt.plot(base_eq.index, base_eq.values, label="Equal-weight(mismo peso a todos)", alpha=0.85)
            plt.plot(base_b.index, base_b.values, label="Benchmark(mercado)", alpha=0.85)
            plt.title("Comparación)")
            plt.ylabel("Índice (100 = inicio)")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return {
        "result": res,
        "tables": {
            "summary_simple": summary_simple,
            "buy_list": buy,
        },
        "series": {
            "value_port": v_port,
            "value_equal": v_eq,
            "value_bench": v_bench,
            "returns_port": r_port,
            "returns_equal": r_eq,
            "returns_bench": r_bench
        },
        "alloc": {
            "alloc_active": alloc_active,
            "alloc_full": alloc
        }
    }



