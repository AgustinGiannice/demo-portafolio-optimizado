from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from portfolio_app_core import align_and_impute_prices, load_prices_csv

try:
    from portfolio_app_core import run_portfolio_app as _run_core
except ImportError:
    from portfolio_app_core import run_portfolio_app_simple as _run_core


@dataclass(frozen=True)
class UiParams:
    budget: float
    max_risk_pct: float
    min_return_pct: float
    diversification_pct: float
    risk_return_pct: float
    investment_days: int
    show_compare: bool


def _inject_base_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
          section.main > div { max-width: 980px; margin-left: auto; margin-right: auto; }
          .block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
          .app-card { background: white; border-radius: 12px; padding: 28px 28px 22px 28px; box-shadow: 0 8px 28px rgba(0,0,0,0.18); }
          .app-title { font-size: 2.3rem; font-weight: 800; margin: 0 0 10px 0; color: #2b2b2b; }
          .app-subtitle { color: #667; margin: 0 0 22px 0; font-size: 0.95rem; }
          div[data-testid="stMetricValue"] { font-size: 1.55rem; }
          .stDataFrame { border-radius: 10px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_prices_aligned(csv_path: str, csv_mtime: float) -> pd.DataFrame:
    prices = load_prices_csv(csv_path)
    prices_aligned, _ = align_and_impute_prices(prices)
    return prices_aligned


def _read_universe(prices_aligned: pd.DataFrame, benchmark_ticker: str = "^GSPC") -> list[str]:
    cols = [c for c in prices_aligned.columns if isinstance(c, str)]
    return [c for c in cols if c != benchmark_ticker]


def _sidebar_inputs() -> UiParams:
    st.sidebar.header("Investment Parameters")

    budget = st.sidebar.number_input("Budget ($)", min_value=1.0, value=100_000.0, step=1_000.0, format="%.0f")
    max_risk_pct = st.sidebar.number_input(
        "Maximum Risk (annual vol, %)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.25,
        format="%.2f",
    )
    min_return_pct = st.sidebar.number_input(
        "Minimum Return (annual, %)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.25,
        format="%.2f",
    )
    diversification_pct = st.sidebar.slider("Diversification Level (0–100)", min_value=0, max_value=100, value=100, step=1)
    risk_return_pct = st.sidebar.slider("Risk–Return Ponderation (0–100)", min_value=0, max_value=100, value=50, step=1)
    investment_days = int(st.sidebar.number_input("Investment period (days)", min_value=20, value=365, step=5))

    st.sidebar.divider()
    show_compare = st.sidebar.checkbox("Mostrar comparación vs Benchmark", value=True)

    return UiParams(
        budget=float(budget),
        max_risk_pct=float(max_risk_pct),
        min_return_pct=float(min_return_pct),
        diversification_pct=float(diversification_pct),
        risk_return_pct=float(risk_return_pct),
        investment_days=investment_days,
        show_compare=bool(show_compare),
    )


def _money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.2%}"


def _call_core(func, prices_aligned: pd.DataFrame, params: UiParams) -> dict:
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())

    base = {
        "prices_aligned": prices_aligned,
        "budget": params.budget,
        "max_risk_pct": params.max_risk_pct,
        "min_return_pct": params.min_return_pct,
        "diversification_pct": params.diversification_pct,
        "risk_return_pct": params.risk_return_pct,
        "investment_days": params.investment_days,
        "tickers": None,
        "benchmark_ticker": "^GSPC",
        "show_plots": False,
        "print_summary": False,
        "show_compare": params.show_compare,
    }

    if "prices_aligned" not in accepted:
        if "prices" in accepted:
            base["prices"] = base.pop("prices_aligned")
        elif "prices_df" in accepted:
            base["prices_df"] = base.pop("prices_aligned")

    call_kwargs = {k: v for k, v in base.items() if k in accepted}
    return func(**call_kwargs)


def _metrics_from_returns(r: pd.Series | None, n_assets: float | int | None, ppy: int = 252) -> dict:
    if r is None:
        return {
            "CAGR": np.nan,
            "Vol anual": np.nan,
            "Sharpe": np.nan,
            "Máx Drawdown": np.nan,
            "VaR 5% diario": np.nan,
            "CVaR 5% diario": np.nan,
            "Nº acciones": n_assets,
        }

    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {
            "CAGR": np.nan,
            "Vol anual": np.nan,
            "Sharpe": np.nan,
            "Máx Drawdown": np.nan,
            "VaR 5% diario": np.nan,
            "CVaR 5% diario": np.nan,
            "Nº acciones": n_assets,
        }

    eq = (1.0 + r).cumprod()
    years = len(r) / ppy
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan

    vol = float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) > 1 else np.nan
    mu_ann = float(r.mean() * ppy)
    sharpe = float(mu_ann / vol) if vol and not np.isnan(vol) and vol != 0 else np.nan

    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min()) if len(dd) else np.nan

    q = float(r.quantile(0.05))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else np.nan
    var_loss = float(-q)
    cvar_loss = float(-cvar) if not np.isnan(cvar) else np.nan

    return {
        "CAGR": cagr,
        "Vol anual": vol,
        "Sharpe": sharpe,
        "Máx Drawdown": mdd,
        "VaR 5% diario": var_loss,
        "CVaR 5% diario": cvar_loss,
        "Nº acciones": n_assets,
    }


def _build_metrics_table(out: dict, universe: list[str]) -> pd.DataFrame:
    series = out.get("series", {}) if isinstance(out, dict) else {}
    r_port = series.get("returns_port")
    r_eq = series.get("returns_equal")
    r_b = series.get("returns_bench")

    alloc = out.get("alloc", {}) if isinstance(out, dict) else {}
    alloc_active = alloc.get("alloc_active")
    buy = (out.get("tables", {}) or {}).get("buy_list")

    n_relevant = (
        int(len(alloc_active))
        if isinstance(alloc_active, pd.DataFrame)
        else int(len(buy)) if isinstance(buy, pd.DataFrame) else np.nan
    )

    n_assets = (
        int(len(alloc.get("alloc_full")))
        if isinstance(alloc.get("alloc_full"), pd.DataFrame)
        else int(len(universe))
    )

    rows = {
        "Portafolio": _metrics_from_returns(r_port, n_relevant),
        "Equal-weight": _metrics_from_returns(r_eq, n_assets),
        "Benchmark": _metrics_from_returns(r_b if isinstance(r_b, pd.Series) else None, 1 if r_b is not None else np.nan),
    }
    return pd.DataFrame(rows).reset_index().rename(columns={"index": "Métrica"})


def _render_summary_simple(df: pd.DataFrame) -> None:
    show = df.copy()
    cols = list(show.columns)

    if len(cols) >= 2:
        show = show.rename(columns={cols[0]: "Concepto", cols[1]: "Valor"})
        st.dataframe(show.style.hide(axis="index"), use_container_width=True)
        return

    st.dataframe(show.style.hide(axis="index"), use_container_width=True)


def _render_buy_table(buy: pd.DataFrame, budget: float) -> None:
    buy_show = buy.copy()

    if "Monto ($)" in buy_show.columns:
        buy_show["Monto ($)"] = buy_show["Monto ($)"].astype(float)
    if "Peso (%)" in buy_show.columns:
        buy_show["Peso (%)"] = buy_show["Peso (%)"].astype(float)

    total = float(buy_show["Monto ($)"].sum()) if "Monto ($)" in buy_show.columns and len(buy_show) else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Budget", _money(budget))
    c2.metric("Total asignado", _money(total))
    c3.metric("Budget Utilized", f"{(total / budget):.2%}" if budget > 0 else "")

    formats = {}
    if "Peso (%)" in buy_show.columns:
        formats["Peso (%)"] = "{:.2f}"
    if "Monto ($)" in buy_show.columns:
        formats["Monto ($)"] = "${:,.0f}"

    st.dataframe(buy_show.style.format(formats).hide(axis="index"), use_container_width=True)


def _plot_value_curve(value_series: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(value_series.index, value_series.values, linewidth=2.2)
    ax.set_title("Evolución del valor del portafolio ($)")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def _plot_comparison(v_port: pd.Series, v_eq: pd.Series, v_bench: pd.Series | None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    if len(v_port):
        ax.plot(v_port.index, 100.0 * v_port / float(v_port.iloc[0]), label="Portafolio", linewidth=2.2)
    if len(v_eq):
        ax.plot(v_eq.index, 100.0 * v_eq / float(v_eq.iloc[0]), label="Equal-weight", alpha=0.9)
    if v_bench is not None and len(v_bench):
        ax.plot(v_bench.index, 100.0 * v_bench / float(v_bench.iloc[0]), label="Benchmark", alpha=0.9)
    ax.set_title("Portafolio vs Equal-weight vs Benchmark (base 100)")
    ax.set_ylabel("Índice (100 = inicio)")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def _plot_drawdown(value_series: pd.Series) -> None:
    dd = value_series / value_series.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(dd.index, dd.values, linewidth=2.0)
    ax.axhline(0.0, linewidth=0.9)
    ax.set_title("Drawdown del portafolio")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def _plot_allocation_all(buy: pd.DataFrame) -> None:
    if not isinstance(buy, pd.DataFrame) or len(buy) == 0:
        st.info("No hay asignación para graficar.")
        return

    if "Ticker" not in buy.columns or "Monto ($)" not in buy.columns:
        st.info("La tabla de compra no tiene columnas esperadas para graficar.")
        return

    show = buy.copy()
    height = max(4.0, 0.28 * max(len(show), 10))
    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(show["Ticker"][::-1], show["Monto ($)"][::-1])
    ax.set_title("Asignación recomendada en $ (todas las posiciones)")
    ax.set_xlabel("USD")
    ax.grid(alpha=0.25, axis="x")
    st.pyplot(fig, clear_figure=True)


def _safe_metric(df: pd.DataFrame, metric: str, col: str) -> float:
    try:
        v = df.set_index("Métrica").loc[metric, col]
        return float(v) if not pd.isna(v) else np.nan
    except Exception:
        return np.nan


def _render_client_summary(out: dict, universe: list[str]) -> None:
    df = _build_metrics_table(out, universe)

    ret = _safe_metric(df, "CAGR", "Portafolio")
    vol = _safe_metric(df, "Vol anual", "Portafolio")
    dd = _safe_metric(df, "Máx Drawdown", "Portafolio")

    st.subheader("Resumen para decidir (simple)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendimiento anual histórico", _pct(ret))
    c2.metric("Altibajos típicos", _pct(vol))
    c3.metric("Peor caída histórica", _pct(dd))

    st.caption("Valores históricos. No garantizan rendimientos futuros.")

    st.divider()

    st.subheader("Comparación (solo lo esencial)")
    view = pd.DataFrame(
        {
            "Indicador": ["Rendimiento anual", "Altibajos", "Peor caída"],
            "Tu portafolio": [
                _safe_metric(df, "CAGR", "Portafolio"),
                _safe_metric(df, "Vol anual", "Portafolio"),
                _safe_metric(df, "Máx Drawdown", "Portafolio"),
            ],
            "Alternativa simple": [
                _safe_metric(df, "CAGR", "Equal-weight"),
                _safe_metric(df, "Vol anual", "Equal-weight"),
                _safe_metric(df, "Máx Drawdown", "Equal-weight"),
            ],
            "Mercado": [
                _safe_metric(df, "CAGR", "Benchmark"),
                _safe_metric(df, "Vol anual", "Benchmark"),
                _safe_metric(df, "Máx Drawdown", "Benchmark"),
            ],
        }
    )

    for col in ["Tu portafolio", "Alternativa simple", "Mercado"]:
        view[col] = ["" if pd.isna(x) else f"{float(x):.2%}" for x in view[col]]

    st.dataframe(view.style.hide(axis="index"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Portfolio Optimization", layout="wide")
    _inject_base_css()

    csv_path = "precios.csv"
    if not os.path.exists(csv_path):
        st.error("No se encontró precios.csv en el directorio de la app.")
        st.stop()

    prices_aligned = _load_prices_aligned(csv_path, os.path.getmtime(csv_path))
    universe = _read_universe(prices_aligned, benchmark_ticker="^GSPC")

    params = _sidebar_inputs()

    st.sidebar.header("Stock Tickers")
    with st.sidebar.expander("Ver tickers (desde precios.csv)", expanded=False):
        st.write(", ".join(universe) if universe else "—")

    run_clicked = st.sidebar.button("Optimize Portfolio", use_container_width=True)

    st.markdown(
        """
        <div class="app-card">
          <div class="app-title">Portfolio Optimization</div>
          <div class="app-subtitle">Inputs en el sidebar • Resultados claros • Sin tecnicismos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "out" not in st.session_state:
        st.session_state.out = None

    if run_clicked:
        with st.spinner("Optimizando portafolio..."):
            try:
                st.session_state.out = _call_core(_run_core, prices_aligned, params)
            except Exception as e:
                st.session_state.out = None
                st.error(str(e))

    out = st.session_state.out
    if out is None:
        st.markdown(
            """
            <div class="app-card" style="margin-top: 18px;">
              <b>Listo.</b> Ajustá los parámetros en el sidebar y presioná <b>Optimize Portfolio</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    series = out.get("series", {})
    tables = out.get("tables", {})
    alloc = out.get("alloc", {})

    v_port = series.get("value_port")
    v_eq = series.get("value_equal")
    v_b = series.get("value_bench")
    buy = tables.get("buy_list")
    summary_simple = tables.get("summary_simple")

    st.markdown('<div class="app-card" style="margin-top: 18px;">', unsafe_allow_html=True)

    if isinstance(summary_simple, pd.DataFrame) and len(summary_simple):
        st.subheader("Resumen")
        _render_summary_simple(summary_simple)
        st.divider()
    else:
        metrics_df = _build_metrics_table(out, universe)
        port_metrics = metrics_df.set_index("Métrica")["Portafolio"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR (Portafolio)", _pct(float(port_metrics.loc["CAGR"])))
        c2.metric("Vol anual (Portafolio)", _pct(float(port_metrics.loc["Vol anual"])))
        c3.metric("Sharpe (Portafolio)", "" if pd.isna(port_metrics.loc["Sharpe"]) else f"{float(port_metrics.loc['Sharpe']):.2f}")
        c4.metric("Nº acciones relevantes", "" if pd.isna(port_metrics.loc["Nº acciones"]) else f"{int(port_metrics.loc['Nº acciones'])}")
        st.divider()

    st.subheader("Tabla de compra recomendada")
    if isinstance(buy, pd.DataFrame) and len(buy):
        _render_buy_table(buy, params.budget)
    else:
        st.info("No se generó una tabla de compra. Revisá restricciones o universo.")

    st.divider()
    st.subheader("Gráficos")
    if isinstance(v_port, pd.Series) and len(v_port):
        _plot_value_curve(v_port)

    if params.show_compare and isinstance(v_port, pd.Series) and isinstance(v_eq, pd.Series):
        _plot_comparison(v_port, v_eq, v_b if isinstance(v_b, pd.Series) else None)

    if isinstance(v_port, pd.Series) and len(v_port):
        _plot_drawdown(v_port)

    if isinstance(buy, pd.DataFrame) and len(buy):
        _plot_allocation_all(buy)

    st.divider()
    _render_client_summary(out, universe)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
