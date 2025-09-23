"""
app_streamlit.py
----------------
Interactive Streamlit dashboard for the systematic backtester.

Features
--------
- Sidebar controls:
    * Strategy selection (SMA crossover, Bollinger mean-reversion)
    * Symbol, date range, bar size
    * Strategy params (short/long windows, Bollinger k, lookback)
    * Trading costs (slippage, commission)
    * Risk thresholds (max drawdown, vol, VaR)

- Main tabs:
    1. Overview: KPI cards + equity/drawdown charts
    2. Risk: rolling volatility, VaR, and breach alerts
    3. Trades: trade table, CSV download, overlay on price chart
    4. PnL: daily returns histogram, rolling Sharpe ratio
    5. DB: run metadata (from SQLite/MySQL/Postgres)

Usage
-----
Run from project root:
    streamlit run dashboards/app_streamlit.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure `src` package is importable when running via `streamlit run dashboards/app_streamlit.py`
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Project imports ---
from src.data.loaders import get_price_data
from src.strategies.sma_crossover import sma_crossover
from src.strategies.bollinger_meanrev import bollinger_meanrev
from src.engine.backtester import simulate
from src.engine.metrics import sharpe, max_drawdown, ann_return, ann_vol
from src.risk.monitor import check_breaches, RiskThresholds
from src.config import INIT_CASH, SLIPPAGE_BPS, COMM_PER_TRADE, START, END, BAR, RISK


# ----------------------------
# Sidebar controls
# ----------------------------
st.set_page_config(page_title="Systematic Backtester", layout="wide")

st.sidebar.header("Backtest Controls")

strategy = st.sidebar.selectbox("Strategy", ["SMA", "Bollinger"])
symbol = st.sidebar.text_input("Symbol", "SPY")
start = pd.to_datetime(st.sidebar.date_input("Start Date", pd.to_datetime(START)))
end = pd.to_datetime(st.sidebar.date_input("End Date", pd.to_datetime(END)))
bar = st.sidebar.selectbox("Bar Size", ["1d", "1h"], index=0)
df = get_price_data(symbol, start, end, bar)


# Strategy-specific params
if strategy == "SMA":
    short = st.sidebar.number_input("Short Window", min_value=5, value=50, step=5)
    long = st.sidebar.number_input("Long Window", min_value=20, value=200, step=10)
    allow_short = st.sidebar.checkbox("Allow Short", value=False)
elif strategy == "Bollinger":
    lookback = st.sidebar.number_input("Lookback", min_value=10, value=20, step=5)
    k = st.sidebar.number_input("Band Width (k)", min_value=1.0, value=2.0, step=0.5)
    band_exit = st.sidebar.number_input("Exit Band", min_value=0.1, value=0.5, step=0.1)
    allow_short = st.sidebar.checkbox("Allow Short", value=True)

# Costs
slippage = st.sidebar.number_input("Slippage (bps)", min_value=0.0, value=SLIPPAGE_BPS, step=0.5)
comm = st.sidebar.number_input("Commission per trade ($/share)", min_value=0.0, value=COMM_PER_TRADE, step=0.01)

# Risk thresholds
st.sidebar.subheader("Risk Thresholds")
thr = RiskThresholds(
    max_dd=st.sidebar.number_input("Max Drawdown", value=RISK["max_dd"]),
    var_95=st.sidebar.number_input("VaR 95%", value=RISK["var_95"]),
    vol_ann=st.sidebar.number_input("Annual Vol", value=RISK["vol_ann"]),
)

# ----------------------------
# Run backtest
# ----------------------------
df = get_price_data(symbol, str(start), str(end), bar)

# Strategy signals
if strategy == "SMA":
    pos = sma_crossover(df, short=short, long=long, allow_short=allow_short)
else:
    pos = bollinger_meanrev(df, lookback=lookback, k=k, band_exit=band_exit, allow_short=allow_short)

# Backtest simulation
bt = simulate(df, pos, INIT_CASH, slippage_bps=slippage, comm_per_sh=comm)
equity, returns, trades = bt["equity"], bt["returns"], bt["trades"]

# Guard: stop early if no data/returns
if returns.dropna().empty or equity.dropna().empty:
    import streamlit as st
    st.warning("No data/returns for the selected inputs. Try a different date range, symbol, or bar size.")
    st.stop()

# Risk checks
ppy = 252 if bar == "1d" else int(252 * 6.5)
breaches, stats = check_breaches(equity, returns, ppy, thr)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risk", "Trades", "PnL", "DB"])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("Overview")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return", f"{ann_return(returns, ppy):.2%}")
    col2.metric("Volatility", f"{ann_vol(returns, ppy):.2%}")
    col3.metric("Sharpe", f"{sharpe(returns, ppy):.2f}")
    col4.metric("Max Drawdown", f"{stats['dd']:.2%}")

    # Equity curve
    st.line_chart(equity)

    # Drawdown chart
    st.area_chart(stats["dd_series"])

# --- Tab 2: Risk ---
with tab2:
    st.subheader("Risk Monitoring")

    # nicer labels & limits
    labels = {"MAX_DD": "Max Drawdown", "VAR_95": "VaR 95% (1-day)", "VOL": "Annual Vol"}
    limits = {"MAX_DD": thr.max_dd, "VAR_95": thr.var_95, "VOL": thr.vol_ann}
    pct = lambda x: f"{x*100:.2f}%"

    # breach banner(s)
    if breaches:
        for metric, value in breaches:
            limit = limits[metric]
            limit_txt = f"-{limit*100:.2f}%" if metric == "MAX_DD" else pct(limit)
            st.error(
                f"**{labels[metric]} breached** — value {pct(value)} vs limit {limit_txt}",
                icon="⚠️",
            )
    else:
        st.success("No risk limits breached.", icon="✅")

    # numeric margins vs limits (positive = within limit, negative = breach)
    dd     = float(stats["dd"])           # drawdown is negative, use abs for compare
    var95  = float(stats["var95"])
    vol    = float(stats["vol"])

    dd_lim = float(thr.max_dd)
    var_lim= float(thr.var_95)
    vol_lim= float(thr.vol_ann)

    # margins in percentage points to the limit
    dd_delta  = (dd_lim - abs(dd)) * 100.0    # < 0 => breach (red ↓)
    var_delta = (var_lim - var95) * 100.0     # < 0 => breach (red ↓)
    vol_delta = (vol_lim - vol) * 100.0       # < 0 => breach (red ↓)

    c1, c2, c3 = st.columns(3)
    c1.metric("Max Drawdown", pct(dd),  delta=f"{dd_delta:.2f} pp to limit")
    c1.caption(f"limit {pct(-dd_lim)}")  # show as a negative percentage

    c2.metric("VaR 95% (1-day)", pct(var95), delta=f"{var_delta:.2f} pp to limit")
    c2.caption(f"limit {pct(var_lim)}")

    c3.metric("Annual Vol", pct(vol),   delta=f"{vol_delta:.2f} pp to limit")
    c3.caption(f"limit {pct(vol_lim)}")

    # drawdown series for context
    st.caption("Drawdown over time")
    st.area_chart(stats["dd_series"].rename("drawdown"))

# --- Tab 3: Trades ---
with tab3:
    st.subheader("Trades Log")
    st.dataframe(trades)
    st.download_button(
        "Download Trades CSV", trades.to_csv(index=False), file_name=f"{symbol}_{strategy}_trades.csv"
    )

# --- Tab 4: PnL ---
with tab4:
    st.subheader("PnL Analysis")

    # Returns histogram
    fig, ax = plt.subplots(figsize=(1, 1))  # width=4, height=3 inches
    ax.hist(returns.dropna(), bins=50)
    ax.set_title("Returns Distribution")
    st.pyplot(fig)

    # Rolling Sharpe
    roll_sharpe = returns.rolling(60).mean() / returns.rolling(60).std()
    st.line_chart(roll_sharpe)

# --- Tab 5: DB ---
with tab5:
    st.subheader("Database (future)")
    st.info("Here you could query past runs from the SQL database and display metadata.")
