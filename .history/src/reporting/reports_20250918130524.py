"""
reports.py
----------
Reporting utilities for backtest results.

Responsibilities
---------------
- Compute summary performance & risk metrics.
- Export an Excel report with multiple sheets:
    * Summary (key metrics)
    * EquityCurve
    * Returns
    * Drawdown
    * Trades
- Save PNG charts (equity, drawdown, returns histogram).
- Optionally insert charts into the Summary sheet.

Notes
-----
- Assumes `equity`, `returns`, and `dd_series` are all aligned to the same index.
- Uses metrics from engine/risk modules (imported below).
- Attempts to infer periods_per_year from the index cadence; defaults to 252 if unclear.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Adjust these imports to your package layout.
# If this file lives in src/reporting/reports.py and your project uses a package "src",
# relative imports would look like:
# from ..engine.metrics import ann_return, ann_vol, sharpe, max_drawdown
# from ..risk.risk_metrics import var_historical

# If you're running without packages, fall back to absolute names.
try:
    from ..engine.metrics import ann_return, ann_vol, sharpe, max_drawdown
    from ..risk.risk_metrics import var_historical
except Exception:
    from src.engine.metrics import ann_return, ann_vol, sharpe, max_drawdown  # type: ignore
    from src.risk.risk_metrics import var_historical  # type: ignore


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """
    Infer number of periods per year from a DatetimeIndex.

    Heuristic:
    - If cadence >= ~12h, treat as daily business bars -> 252.
    - Else compute based on median delta (seconds) -> round(365d / delta).
    """
    if not isinstance(index, pd.DatetimeIndex) or index.size < 3:
        return 252

    # Median spacing
    deltas = np.diff(index.view("i8"))  # nanoseconds
    if len(deltas) == 0:
        return 252

    med_ns = float(np.median(deltas))
    sec = med_ns / 1e9
    if sec >= 12 * 3600:
        return 252

    periods = int(round((365 * 24 * 3600) / sec))
    # Keep in a reasonable range
    return max(52, min(periods, 24 * 365))


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_equity(equity: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_drawdown(dd_series: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(10, 3.5))
    plt.fill_between(dd_series.index, dd_series.values, 0.0, step=None)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_returns_hist(returns: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(returns.dropna().values, bins=50)
    plt.title("Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_performance_report(
    equity: pd.Series,
    returns: pd.Series,
    dd_series: pd.Series,
    trades: pd.DataFrame,
    out_xlsx: str,
    out_png_dir: str,
) -> dict:
    """
    Build a full performance report:
      1) Compute summary metrics
      2) Write Excel with multiple sheets
      3) Save charts as PNGs
      4) Insert charts into Summary sheet (optional, if writer engine supports it)

    Parameters
    ----------
    equity : pd.Series
        Portfolio equity curve over time.
    returns : pd.Series
        Per-period strategy returns aligned to equity index.
    dd_series : pd.Series
        Drawdown series (same index as equity).
    trades : pd.DataFrame
        Trade log with columns like: ['timestamp','target_w','delta_w','price','cost'].
    out_xlsx : str
        Output Excel file path (e.g., "reports/perf_report.xlsx").
    out_png_dir : str
        Directory to write PNG charts.

    Returns
    -------
    dict
        {
          "summary": pd.DataFrame,  # single-row metrics table
          "charts": { "equity": Path, "drawdown": Path, "returns_hist": Path }
        }
    """
    # --- Preflight & alignment ---
    if not isinstance(equity.index, pd.DatetimeIndex):
        equity.index = pd.to_datetime(equity.index)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    if not isinstance(dd_series.index, pd.DatetimeIndex):
        dd_series.index = pd.to_datetime(dd_series.index)

    # Align series just in case
    df_all = pd.concat(
        {
            "equity": equity,
            "returns": returns,
            "drawdown": dd_series,
        },
        axis=1,
    ).dropna(subset=["equity"])  # keep equity as anchor

    equity = df_all["equity"]
    returns = df_all["returns"].fillna(0.0)
    dd_series = df_all["drawdown"].fillna(method="ffill").fillna(0.0)

    # --- Metrics ---
    ppy = _infer_periods_per_year(equity.index)
    ar = ann_return(returns, ppy)
    av = ann_vol(returns, ppy)
    sh = sharpe(returns, ppy, rf=0.0)
    dd_val, _ = max_drawdown(equity)
    var95 = var_historical(returns, 0.95)

    summary = pd.DataFrame(
        {
            "annual_return": [ar],
            "annual_vol": [av],
            "sharpe": [sh],
            "max_drawdown": [dd_val],
            "var_95": [var95],
            "periods_per_year": [ppy],
            "start": [equity.index.min()],
            "end": [equity.index.max()],
            "bars": [equity.size],
        }
    )

    # --- Charts ---
    charts_dir = _ensure_dir(out_png_dir)
    equity_png = charts_dir / "equity_curve.png"
    dd_png = charts_dir / "drawdown.png"
    ret_png = charts_dir / "returns_hist.png"

    _plot_equity(equity, equity_png)
    _plot_drawdown(dd_series, dd_png)
    _plot_returns_hist(returns, ret_png)

    # --- Excel export ---
    out_xlsx_path = Path(out_xlsx)
    _ensure_dir(out_xlsx_path.parent)

    with pd.ExcelWriter(out_xlsx_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        # Summary
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # EquityCurve
        equity.to_frame("equity").to_excel(writer, sheet_name="EquityCurve")

        # Returns
        returns.to_frame("return").to_excel(writer, sheet_name="Returns")

        # Drawdown
        dd_series.to_frame("drawdown").to_excel(writer, sheet_name="Drawdown")

        # Trades (if provided)
        if trades is not None and len(trades) > 0:
            # Standardize timestamp index if needed
            tdf = trades.copy()
            if "timestamp" in tdf.columns:
                tdf = tdf.sort_values("timestamp")
            tdf.to_excel(writer, sheet_name="Trades", index=False)

        # Try inserting images in Summary sheet
        try:
            workbook = writer.book
            worksheet = writer.sheets["Summary"]
            # Insert images with some spacing (row, col, options)
            worksheet.insert_image("H2", str(equity_png))
            worksheet.insert_image("H22", str(dd_png))
            worksheet.insert_image("H42", str(ret_png))
        except Exception:
            # If anything goes wrong (e.g., different engine), just skip embedding
            pass

    return {"summary": summary, "charts": {"equity": equity_png, "drawdown": dd_png, "returns_hist": ret_png}}
