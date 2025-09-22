"""
metrics.py
----------
Performance and risk metrics for backtesting results.

Functions
---------
- ann_return : Annualized return based on compounded growth.
- ann_vol    : Annualized volatility (std of returns).
- sharpe     : Sharpe ratio (risk-adjusted return).
- max_drawdown : Maximum portfolio drawdown and drawdown series.

These metrics are used to evaluate strategy performance
after running simulations in `backtester.py`.
"""

import numpy as np
import pandas as pd


def ann_return(returns: pd.Series, periods_per_year: int) -> float:
    """
    Annualized return (CAGR).

    Parameters
    ----------
    returns : pd.Series
        Strategy returns per period (not cumulative).
    periods_per_year : int
        Number of return observations per year (e.g., 252 for daily, 12 for monthly).

    Returns
    -------
    float
        Annualized return as a decimal (e.g., 0.12 = 12%).
    """
    r = pd.Series(returns).dropna()
    n = r.size
    if n == 0:
        return float("nan")
    growth = (1.0 + r).prod()
    if growth <= 0:
        return float("nan")
    return growth ** (periods_per_year / n) - 1.0


def ann_vol(returns: pd.Series, periods_per_year: int) -> float:
    """
    Annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns per period.
    periods_per_year : int
        Number of return observations per year.

    Returns
    -------
    float
        Annualized standard deviation of returns.
    """
    r = pd.Series(returns).dropna()
    if r.size == 0:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    """
    Sharpe ratio (risk-adjusted return).

    Parameters
    ----------
    returns : pd.Series
        Strategy returns per period.
    periods_per_year : int
        Number of return observations per year.
    rf : float, default 0.0
        Risk-free rate (annualized).

    Returns
    -------
    float
        Sharpe ratio.
    """
    r = pd.Series(returns).dropna()
    if r.size == 0:
        return float("nan")
    excess = r - (rf / periods_per_year)
    s = excess.std(ddof=0)
    if s == 0 or np.isnan(s):
        return float("nan")
    return float(excess.mean() / s * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    """
    Maximum drawdown.

    Parameters
    ----------
    equity : pd.Series
        Equity curve (portfolio value over time).

    Returns
    -------
    tuple
        (max_drawdown, drawdown_series)
        max_drawdown : float (worst % drop from peak)
        drawdown_series : pd.Series of drawdowns over time
    """
    e = pd.Series(equity).dropna()
    if e.size == 0:
        dd = pd.Series(dtype=float, index=equity.index if hasattr(equity, "index") else None)
        return 0.0, dd  # no drawdown
    roll_max = e.cummax()
    dd = e / roll_max - 1.0
    return float(dd.min()), dd.reindex(equity.index) if hasattr(equity, "index") else dd
