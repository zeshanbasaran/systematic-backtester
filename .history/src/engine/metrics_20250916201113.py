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
    return (1 + returns).prod()**(periods_per_year / returns.size) - 1


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
    return returns.std(ddof=0) * np.sqrt(periods_per_year)


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
    excess = returns - (rf / periods_per_year)
    return excess.mean() / excess.std(ddof=0) * np.sqrt(periods_per_year)


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
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min(), dd
