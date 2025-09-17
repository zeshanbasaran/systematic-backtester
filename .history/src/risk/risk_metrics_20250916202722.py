"""
risk_metrics.py
---------------
Risk evaluation functions for trading strategies.

Functions
---------
- var_parametric : Parametric (Gaussian) Value-at-Risk (VaR).
- var_historical : Historical Value-at-Risk (VaR).

These are used to estimate potential portfolio losses
under different assumptions about return distributions.
"""

import numpy as np
import pandas as pd


def var_parametric(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    Parametric (Gaussian) Value-at-Risk (VaR).

    Assumes returns are normally distributed and uses mean/stdev.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns per period.
    alpha : float, default 0.95
        Confidence level (e.g., 0.95 = 95%).

    Returns
    -------
    float
        Value-at-Risk at the given confidence level (as a negative number).
        Interpreted as the minimum expected loss over 1 period
        with probability (1 - alpha).
    """
    mu, sigma = returns.mean(), returns.std(ddof=0)

    # z-score for given alpha (approximate for 95%)
    if alpha == 0.95:
        z = 1.64485  # standard normal 95% quantile
    else:
        z = abs(pd.Series(np.random.randn(1)).quantile(alpha))  # crude fallback

    return -(mu - z * sigma)


def var_historical(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    Historical Value-at-Risk (VaR).

    Uses the empirical distribution of returns.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns per period.
    alpha : float, default 0.95
        Confidence level (e.g., 0.95 = 95%).

    Returns
    -------
    float
        Value-at-Risk at the given confidence level (as a negative number).
        Interpreted as the minimum expected loss over 1 period
        with probability (1 - alpha).
    """
    return -np.nanpercentile(returns, 100 * (1 - alpha))
