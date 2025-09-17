"""
monitor.py
----------
Risk monitoring for backtested strategies.

- Defines thresholds for acceptable risk levels.
- Checks portfolio performance against those thresholds.
- Reports any breaches (alerts).

Uses metrics from `metrics.py` and `risk_metrics.py`:
- max_drawdown
- ann_vol
- var_historical
"""

from dataclasses import dataclass
# Assumes these functions are imported elsewhere in the project:
# from metrics import ann_vol, max_drawdown
# from risk_metrics import var_historical


@dataclass
class RiskThresholds:
    """
    Configuration for risk limits.

    Attributes
    ----------
    max_dd : float
        Maximum allowed drawdown (e.g., 0.20 = 20%).
    var_95 : float
        Maximum 95% Value-at-Risk.
    vol_ann : float
        Maximum annualized volatility.
    """
    max_dd: float
    var_95: float
    vol_ann: float


def check_breaches(equity, returns, periods_per_year, thr: RiskThresholds):
    """
    Check whether risk metrics exceed thresholds.

    Parameters
    ----------
    equity : pd.Series
        Equity curve (portfolio value over time).
    returns : pd.Series
        Strategy returns per period.
    periods_per_year : int
        Number of return observations per year (e.g., 252 for daily).
    thr : RiskThresholds
        Risk limits to monitor.

    Returns
    -------
    tuple
        breaches : list of (metric_name, value)
            List of all risk metrics that exceeded thresholds.
        stats : dict
            {
              'dd' : float, worst drawdown,
              'var95' : float, 95% historical VaR,
              'vol' : float, annualized volatility,
              'dd_series' : pd.Series of drawdowns over time
            }
    """
    dd_min, dd_series = max_drawdown(equity)
    vol = ann_vol(returns, periods_per_year)
    var95 = var_historical(returns, 0.95)

    breaches = []
    if dd_min < -thr.max_dd:  # drawdown worse than allowed
        breaches.append(("MAX_DD", dd_min))
    if var95 > thr.var_95:    # VaR worse than allowed
        breaches.append(("VAR_95", var95))
    if vol > thr.vol_ann:     # volatility too high
        breaches.append(("VOL", vol))

    return breaches, {"dd": dd_min, "var95": var95, "vol": vol, "dd_series": dd_series}
