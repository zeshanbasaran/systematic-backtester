"""
sma_crossover.py
----------------
Simple Moving Average (SMA) crossover trading strategy.

Logic
-----
- Compute two moving averages on the adjusted close price:
    * `short` window (faster, more sensitive)
    * `long` window (slower, smoother)
- Generate a position signal based on their relationship:
    * Long (1) when short SMA > long SMA.
    * Flat (0) otherwise.
    * If `allow_short=True`, Flat (0) is replaced with Short (-1).
- Shift signals forward by 1 bar to simulate realistic execution
  (i.e., trade on the next bar after the signal appears).

Returns
-------
pd.Series of target positions aligned with df.index, with values:
    -1 = short
     0 = flat
    +1 = long
"""

import pandas as pd


def sma_crossover(df: pd.DataFrame, short: int = 50, long: int = 200, allow_short: bool = False) -> pd.Series:
    """
    Generate SMA crossover signals.

    Parameters
    ----------
    df : pd.DataFrame
        Price data containing an 'adj_close' column.
    short : int, default 50
        Lookback window for the short (fast) moving average.
    long : int, default 200
        Lookback window for the long (slow) moving average.
    allow_short : bool, default False
        If True, generates short (-1) positions when not long.
        If False, stays flat (0) instead.

    Returns
    -------
    pd.Series
        Target position series in [-1, 0, 1], shifted by one bar to avoid lookahead bias.
    """
    s = df["adj_close"].rolling(short).mean()
    l = df["adj_close"].rolling(long).mean()

    pos = (s > l).astype(int)  # 1 when short SMA > long SMA, else 0
    if allow_short:
        pos = pos.replace({0: -1})

    # Shift forward to avoid using future information
    return pos.shift(1).fillna(0)
