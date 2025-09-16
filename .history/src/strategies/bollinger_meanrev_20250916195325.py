"""
bollinger_meanrev.py
--------------------
Bollinger Bands mean-reversion trading strategy.

Logic
-----
- Compute Bollinger Bands:
    * Middle band = moving average of adjusted close.
    * Standard deviation = rolling volatility measure.
    * Z-score = (price - mean) / std.
- Entry rules:
    * Go long (+1) when price is below -k standard deviations.
    * Go short (-1) when price is above +k standard deviations (if allow_short=True).
- Exit rules:
    * Close positions when price reverts toward the mean, i.e. |z| < band_exit.
- Position persistence:
    * Uses forward fill (`ffill`) to hold positions until exit.
    * Signals are shifted by 1 bar to avoid lookahead bias.

Returns
-------
pd.Series of target positions aligned with df.index, values in [-1, 0, 1]:
    -1 = short
     0 = flat
    +1 = long
"""

import pandas as pd


def bollinger_meanrev(
    df: pd.DataFrame,
    lookback: int = 20,
    k: float = 2.0,
    band_exit: float = 0.5,
    allow_short: bool = True
) -> pd.Series:
    """
    Generate Bollinger Bands mean-reversion signals.

    Parameters
    ----------
    df : pd.DataFrame
        Price data containing an 'adj_close' column.
    lookback : int, default 20
        Rolling window size for moving average and standard deviation.
    k : float, default 2.0
        Number of standard deviations for entry bands.
    band_exit : float, default 0.5
        Threshold (in z-score units) to flatten positions when price reverts.
    allow_short : bool, default True
        If True, enables short entries when price > +k std.

    Returns
    -------
    pd.Series
        Target position series in [-1, 0, 1], shifted by one bar.
    """
    # rolling mean and std
    m = df["adj_close"].rolling(lookback).mean()
    sd = df["adj_close"].rolling(lookback).std(ddof=0)

    # standardized z-score of price relative to mean
    z = (df["adj_close"] - m) / sd

    # start with flat positions
    pos = pd.Series(0.0, index=df.index)

    # long entry: below -k std deviations
    pos = pos.where(~(z < -k), 1.0)

    # short entry: above +k std deviations
    if allow_short:
        pos = pos.where(~(z > k), -1.0)

    # exit: flatten when z returns inside ±band_exit
    pos = pos.where(~(z.between(-band_exit, band_exit)), 0.0)

    # hold positions until exit, shift to avoid lookahead
    return pos.ffill().shift(1).fillna(0)
