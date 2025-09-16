"""
base.py
-------
Core abstractions for the trading system.

- `Signal`: A lightweight data structure representing a single trading
  instruction (time, side, and size).
- `Strategy`: Abstract base class for trading strategies.
   * Each strategy should subclass this and implement `generate_signals()`
     to output position instructions over time.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class Signal:
    """
    Represents a single trading signal.

    Attributes
    ----------
    timestamp : pd.Timestamp
        The time the signal is generated or acted upon.
    side : str
        The trade direction: "BUY", "SELL", or "FLAT" (no position).
    size : float
        Target position size:
          - Could be units of the asset (e.g., number of shares).
          - Or a fraction of equity exposure (e.g., 0.5 = 50% of portfolio).
    """
    timestamp: pd.Timestamp
    side: str    # "BUY"/"SELL"/"FLAT"
    size: float  # target position in units or fraction of equity


class Strategy:
    """
    Abstract base class for all trading strategies.

    Attributes
    ----------
    name : str
        Human-readable name of the strategy (e.g., "SMA Crossover").

    Methods
    -------
    generate_signals(df: pd.DataFrame) -> pd.Series
        Must be implemented by subclasses.
        Should return a pandas Series indexed by df.index,
        with values in [-1, 1] representing:
          -1 = fully short
           0 = flat / no position
          +1 = fully long
    """
    name: str

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target position series in [-1, 1] aligned to df.index."""
        raise NotImplementedError("Subclasses must implement generate_signals()")
