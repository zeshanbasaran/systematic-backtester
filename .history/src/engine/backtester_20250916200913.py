"""
backtester.py
--------------
Core simulation engine for backtesting trading strategies.

- Takes in price data and a position series (signals).
- Simulates portfolio performance with slippage and commissions.
- Outputs equity curve, PnL, returns, and trade log.

Workflow
--------
1. Compute daily (or bar-level) returns from adjusted close prices.
2. Apply target position weights `w` in [-1, 1]:
   * -1 = fully short
   *  0 = flat
   * +1 = fully long
3. Adjust returns for trading costs:
   * Slippage (bps cost when changing positions).
   * Commission (per-share, optional).
4. Build equity curve by compounding strategy returns.
5. Log trades whenever the position changes (dw ≠ 0).
"""

import numpy as np
import pandas as pd


def simulate(
    df: pd.DataFrame,
    pos: pd.Series,
    init_cash: float,
    slippage_bps: float = 1.0,
    comm_per_sh: float = 0.0
) -> dict:
    """
    Run a backtest simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with at least an 'adj_close' column.
    pos : pd.Series
        Target position series, aligned with df.index.
        - If in [-1, 1], interpreted as fraction of equity.
        - Could also be number of shares (future extension).
    init_cash : float
        Initial portfolio cash (starting equity).
    slippage_bps : float, default 1.0
        Slippage cost in basis points per trade.
        (1 bps = 0.01% of notional traded).
    comm_per_sh : float, default 0.0
        Commission per share traded (not fully implemented here).

    Returns
    -------
    dict
        {
          'equity': pd.Series of portfolio equity curve,
          'pnl':    pd.Series of profit & loss,
          'returns':pd.Series of per-period strategy returns,
          'trades': pd.DataFrame log of trades when position changes
        }
    """
    # --- Prices & returns ---
    px = df["adj_close"].astype(float)
    ret = px.pct_change().fillna(0)

    # --- Target weights ---
    w = pos.clip(-1, 1).astype(float)   # positions in [-1, 1]

    # --- Position changes (for trading costs) ---
    dw = w.diff().fillna(w)             # how much position changes each step
    trading_cost = np.abs(dw) * (slippage_bps / 1e4)

    # --- Strategy returns ---
    strat_ret = w * ret - trading_cost

    # --- Equity curve & PnL ---
    equity = (1 + strat_ret).cumprod() * init_cash
    pnl = equity.diff().fillna(equity - init_cash)

    # --- Trade log (when position changes) ---
    trades = pd.DataFrame({
        "timestamp": df.index,
        "target_w": w.values,
        "delta_w": dw.values,
        "price": px.values,
        "cost": trading_cost.values * equity.shift(1).fillna(init_cash).values
    }).query("delta_w != 0")

    return dict(equity=equity, pnl=pnl, returns=strat_ret, trades=trades)
