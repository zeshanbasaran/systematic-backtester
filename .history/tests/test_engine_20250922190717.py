"""
test_engine.py
--------------
Unit tests for the simulation engine, metrics, and risk monitor.

Run with:
    pytest tests/test_engine.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# --- ensure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Engine
from src.engine.backtester import simulate
from src.engine.metrics import ann_return, ann_vol, sharpe, max_drawdown

# Risk
from src.risk.monitor import check_breaches, RiskThresholds


# ----------------------------
# Helpers
# ----------------------------

def make_price_df(vals, start="2024-01-01", freq="D"):
    """Build a minimal OHLCV frame around an adj_close path."""
    idx = pd.date_range(start=start, periods=len(vals), freq=freq)
    ac = pd.Series(vals, index=idx, dtype=float)
    # For completeness, mirror adj_close to OHLC; set volume=1
    df = pd.DataFrame(
        {
            "open": ac.values,
            "high": ac.values,
            "low": ac.values,
            "close": ac.values,
            "adj_close": ac.values,
            "volume": 1.0,
        },
        index=idx,
    )
    return df


# ----------------------------
# Backtester: simulate()
# ----------------------------

def test_simulate_long_only_increasing_prices():
    # Prices: strictly up → positive returns when long
    df = make_price_df([100, 101, 102, 103, 104])
    # Always long after the first bar (shift usually happens in strategy; we emulate here)
    pos = pd.Series([0, 1, 1, 1, 1], index=df.index, dtype=float)

    out = simulate(df, pos, init_cash=100_000, slippage_bps=0.0)
    equity, returns, trades = out["equity"], out["returns"], out["trades"]

    assert len(equity) == len(df)
    assert len(returns) == len(df)
    assert equity.iloc[-1] > equity.iloc[0]  # grew
    # We should have one trade when entering long (delta from 0->1)
    assert (np.sign(trades["delta_w"]) == 1).sum() == 1


def test_simulate_slippage_cost_applied():
    # Three steps: flat -> long -> flat
    # Day 0: 100, Day 1: 100 (no price change), Day 2: 100 (no change)
    df = make_price_df([100, 100, 100])
    pos = pd.Series([0.0, 1.0, 0.0], index=df.index)
    # Set slippage to 50 bps = 0.50% per unit of |Δw|
    out = simulate(df, pos, init_cash=10_000, slippage_bps=50.0)
    returns = out["returns"]

    # With no price change, the only PnL impact is trading cost.
    # Δw at t=0: +1.0  → cost = 0.005
    # Δw at t=1: -1.0  → cost = 0.005
    # So both t=0 and t=1 should have negative returns equal to -0.5%.
    # (Implementation applies cost per step based on |Δw|.)
    assert pytest.approx(returns.iloc[0], rel=1e-9, abs=1e-9) == -0.005
    assert pytest.approx(returns.iloc[1], rel=1e-9, abs=1e-9) == -0.005
    # Final bar has Δw=0 and no price move → zero return
    assert returns.iloc[2] == 0.0


# ----------------------------
# Metrics
# ----------------------------

def test_metrics_shapes_and_values():
    # Simple return series: +1%, 0%, -1%, +2%
    r = pd.Series([0.01, 0.0, -0.01, 0.02])
    ppy = 252

    ar = ann_return(r, ppy)
    av = ann_vol(r, ppy)
    sh = sharpe(r, ppy, rf=0.0)

    assert np.isfinite(ar)
    assert av > 0
    assert np.isfinite(sh)

    # Zero-variance returns → Sharpe should be NaN (by implementation)
    r_flat = pd.Series([0.001] * 10)
    sh_flat = sharpe(r_flat, ppy)
    assert np.isnan(sh_flat)


def test_max_drawdown_values():
    # Equity path with a clear 20% drawdown: 100 → 120 → 96 → 110
    e = pd.Series([100, 120, 96, 110], index=pd.date_range("2024-01-01", periods=4))
    dd_min, dd_series = max_drawdown(e)
    # Worst is from 120 down to 96: (96/120 - 1) = -0.2
    assert pytest.approx(dd_min) == -0.20
    assert len(dd_series) == len(e)
    # At the peak (t=1), drawdown is 0
    assert dd_series.iloc[1] == 0.0
    # At trough (t=2), drawdown is -20%
    assert pytest.approx(dd_series.iloc[2]) == -0.20


# ----------------------------
# Risk monitor
# ----------------------------

def test_check_breaches_flags_expected():
    # Build a series of modestly volatile returns that will trip VaR and vol,
    # and an equity path that trips max drawdown.
    idx = pd.date_range("2024-01-01", periods=6)
    returns = pd.Series([0.0, -0.03, 0.01, -0.04, 0.0, 0.02], index=idx)
    equity = (1 + returns.fillna(0)).cumprod() * 100_000  # synthetic equity
    ppy = 252

    thr = RiskThresholds(max_dd=0.02, var_95=0.015, vol_ann=0.10)
    breaches, stats = check_breaches(equity, returns, ppy, thr)

    # Expect at least one breach; specific set may include MAX_DD, VAR_95, and/or VOL
    codes = {code for code, _ in breaches}
    assert len(breaches) >= 1
    assert {"MAX_DD", "VAR_95", "VOL"}.intersection(codes)

    # Stats should include the requested fields
    for k in ("dd", "var95", "vol", "dd_series"):
        assert k in stats
