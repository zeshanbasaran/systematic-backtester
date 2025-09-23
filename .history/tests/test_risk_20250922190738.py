"""
test_risk.py
------------
Unit tests for risk metrics and the risk monitor.

Run with:
    pytest tests/test_risk.py -v
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

from src.risk.risk_metrics import var_parametric, var_historical
from src.risk.monitor import RiskThresholds, check_breaches


# ----------------------------
# VaR functions
# ----------------------------

def test_var_parametric_basic():
    # Zero-mean returns with known std → VaR ≈ z * sigma for alpha=0.95
    rng = np.random.default_rng(42)
    sigma = 0.01  # 1% per-period
    r = pd.Series(rng.normal(0.0, sigma, size=10_000))

    v = var_parametric(r, alpha=0.95)
    # Implementation uses z=1.64485 for 95%
    assert v > 0
    assert pytest.approx(v, rel=0.05) == 1.64485 * sigma


def test_var_historical_lower_tail():
    # Construct a simple empirical distribution: mostly small returns, one big loss
    r = pd.Series([0.002, 0.001, 0.0, -0.015, 0.001, 0.002, 0.001, 0.0, 0.001, 0.002])
    # 5% quantile (alpha=0.95) sits near the worst observation; VaR is positive magnitude of loss
    v = var_historical(r, alpha=0.95)
    assert v > 0
    # Worst loss is -1.5%; with such a small sample the 5% quantile ~ that tail value
    assert 0.010 <= v <= 0.020


# ----------------------------
# Risk monitor
# ----------------------------

def test_check_breaches_all_three_trip():
    # Build an equity/returns set that will exceed all thresholds:
    # - Large drawdown (MAX_DD)
    # - High volatility (VOL)
    # - Large lower-tail loss (VAR_95)

    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    # Returns include two sharp down moves to ensure high VaR & large DD
    returns = pd.Series([0.0, -0.04, 0.015, -0.05, 0.0, 0.02, -0.03, 0.01], index=idx)
    equity = (1 + returns.fillna(0)).cumprod() * 100_000
    ppy = 252

    thr = RiskThresholds(
        max_dd=0.03,   # 3% max drawdown allowed (we'll exceed)
        var_95=0.015,  # 1.5% one-period VaR allowed (we'll exceed)
        vol_ann=0.12,  # 12% annualized vol allowed (we'll exceed)
    )

    breaches, stats = check_breaches(equity, returns, ppy, thr)

    # Should have at least MAX_DD, VAR_95, VOL all breached
    codes = {c for c, _ in breaches}
    assert {"MAX_DD", "VAR_95", "VOL"}.issubset(codes)

    # Stats payload sanity
    assert "dd" in stats and isinstance(stats["dd"], float)
    assert "var95" in stats and stats["var95"] > 0
    assert "vol" in stats and stats["vol"] > 0
    assert "dd_series" in stats and isinstance(stats["dd_series"], pd.Series)
    # Drawdown min should be negative (a drop from prior peak)
    assert stats["dd"] < 0


def test_check_breaches_none_when_safe():
    # Gentle, steady, positive path: no breaches expected
    idx = pd.date_range("2024-02-01", periods=60, freq="D")
    # ~0.05% mean per day, tiny noise
    r = pd.Series(0.0005, index=idx) + pd.Series(0.0, index=idx)
    eq = (1 + r).cumprod() * 50_000
    ppy = 252

    thr = RiskThresholds(max_dd=0.10, var_95=0.02, vol_ann=0.20)
    breaches, stats = check_breaches(eq, r, ppy, thr)

    assert breaches == []
    assert stats["dd"] <= 0
    assert stats["var95"] >= 0
    assert stats["vol"] >= 0
