"""
test_strategies.py
------------------
Unit tests for strategy signal generators.

Run:
    pytest tests/test_strategies.py -v
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

from src.strategies.sma_crossover import sma_crossover
from src.strategies.bollinger_meanrev import bollinger_meanrev


# ----------------------------
# Helpers
# ----------------------------

def make_df(prices, start="2024-01-01", freq="D"):
    """Create a minimal OHLCV DataFrame with an adj_close path."""
    idx = pd.date_range(start=start, periods=len(prices), freq=freq)
    ac = pd.Series(prices, index=idx, dtype=float)
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
# SMA Crossover
# ----------------------------

def test_sma_crossover_long_and_shift():
    # Prices move up → short MA crosses above long → long signal expected.
    # Use tiny windows to keep test compact.
    df = make_df([100, 101, 102, 103, 104])
    pos = sma_crossover(df, short=2, long=3, allow_short=False)

    # Output must align and be in [-1,0,1]
    assert pos.index.equals(df.index)
    assert set(np.unique(pos.values)).issubset({-1, 0, 1})

    # Shift-by-1: first bar must be 0 (can't act on first observation)
    assert pos.iloc[0] == 0

    # After MA(2) > MA(3) occurs (starts around 3rd/4th bar), we expect long=1
    # But due to shift-by-1, the first 1 will appear one bar after crossover.
    assert (pos.iloc[1:] >= 0).all()           # never short because allow_short=False
    assert pos.max() == 1                       # we do get long at some point
    # ensure at least one transition 0->1 (an entry trade)
    assert ((pos.shift(1).fillna(0) == 0) & (pos == 1)).any()


def test_sma_crossover_allow_short():
    # Sideways to slightly down: fast <= slow most of the time → short (-1) allowed.
    df = make_df([100, 100, 99.9, 100.0, 99.8, 99.7, 99.6])
    pos = sma_crossover(df, short=2, long=3, allow_short=True)

    assert pos.index.equals(df.index)
    assert set(np.unique(pos.values)).issubset({-1, 0, 1})

    # With allow_short=True, when not long we should be short (-1), not flat.
    # After warm-up and shift, ensure a -1 appears.
    assert (pos == -1).any()

    # Still shifted: first bar must be 0.
    assert pos.iloc[0] == 0


# ----------------------------
# Bollinger Mean-Reversion
# ----------------------------

def test_bollinger_long_entry_and_exit():
    # Build a path that spends time far below the mean, then reverts.
    prices = [100]*20  # warm-up flat window for lookback
    prices += [95, 94, 93]  # deep below → expect long entries (z < -k)
    prices += [96, 97, 98]  # revert back inside band → expect flatten when |z| < band_exit
    df = make_df(prices)
    lookback, k, exit_band = 20, 2.0, 0.5

    pos = bollinger_meanrev(df, lookback=lookback, k=k, band_exit=exit_band, allow_short=True)

    # Alignment & range
    assert pos.index.equals(df.index)
    assert set(np.unique(pos.values)).issubset({-1.0, 0.0, 1.0})

    # Shift-by-1 guarantees first bar = 0
    assert pos.iloc[0] == 0.0

    # Expect at least one long (+1) during deep-below-mean segment
    assert (pos == 1.0).any()

    # After reverting (|z| < exit_band), position should flatten to 0 at least once
    # Note: due to shift, flatten will show after the reversion bar.
    flatten_indices = pos[(pos.shift(1).fillna(0) != 0.0) & (pos == 0.0)].index
    assert len(flatten_indices) >= 1


def test_bollinger_no_shorts_when_disallowed():
    # Make a path that goes far ABOVE mean; with allow_short=False we should never see -1.
    prices = [100]*20 + [105, 106, 107, 108, 107, 106, 105]
    df = make_df(prices)

    pos = bollinger_meanrev(df, lookback=20, k=2.0, band_exit=0.5, allow_short=False)

    assert pos.index.equals(df.index)
    assert set(np.unique(pos.values)).issubset({0.0, 1.0})  # no -1s


def test_bollinger_handles_nan_warmup_and_shift():
    # With lookback=5, the first 4 z-scores are NaN; ensure strategy returns valid positions (0,±1) and shifts.
    df = make_df([100, 101, 102, 103, 104, 105, 104, 103, 102])
    pos = bollinger_meanrev(df, lookback=5, k=1.5, band_exit=0.25, allow_short=True)

    # All outputs are finite and in the allowed set
    assert set(np.unique(pos.values)).issubset({-1.0, 0.0, 1.0})

    # First bar still zero because of shift
    assert pos.iloc[0] == 0.0


# ----------------------------
# General signal hygiene
# ----------------------------

def test_signals_range_and_dtype_and_index():
    df = make_df([100, 101, 99, 102, 101, 100, 99, 98, 100])
    s1 = sma_crossover(df, short=2, long=3, allow_short=True)
    s2 = bollinger_meanrev(df, lookback=5, k=1.5, band_exit=0.25, allow_short=True)

    for s in (s1, s2):
        assert s.index.equals(df.index)
        assert s.dtype.kind in ("f", "i")
        assert set(np.unique(s.values)).issubset({-1, 0, 1})
        # No NaNs expected after shift/fill
        assert not s.isna().any()
