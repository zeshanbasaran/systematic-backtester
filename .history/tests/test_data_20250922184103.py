"""
test_data.py
------------
Unit tests for data loading (loaders.py).

Run with:
    pytest tests/test_data.py -v
"""

import os
import pandas as pd
import pytest
from pathlib import Path

from src.data.loaders import get_price_data, _cache_path, _standardize_columns


def test_cache_path_formatting():
    path = _cache_path("SPY", "1d")
    assert str(path).endswith("SPY_1d.parquet")


def test_get_price_data_schema(tmp_path, monkeypatch):
    """Ensure returned DataFrame has expected OHLCV schema."""
    df = get_price_data("SPY", "2023-01-01", "2023-03-01", "1d")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index.tz is None


def test_invalid_symbol_returns_empty(monkeypatch):
    """Invalid ticker should return empty DataFrame, not crash."""
    df = get_price_data("FAKE_TICKER", "2023-01-01", "2023-02-01", "1d")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_invalid_bar_raises():
    with pytest.raises(ValueError):
        get_price_data("SPY", "2023-01-01", "2023-02-01", "5m")


def test_standardize_columns_handles_empty():
    df = _standardize_columns(pd.DataFrame())
    assert all(c in df.columns for c in ["open", "high", "low", "close", "adj_close", "volume"])
    assert df.empty


def test_cache_reuse(tmp_path, monkeypatch):
    """Second call should reuse parquet cache if it exists."""
    symbol, bar = "SPY", "1d"
    path = _cache_path(symbol, bar)
    if path.exists():
        os.remove(path)

    # First load (creates cache)
    df1 = get_price_data(symbol, "2023-01-01", "2023-02-01", bar)
    assert path.exists()

    # Second load (reuses cache)
    df2 = get_price_data(symbol, "2023-01-01", "2023-02-01", bar)
    pd.testing.assert_index_equal(df1.index, df2.index)
    pd.testing.assert_frame_equal(df1, df2)
