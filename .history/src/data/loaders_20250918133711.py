"""
loader.py
--------------
Utility functions for loading and caching historical price data.

- Retrieves OHLCV (Open, High, Low, Close, Adjusted Close, Volume) data 
  for a given symbol, date range, and bar size.
- Uses a local parquet file cache (e.g., "data/SPY_1d.parquet") to avoid
  repeated downloads and speed up backtests.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

# Optional dependency: pip install yfinance pyarrow
import yfinance as yf


DATA_DIR = Path("data")  # cache lives here
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, bar: str) -> Path:
    """Return Path to parquet cache file for (symbol, bar)."""
    safe_symbol = symbol.replace("/", "_").upper()
    return DATA_DIR / f"{safe_symbol}_{bar}.parquet"


def _yf_interval(bar: str) -> str:
    """Map our 'bar' to yfinance interval."""
    if bar in ("1d", "1h"):
        return bar
    raise ValueError(f"Unsupported bar '{bar}'. Use '1d' or '1h'.")


def _fetch_from_yahoo(symbol: str, start: str, end: str, bar: str) -> pd.DataFrame:
    """Fetch raw data from Yahoo Finance and return standardized OHLCV DataFrame."""
    interval = _yf_interval(bar)
    # auto_adjust=False gives 'Adj Close' column alongside raw OHLC
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"], dtype="float64")

    # yfinance column names come like 'Open','High','Low','Close','Adj Close','Volume'
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    # If multi-index columns (happens on some tickers), flatten first
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in col if c]).strip() for col in df.columns]
        # try to remap sensible variants
        for k, v in list(rename_map.items()):
            if k not in df.columns and k.replace(" ", "_") in df.columns:
                rename_map[k.replace(" ", "_")] = v

    df = df.rename(columns=rename_map)
    keep = ["open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Ensure DatetimeIndex (timezone-naive)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # Sort & drop bad rows
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


def _load_cache(path: Path) -> pd.DataFrame:
    """Load parquet cache if exists; else empty frame with proper columns."""
    if path.exists():
        df = pd.read_parquet(path)
        # safety: normalize index/columns
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = df.sort_index()
        return df
    # empty with schema
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    return pd.DataFrame(columns=cols).astype({c: "float64" for c in cols})


def _needed_ranges(
    have: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determine which (start, end) ranges are missing from cached data.
    Returns a list of missing intervals in ascending order.
    """
    missing = []
    if have.empty:
        return [(start, end)]

    have_start, have_end = have.index.min(), have.index.max()

    # Leading gap
    if start < have_start:
        missing.append((start, min(have_start, end)))

    # Trailing gap
    if end > have_end:
        missing.append((max(have_end, start), end))

    # If requested range is entirely inside have, nothing missing
    # (We are not patching internal holes here; parquet should be contiguous. If needed, add logic.)
    return [(s, e) for s, e in missing if s < e]


def get_price_data(symbol: str, start: str, end: str, bar: str = "1d") -> pd.DataFrame:
    """
    Fetch historical price data for a given symbol with local parquet caching.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., "SPY", "AAPL").
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD). Exclusive for Yahoo; we treat it as exclusive too.
    bar : str, default "1d"
        Bar size: "1d" (daily) or "1h" (hourly).
        Note: Yahoo typically limits intraday history (e.g., ~730 days for 1h).

    Returns
    -------
    pd.DataFrame
        Datetime-indexed DataFrame with columns:
        ['open','high','low','close','adj_close','volume'].

    Caching
    -------
    - Cache file: data/{SYMBOL}_{BAR}.parquet (e.g., data/SPY_1d.parquet)
    - If cache partially covers [start, end), fetches only missing ranges and merges.
    - Saves the merged result back to the cache.
    """
    cache_file = _cache_path(symbol, bar)
    cached = _load_cache(cache_file)

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Determine what we still need
    gaps = _needed_ranges(cached, start_ts, end_ts)

    fetched_parts = []
    for (s, e) in gaps:
        # yfinance end is exclusive; we pass as-is
        part = _fetch_from_yahoo(symbol, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), bar)
        if not part.empty:
            # clip strictly to [s, e)
            part = part.loc[(part.index >= s) & (part.index < e)]
            fetched_parts.append(part)

    if fetched_parts:
        new_data = pd.concat([cached] + fetched_parts, axis=0)
        new_data = new_data[~new_data.index.duplicated(keep="last")].sort_index()
        # Persist cache (requires pyarrow or fastparquet)
        new_data.to_parquet(cache_file, index=True)
    else:
        new_data = cached

    # Return only the requested window
    new_data.index = pd.to_datetime(new_data.index)
    window = new_data.loc[(new_data.index >= start_ts) & (new_data.index < end_ts)].copy()

    # Ensure consistent dtypes
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in window.columns:
            window[col] = pd.to_numeric(window[col], errors="coerce")

    return window
