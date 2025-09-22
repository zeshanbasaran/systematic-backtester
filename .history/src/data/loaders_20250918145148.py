"""
loaders.py
----------
Utility functions for loading and caching historical price data.

- Retrieves OHLCV (Open, High, Low, Close, Adjusted Close, Volume) data
  for a given symbol, date range, and bar size.
- Uses a local parquet file cache (e.g., "data/SPY_1d.parquet") to avoid
  repeated downloads and speed up backtests.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import pandas as pd

# Optional dependency: pip install yfinance pyarrow (or fastparquet)
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


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common Yahoo columns to standard OHLCV names and ensure adj_close exists."""
    if df is None or df.empty:
        cols = ["open", "high", "low", "close", "adj_close", "volume"]
        return pd.DataFrame(columns=cols).astype({c: "float64" for c in cols})

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

    # Ensure adj_close
    if "adj_close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "adj_close"})
        elif "close" in df.columns:
            df["adj_close"] = df["close"]

    keep = ["open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Ensure DatetimeIndex (timezone-naive), sorted, no dups
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def _fetch_from_yahoo(symbol: str, start: str, end: str, bar: str) -> pd.DataFrame:
    """Fetch raw data from Yahoo Finance and return standardized OHLCV DataFrame."""
    interval = _yf_interval(bar)
    df = yf.download(
        symbol,
        start=start,
        end=end,            # yfinance treats end as exclusive
        interval=interval,
        auto_adjust=False,  # keep both Close and Adj Close
        progress=False,
        threads=True,
    )
    return _standardize_columns(df)


def _load_cache(path: Path) -> pd.DataFrame:
    """Load parquet cache if exists; else empty frame with proper columns."""
    if path.exists():
        df = pd.read_parquet(path)
        # normalize index/columns
        df = _standardize_columns(df)
        return df
    # empty with schema
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    return pd.DataFrame(columns=cols).astype({c: "float64" for c in cols})


def _needed_ranges(
    have: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determine which (start, end) ranges are missing from cached data.
    Returns a list of missing intervals in ascending order.
    """
    if have.empty:
        return [(start, end)]
    have_start, have_end = have.index.min(), have.index.max()
    missing: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if start < have_start:
        missing.append((start, min(have_start, end)))
    if end > have_end:
        missing.append((max(have_end, start), end))
    return [(s, e) for s, e in missing if s < e]


def get_price_data(symbol: str, start, end, bar: str = "1d") -> pd.DataFrame:
    """
    Fetch historical price data for a given symbol with local parquet caching.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., "SPY", "AAPL").
    start, end : str | datetime-like
        Start and end dates. For daily bars we treat end as inclusive;
        for intraday bars end is exclusive (yfinance convention).
    bar : str, default "1d"
        "1d" (daily) or "1h" (hourly).

    Returns
    -------
    pd.DataFrame
        Datetime-indexed DataFrame with:
        ['open','high','low','close','adj_close','volume'].
    """
    cache_file = _cache_path(symbol, bar)
    cached = _load_cache(cache_file)

    # Normalize inputs
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Determine gaps vs cache, fetch only what’s missing
    gaps = _needed_ranges(cached, start_ts, end_ts)
    fetched_parts = []
    for (s, e) in gaps:
        part = _fetch_from_yahoo(symbol, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), bar)
        if not part.empty:
            fetched_parts.append(part.loc[(part.index >= s) & (part.index < e)])

    if fetched_parts:
        new_data = pd.concat([cached] + fetched_parts, axis=0)
        new_data = new_data[~new_data.index.duplicated(keep="last")].sort_index()
        new_data.to_parquet(cache_file, index=True)  # requires pyarrow/fastparquet
    else:
        new_data = cached

    # --- Final windowing: inclusive end for daily, exclusive for intraday ---
    new_data.index = pd.to_datetime(new_data.index)
    new_data = new_data.sort_index()

    if bar == "1d":
        # Normalize to date so boundaries match midnight
        new_data.index = new_data.index.normalize()
        s = pd.to_datetime(start_ts).normalize()
        e = pd.to_datetime(end_ts).normalize()
        window = new_data.loc[(new_data.index >= s) & (new_data.index <= e)].copy()
    else:
        # Intraday: keep timestamps
        s = pd.to_datetime(start_ts)
        e = pd.to_datetime(end_ts)
        window = new_data.loc[(new_data.index >= s) & (new_data.index < e)].copy()

    # --- If empty, try a forced refetch with a small buffer to heal off-by-ones ---
    if window.empty:
        buf_start = (s - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        buf_end = (e + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        fresh = _fetch_from_yahoo(symbol, buf_start, buf_end, bar)
        if not fresh.empty:
            fresh.index = pd.to_datetime(fresh.index)
            fresh = fresh.sort_index()
            if bar == "1d":
                fresh.index = fresh.index.normalize()
                window = fresh.loc[(fresh.index >= s) & (fresh.index <= e)].copy()
            else:
                window = fresh.loc[(fresh.index >= s) & (fresh.index < e)].copy()

            # Update cache too
            merged = pd.concat([new_data, fresh], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_parquet(cache_file, index=True)

    # Ensure consistent dtypes
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in window.columns:
            window[col] = pd.to_numeric(window[col], errors="coerce")

    return window
