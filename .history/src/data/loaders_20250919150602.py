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


def _standardize_columns(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Map Yahoo columns to ['open','high','low','close','adj_close','volume'] and
    ensure tz-naive DatetimeIndex."""
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols).astype({c: "float64" for c in cols})

    # If MultiIndex (rare with group_by='column'), slice last level
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=0, drop_level=True) if symbol else df
        except Exception:
            pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    df = df.copy()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Rename variants
    rename_map = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "adjclose": "adj_close",
        "adj_close": "adj_close", "adjusted_close": "adj_close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Ensure adj_close exists
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Keep desired columns
    df = df[[c for c in cols if c in df.columns]]

    # DatetimeIndex (tz-naive), sorted, dedup
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)  # safer than tz_convert(None) here
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fetch_from_yahoo(symbol: str, start: str, end: str, bar: str) -> pd.DataFrame:
    """Fetch from Yahoo, trying threads=False first (Windows-safe), then True."""
    interval = _yf_interval(bar)

    # Clamp hourly to ~2y
    if bar == "1h":
        end_ts = pd.to_datetime(end)
        min_start = end_ts - pd.Timedelta(days=730)
        start = max(pd.to_datetime(start), min_start).strftime("%Y-%m-%d")

    last_err = None
    for threads in (False, True):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,               # end is exclusive in yfinance
                interval=interval,
                auto_adjust=False,     # keep Adj Close separate
                group_by="column",
                progress=False,
                threads=threads,
            )
            out = _standardize_columns(df, symbol=symbol)
            if not out.empty:
                return out
        except Exception as e:
            last_err = e

    if last_err:
        raise RuntimeError(f"yfinance download failed for {symbol} {start}->{end} {bar}: {last_err}")
    return _standardize_columns(pd.DataFrame(), symbol=symbol)


def _load_cache(path: Path) -> pd.DataFrame:
    """Load parquet cache if exists; else empty frame with proper columns."""
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    if path.exists():
        try:
            df = pd.read_parquet(path)
            return _standardize_columns(df)  # already normalized; symbol not needed for cache
        except Exception:
            pass
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
