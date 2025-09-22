import yfinance as yf
df = yf.download(
    "SPY",
    start="2013-01-01",
    end="2024-12-31",
    interval="1d",
    auto_adjust=False,       # get 'Adj Close' back
    group_by="column",       # avoid MultiIndex
    progress=False
)
df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
