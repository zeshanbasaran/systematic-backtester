import yfinance as yf
df = yf.download("SPY", start="2013-01-01", end="2024-12-31", interval="1d")
print(df.shape)
print(df.head())
