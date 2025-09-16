# global settings (symbols, start/end dates, bar size, initial cash, slippage, commissions, 
# risk thresholds, DB URL)

SYMBOLS = ["SPY", "TLT"]
START = "2013-01-01"
END   = "2025-01-01"
BAR = "1d"   # or "1h"
INIT_CASH = 100_000
SLIPPAGE_BPS = 1.0
COMM_PER_TRADE = 0.005  # $/share or bps
RISK = dict(max_dd=0.20, var_95=0.025, vol_ann=0.25)
DB_URL = "sqlite:///data/trades.db"  
