"""
config.py
----------
Configuration file for the systematic trading backtester & risk monitor.

Defines key parameters for data ingestion, strategy backtesting,
risk monitoring, and database/reporting integration.
"""

# --- Trading Universe & Data Parameters ---
SYMBOLS = ["SPY", "TLT"]        # List of tickers to trade (stocks, ETFs, futures, etc.)
START   = "2013-01-01"          # Start date for backtest
END     = "2025-01-01"          # End date for backtest
BAR     = "1d"                  # Data frequency: "1d" (daily) or "1h" (hourly)

# --- Portfolio & Transaction Settings ---
INIT_CASH      = 100_000        # Initial portfolio value in USD
SLIPPAGE_BPS   = 1.0            # Slippage cost in basis points (1 bps = 0.01%)
COMM_PER_TRADE = 0.005          # Commission per trade: $/share OR basis points

# --- Risk Management Thresholds ---
RISK = dict(
    max_dd = 0.20,              # Maximum allowed drawdown (20%)
    var_95 = 0.025,             # Value-at-Risk at 95% confidence (2.5%)
    vol_ann = 0.25              # Annualized volatility cap (25%)
)

# --- Database Configuration ---
DB_URL = "sqlite:///data/trades.db"  
# Example alternatives: 
#   MySQL -> "mysql+pymysql://user:password@localhost/trades"
#   PostgreSQL -> "postgresql://user:password@localhost/trades"
