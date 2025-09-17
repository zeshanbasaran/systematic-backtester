"""
models.py
---------
Database schema definitions for backtests using SQLAlchemy ORM.

Tables
------
- Run        : Metadata about each backtest run (strategy, symbol, period).
- Trade      : Executed trades with position weights, price, and costs.
- DailyPnl   : Per-day returns and equity values.
- RiskEvent  : Logged risk breaches (drawdown, VaR, volatility).

These models map Python objects <-> database tables so you can
persist and query backtest results in SQLite/MySQL/Postgres.
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, UniqueConstraint

Base = declarative_base()


class Run(Base):
    """
    Backtest run metadata.
    Stores which strategy, symbol, and bar size were tested,
    along with the start/end dates.
    """
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    strategy = Column(String, index=True)  # e.g., "SMA_Crossover"
    symbol = Column(String, index=True)    # e.g., "SPY"
    bar = Column(String)                   # e.g., "1d" or "1h"
    start = Column(DateTime)
    end = Column(DateTime)


class Trade(Base):
    """
    Executed trades log.
    Records each position change (buy/sell), size, price, and cost.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), index=True)
    timestamp = Column(DateTime, index=True)
    side = Column(String)        # "BUY" / "SELL"
    target_weight = Column(Float)  # final target position after trade
    delta_weight = Column(Float)   # change from previous position
    price = Column(Float)
    cost = Column(Float)           # trading cost (slippage/commission)


class DailyPnl(Base):
    """
    Daily profit & loss records.
    Stores returns and equity value for each bar in a run.
    """
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), index=True)
    timestamp = Column(DateTime, index=True)
    ret = Column(Float)     # strategy return for the day/bar
    equity = Column(Float)  # cumulative equity value


class RiskEvent(Base):
    """
    Risk monitoring log.
    Stores risk breaches such as max drawdown, VaR, or volatility
    that exceeded thresholds during a run.
    """
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), index=True)
    timestamp = Column(DateTime, index=True)
    metric = Column(String)   # e.g., "MAX_DD", "VAR_95", "VOL"
    value = Column(Float)     # actual observed metric value
    threshold = Column(Float) # threshold that was breached

    __table_args__ = (
        UniqueConstraint("run_id", "timestamp", "metric", name="uq_event"),
    )
