"""
plots.py
---------
End-to-end runners ("plots") that fly the backtesting pipeline.

What this does
--------------
- Loads config (symbols, dates, cash, risk limits, DB URL).
- Pulls/caches historical prices.
- Generates positions from a chosen strategy.
- Simulates portfolio equity/returns/trades.
- Computes risk metrics and checks thresholds.
- Saves results to DB (runs, trades, daily PnL, risk events).
- Exports an Excel report + PNG charts.

Usage
-----
From project root (so package imports resolve):
    python -m src.pilots

Outputs
-------
- DB rows in the configured database (e.g., data/trades.db).
- Excel + charts under reports/ (one set per run).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

# --- Config ---
from src.config import SYMBOLS, START, END, BAR, INIT_CASH, DB_URL, RISK

# --- Data loader ---
from src.data.loaders import get_price_data

# --- Strategies ---
from src.strategies.sma_crossover import sma_crossover
from src.strategies.bollinger_meanrev import bollinger_meanrev

# --- Engine ---
from src.engine.backtester import simulate
from src.engine.metrics import max_drawdown

# --- Risk monitor ---
from src.risk.monitor import RiskThresholds, check_breaches

# --- Reports ---
from src.reporting.reports import build_performance_report

# --- DB I/O & Models ---
from src.db.io import init_db, make_session
from src.db.models import Run, Trade, DailyPnl, RiskEvent

# -------------------------
# Helpers
# -------------------------

def _periods_per_year_from_bar(bar: str) -> int:
    """
    Infer periods_per_year from bar string.
    - "1d": ~252 trading days/year
    - "1h": ~252 * 6.5 trading hours/year (approx)
    """
    if bar == "1d":
        return 252
    if bar == "1h":
        return int(252 * 6.5)
    # Fallback: daily-like
    return 252


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _threshold_for_metric(thr: RiskThresholds, metric_code: str) -> float:
    """
    Map metric code -> threshold value stored in RiskEvent.
    (Drawdown threshold is stored as positive fraction, e.g., 0.20 for 20%.)
    """
    code = metric_code.upper()
    if code == "MAX_DD":
        return thr.max_dd
    if code == "VAR_95":
        return thr.var_95
    if code == "VOL":
        return thr.vol_ann
    return float("nan")


# -------------------------
# Pilot runner
# -------------------------

def run_pilot_for_symbol(
    strategy_name: str,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    symbol: str,
    *,
    start: str = START,
    end: str = END,
    bar: str = BAR,
    init_cash: float = INIT_CASH,
    db_url: str = DB_URL,
    risk_cfg: Dict = RISK,
    reports_dir: str = "reports",
) -> Dict:
    """
    Run a complete backtest for (strategy, symbol), persist & report.

    Parameters
    ----------
    strategy_name : str
        Display/name for the strategy (e.g., "SMA", "Bollinger").
    strategy_fn : Callable
        Function that takes price DataFrame and returns position Series.
    symbol : str
        Ticker to backtest.
    start, end : str
        Date range (YYYY-MM-DD).
    bar : str
        Bar size ("1d" or "1h").
    init_cash : float
        Starting equity.
    db_url : str
        SQLAlchemy DB URL.
    risk_cfg : dict
        Dict like {"max_dd": 0.20, "var_95": 0.025, "vol_ann": 0.25}.
    reports_dir : str
        Where to write Excel/PNG outputs.

    Returns
    -------
    dict with keys:
      data, pos, result, risk, db_ids, report
    """
    # --- Load data ---
    df = get_price_data(symbol, start, end, bar)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} [{start}..{end}] {bar}")

    # --- Generate positions (shift handled inside strategy functions already) ---
    pos = strategy_fn(df)

    # --- Backtest ---
    bt = simulate(df, pos, init_cash)
    equity, returns, trades = bt["equity"], bt["returns"], bt["trades"]

    # --- Risk checks ---
    ppy = _periods_per_year_from_bar(bar)
    thr = RiskThresholds(**risk_cfg)
    breaches, stats = check_breaches(equity, returns, ppy, thr)

    # --- Reports ---
    out_base = f"{strategy_name}_{symbol}_{bar}"
    xlsx_path = f"{reports_dir}/{out_base}.xlsx"
    png_dir = f"{reports_dir}/img/{out_base}"
    _ensure_dir(Path(xlsx_path).parent)
    _ensure_dir(png_dir)

    report = build_performance_report(
        equity=equity,
        returns=returns,
        dd_series=stats["dd_series"],
        trades=trades,
        out_xlsx=xlsx_path,
        out_png_dir=png_dir,
    )

    # --- Persist to DB ---
    engine = init_db(db_url)  # ensures tables exist
    session = make_session(db_url)

    run_row = Run(
        strategy=strategy_name,
        symbol=symbol,
        bar=bar,
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
    )
    session.add(run_row)
    session.commit()  # get run_id
    run_id = run_row.id

    # Trades
    trade_rows = []
    if trades is not None and len(trades) > 0:
        for _, r in trades.iterrows():
            trade_rows.append(
                Trade(
                    run_id=run_id,
                    timestamp=pd.to_datetime(r["timestamp"]),
                    side=("BUY" if r["delta_w"] > 0 else "SELL"),
                    target_weight=float(r["target_w"]),
                    delta_weight=float(r["delta_w"]),
                    price=float(r["price"]),
                    cost=float(r["cost"]),
                )
            )
    if trade_rows:
        session.add_all(trade_rows)

    # Daily PnL
    pnl_df = pd.DataFrame(
        {"timestamp": equity.index, "equity": equity.values, "ret": returns.values}
    )
    pnl_rows = [
        DailyPnl(
            run_id=run_id,
            timestamp=pd.to_datetime(row.timestamp),
            ret=float(row.ret),
            equity=float(row.equity),
        )
        for row in pnl_df.itertuples(index=False)
    ]
    session.add_all(pnl_rows)

    # Risk events
    event_rows = []
    for metric_code, value in breaches:
        event_rows.append(
            RiskEvent(
                run_id=run_id,
                timestamp=pd.to_datetime(equity.index.max()),
                metric=metric_code,
                value=float(value),
                threshold=float(_threshold_for_metric(thr, metric_code)),
            )
        )
    if event_rows:
        session.add_all(event_rows)

    session.commit()
    session.close()

    return {
        "data": df,
        "pos": pos,
        "result": bt,
        "risk": {"breaches": breaches, "stats": stats, "thresholds": asdict(thr)},
        "db_ids": {"run_id": run_id, "trades": len(trade_rows), "pnl_rows": len(pnl_rows), "events": len(event_rows)},
        "report": {"xlsx": xlsx_path, **report},
    }


def main():
    """
    Run a small suite of pilots over the configured symbols.
    Adds two strategies for variety: SMA crossover & Bollinger mean-reversion.
    """
    pilots: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("SMA", lambda df: sma_crossover(df, short=50, long=200, allow_short=False)),
        ("Bollinger", lambda df: bollinger_meanrev(df, lookback=20, k=2.0, band_exit=0.5, allow_short=True)),
    ]

    for sym in SYMBOLS:
        for strat_name, fn in pilots:
            print(f"[RUN] {strat_name} on {sym} ({BAR}) {START}..{END}")
            out = run_pilot_for_symbol(
                strategy_name=strat_name,
                strategy_fn=fn,
                symbol=sym,
                start=START,
                end=END,
                bar=BAR,
                init_cash=INIT_CASH,
                db_url=DB_URL,
                risk_cfg=RISK,
                reports_dir="reports",
            )
            breaches = out["risk"]["breaches"]
            if breaches:
                print(f"  -> Risk breaches: {breaches}")
            else:
                print("  -> No risk breaches.")
            print(f"  -> Report: {out['report']['xlsx']}")
            print(f"  -> DB run_id: {out['db_ids']['run_id']}")


if __name__ == "__main__":
    main()
