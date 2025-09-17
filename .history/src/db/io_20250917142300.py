"""
io.py
-----
Database utilities for initializing and managing SQLAlchemy sessions.

Responsibilities
---------------
- Create engine connections to SQLite/MySQL/Postgres.
- Initialize the database schema (create tables).
- Provide session factory for reading/writing ORM objects.

Relies on models defined in `models.py`.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import Base 


def make_engine(db_url: str):
    """
    Create a SQLAlchemy engine.

    Parameters
    ----------
    db_url : str
        Database connection URL. Examples:
        - SQLite: "sqlite:///data/trades.db"
        - MySQL: "mysql+pymysql://user:password@localhost/trades"
        - Postgres: "postgresql://user:password@localhost/trades"

    Returns
    -------
    sqlalchemy.Engine
        Database engine for connecting.
    """
    return create_engine(db_url, echo=False, future=True)


def init_db(db_url: str):
    """
    Initialize the database (create tables if not present).

    Parameters
    ----------
    db_url : str
        Database connection URL.

    Returns
    -------
    sqlalchemy.Engine
        Engine object connected to the database.
    """
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    return engine


def make_session(db_url: str):
    """
    Create a session bound to the given database.

    Parameters
    ----------
    db_url : str
        Database connection URL.

    Returns
    -------
    sqlalchemy.orm.Session
        A new SQLAlchemy session for querying and committing ORM objects.
    """
    engine = make_engine(db_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionLocal()
