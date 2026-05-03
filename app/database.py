"""
Conexão com PostgreSQL + health-check de startup.
"""

import os
import time
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://wine_user:wine_pass@db:5432/wine_db",
)

engine = SessionLocal = None  # type: ignore  # inicializados em connect()

def _build_engine():
    global engine, SessionLocal
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def wait_for_db(retries: int = 15, delay: float = 2.0) -> None:
    """Bloqueia até o PostgreSQL aceitar conexões."""
    _build_engine()
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅  Banco de dados disponível.")
            return
        except Exception as exc:
            logger.warning(
                "⏳  Aguardando banco (%d/%d): %s", attempt, retries, exc
            )
            time.sleep(delay)
    raise RuntimeError("❌  Banco de dados não respondeu após %d tentativas." % retries)


class Base(DeclarativeBase):
    pass