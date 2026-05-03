"""
Tabela de log de simulações no PostgreSQL.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, Float, DateTime, JSON, String
from sqlalchemy.dialects.postgresql import JSONB

from wine_project.database import Base


class SimulationLog(Base):
    __tablename__ = "simulation_logs"

    id                = Column(Integer, primary_key=True, index=True)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False)
    predicted_quality = Column(Integer, nullable=False)
    elapsed_ms        = Column(Float, nullable=False)
    input_data        = Column(JSONB, nullable=False)   # features do vinho
    probabilities     = Column(JSONB, nullable=False)   # distribuição de probabilidades