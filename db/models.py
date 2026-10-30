from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class SalesHistory(Base):
    __tablename__ = "sales_history"
    __table_args__ = (UniqueConstraint("date", "store_nbr", "family", name="uq_sales_history_day_store_family"), Index("ix_sales_history_date", "date"))
    date: Mapped[object] = mapped_column(Date, primary_key=True)
    store_nbr: Mapped[int] = mapped_column(Integer, primary_key=True)
    family: Mapped[str] = mapped_column(String(100), primary_key=True)
    sales: Mapped[object] = mapped_column(Numeric(18, 3), nullable=False)
    onpromotion: Mapped[int] = mapped_column(Integer, nullable=False)


class SalesNew(Base):
    __tablename__ = "sales_new"
    __table_args__ = (CheckConstraint("status IN ('pending','processed','failed')", name="ck_sales_new_status"), Index("ix_sales_new_status", "status"), Index("ix_sales_new_date", "date"))
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("batches.batch_id"), nullable=True)
    date: Mapped[object] = mapped_column(Date, nullable=False)
    store_nbr: Mapped[int] = mapped_column(Integer, nullable=False)
    family: Mapped[str] = mapped_column(String(100), nullable=False)
    sales: Mapped[object] = mapped_column(Numeric(18, 3), nullable=False)
    onpromotion: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, server_default="pending")
    received_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Batch(Base):
    __tablename__ = "batches"
    __table_args__ = (Index("ix_batches_received_at", "received_at"),)
    batch_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    original_file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    received_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    first_date: Mapped[object | None] = mapped_column(Date)
    last_date: Mapped[object | None] = mapped_column(Date)
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)


class DailyTotal(Base):
    __tablename__ = "daily_totals"
    __table_args__ = (Index("ix_daily_totals_date", "date"),)
    date: Mapped[object] = mapped_column(Date, primary_key=True)
    sales: Mapped[object] = mapped_column(Numeric(18, 9), nullable=False)


class CleanSeries(Base):
    __tablename__ = "clean_series"
    __table_args__ = (Index("ix_clean_series_date", "date"),)
    date: Mapped[object] = mapped_column(Date, primary_key=True)
    sales: Mapped[object] = mapped_column(Numeric(18, 9), nullable=False)
    sales_raw: Mapped[object] = mapped_column(Numeric(18, 9), nullable=False)
    is_closure: Mapped[bool] = mapped_column(Boolean, nullable=False)


class DataVersion(Base):
    __tablename__ = "data_versions"
    data_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_date: Mapped[object] = mapped_column(Date, nullable=False)
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    run_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (Index("ix_forecasts_version_model", "data_version", "model"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    data_version: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    date: Mapped[object] = mapped_column(Date, nullable=False)
    forecast: Mapped[object] = mapped_column(Numeric(18, 9), nullable=False)


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    model: Mapped[str | None] = mapped_column(String(100))
    started_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[object | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    step: Mapped[str | None] = mapped_column(String(100))
    message: Mapped[str | None] = mapped_column(Text)
    batch_ids: Mapped[object | None] = mapped_column(JSONB)


Run.__tablename__ = "runs"
metadata = Base.metadata
