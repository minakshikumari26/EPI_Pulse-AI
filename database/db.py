import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.config import load_config


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    load_config().get("database", {}).get("url", "postgresql://postgres:password@db:5432/epipulse"),
)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

