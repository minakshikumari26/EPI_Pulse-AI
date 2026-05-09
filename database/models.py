from sqlalchemy import Column, Date, Float, Integer, String

from database.db import Base


class DiseaseRecord(Base):
    __tablename__ = "disease_records"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    region = Column(String, index=True)
    cases = Column(Integer)
    temperature = Column(Float)
    humidity = Column(Float)
    rainfall = Column(Float)

