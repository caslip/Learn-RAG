import os
import json
from datetime import datetime
from sqlalchemy import Boolean, create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, mapped_column
from config import get_config

# Get configuration
config = get_config()
MYSQL_URI = config.MYSQL_URI

# Create database engine
engine = create_engine(MYSQL_URI)

# Create base class (using the new 2.0 style)
Base = declarative_base()

# doc_table.py (assuming you already have it, slightly completed/modified)
# doc_table.py
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== Plural table names, more standard! =====================
class Documents(Base):   # ← Class name capitalized plural
    __tablename__ = "documents"  # Table name lowercase plural (database standard)

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), unique=True, nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    version = mapped_column(Integer, default=1, nullable=False)
    is_deleted = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = mapped_column(DateTime, default=datetime.utcnow, 
                        onupdate=datetime.utcnow, nullable=False)

    # Relationship: one document has many chunks (convenient for reverse lookup, optional)
    chunks = relationship("DocumentChunks", back_populates="document", 
                          cascade="all, delete-orphan")


class DocumentChunks(Base):  # ← Class name capitalized plural
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), 
                         nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)      # Which chunk number
    vector_id = Column(String(255), unique=True, nullable=False, index=True)

    # Reverse relationship
    document = relationship("Documents", back_populates="chunks")

    
# Create database tables
def create_tables():
    """
    Create all defined tables in the database.
    """
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")

# Get database session
def get_db_session():
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Do not close here, let the caller control it (or use contextmanager)
