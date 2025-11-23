# create_rag_tables.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapped_column
from datetime import datetime
import sqlalchemy

# ==================== 1. Configure your MySQL connection ====================
# Change the following items to your own
DB_USER = "root"          # Your username
DB_PASSWORD = "your_password"   # Your password
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "rag_db"        # Database name, will be created if it doesn't exist

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# ==================== 2. Create engine and Base ====================
engine = create_engine(
    DATABASE_URL,
    echo=True,                    # Print SQL for debugging (can be turned off in production)
    future=True,
    pool_pre_ping=True
)

Base = declarative_base()

# ==================== 3. Define two tables (plural convention) ====================
class Documents(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), unique=True, nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    version = mapped_column(Integer, default=1, nullable=False)
    is_deleted = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = mapped_column(DateTime, default=datetime.utcnow, 
                        onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        sqlalchemy.Index('idx_file_name_deleted', 'file_name', 'is_deleted'),
    )


class DocumentChunks(Base):
    __tablename__ = "document_chunks"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    document_id  = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    chunk_index  = Column(Integer, nullable=False, comment="Chunk sequence number, starting from 0")
    vector_id    = Column(String(255), unique=True, nullable=False, index=True,
                         comment="Vector ID in Chroma")

    __table_args__ = (
        sqlalchemy.UniqueConstraint('document_id', 'chunk_index', name='uix_doc_chunk'),
    )


# ==================== 4. One-click create database + tables ====================
def create_database_if_not_exists():
    """Create the database automatically if it doesn't exist"""
    no_db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
    temp_engine = create_engine(no_db_url, echo=False)
    with temp_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        print(f"Database `{DB_NAME}` is ready")

def create_tables():
    create_database_if_not_exists()
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("Tables `documents` and `document_chunks` created successfully!")

# ==================== 5. Execute ====================
if __name__ == "__main__":
    create_tables()

    # Optional: Print table structure for verification
    print("\n=== documents table structure ===")
    with engine.connect() as conn:
        result = conn.execute(text("SHOW CREATE TABLE documents"))
        print(result)

    print("\n=== document_chunks table structure ===")
    with engine.connect() as conn:
        result = conn.execute(text("SHOW CREATE TABLE document_chunks"))
        print(result)
