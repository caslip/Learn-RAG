# create_rag_tables.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapped_column
from datetime import datetime
import sqlalchemy

# ==================== 1. 配置你的 MySQL 连接 ====================
# 把下面这几项改成你自己的
DB_USER = "root"          # 你的用户名
DB_PASSWORD = "123"   # 你的密码
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "rag_db"        # 数据库名，不存在会自动创建

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# ==================== 2. 创建 engine 和 Base ====================
engine = create_engine(
    DATABASE_URL,
    echo=True,                    # 打印 SQL，便于调试（上线可关）
    future=True,
    pool_pre_ping=True
)

Base = declarative_base()

# ==================== 3. 定义两张表（复数规范） ====================
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
    chunk_index  = Column(Integer, nullable=False, comment="块序号，从0开始")
    vector_id    = Column(String(255), unique=True, nullable=False, index=True,
                         comment="Chroma 中的向量 ID")

    __table_args__ = (
        sqlalchemy.UniqueConstraint('document_id', 'chunk_index', name='uix_doc_chunk'),
    )


# ==================== 4. 一键创建数据库 + 表 ====================
def create_database_if_not_exists():
    """如果数据库不存在就自动创建"""
    no_db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
    temp_engine = create_engine(no_db_url, echo=False)
    with temp_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        print(f"数据库 `{DB_NAME}` 已准备好")

def create_tables():
    create_database_if_not_exists()
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("表 `documents` 和 `document_chunks` 创建成功！")

# ==================== 5. 执行 ====================
if __name__ == "__main__":
    create_tables()

    # 可选：打印表结构验证
    print("\n=== documents 表结构 ===")
    with engine.connect() as conn:
        result = conn.execute(text("SHOW CREATE TABLE documents"))
        print(result)

    print("\n=== document_chunks 表结构 ===")
    with engine.connect() as conn:
        result = conn.execute(text("SHOW CREATE TABLE document_chunks"))
        print(result)