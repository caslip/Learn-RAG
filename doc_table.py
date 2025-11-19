import os
import json
from datetime import datetime
from sqlalchemy import Boolean, create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, mapped_column
from config import get_config

# 获取配置
config = get_config()
MYSQL_URI = config.MYSQL_URI

# 创建数据库引擎
engine = create_engine(MYSQL_URI)

# 创建基类 (使用新的 2.0 风格)
Base = declarative_base()

# doc_table.py （假设你已经有了，稍微补全/修改）
# doc_table.py
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== 复数表名，更规范！====================
class Documents(Base):   # ← 类名大写复数
    __tablename__ = "documents"  # 表名小写复数（数据库标准）

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), unique=True, nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    version = mapped_column(Integer, default=1, nullable=False)
    is_deleted = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = mapped_column(DateTime, default=datetime.utcnow, 
                        onupdate=datetime.utcnow, nullable=False)

    # 关系：一个文档有多个 chunks（方便反向查询，可选）
    chunks = relationship("DocumentChunks", back_populates="document", 
                          cascade="all, delete-orphan")


class DocumentChunks(Base):  # ← 类名大写复数
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), 
                         nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)      # 第几个 chunk
    vector_id = Column(String(255), unique=True, nullable=False, index=True)

    # 反向关系
    document = relationship("Documents", back_populates="chunks")

    
# 创建数据库表
def create_tables():
    """
    在数据库中创建所有定义的表。
    """
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")

# 获取数据库会话
def get_db_session():
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # 不要在这里 close，由调用方控制（或用 contextmanager）

