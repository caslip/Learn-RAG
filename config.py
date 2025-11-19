import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

class Config:
    # MySQL 配置
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_db")
    
    # MySQL 连接 URI
    # 格式: mysql+pymysql://user:password@host:port/database
    MYSQL_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

    # 其他应用配置
    APP_NAME = "RAG Application"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")

    # Vector Database (Chroma) 配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # LLM 配置
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:8b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434") # Ollama default URL

    # Embedding 模型配置
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_MODEL_DEVICE = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")

    # RAG 配置
    RAG_K = int(os.getenv("RAG_K", "3")) # 检索的文档数量

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# 配置字典
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}

# 获取当前配置
def get_config(config_name="default"):
    return config_by_name[config_name]
