# Learn-RAG

A comprehensive Retrieval-Augmented Generation (RAG) system implementation, exploring various vector databases and document processing techniques.

## Features

- **Multiple Vector Database Support**: Utilizes both Chroma and FAISS for vector storage and retrieval.
- **Document Processing**: Supports various file formats (TXT, PDF, DOCX) with intelligent chunking strategies.
- **RAG Implementation**: Implements a RAG agent with LLM integration (Ollama) for generating responses based on retrieved contexts.
- **Document Management**: Includes versioning and logical deletion for documents.
- **Evaluation Framework**: Provides tools for evaluating RAG system performance.

## Project Structure

```
Learn-RAG/
├── chroma_db.py              # Chroma vector store implementation
├── vector_db/
│   └── faiss_db.py          # FAISS vector store implementation
├── rag.py                    # RAG agent implementation
├── db.py                     # Basic FAISS vector store example
├── doc_table.py             # Database models for document management
├── config.py                # Configuration settings
├── test.py                  # Testing script
├── db/
│   └── create.py            # Database table creation script
├── evaluation/
│   └── dataset.py           # Dataset creation for evaluation
├── demo/                    # Demo scripts for text splitting
├── tests/                   # Test documents
├── output/                  # Output directory for generated responses
└── chroma_langchain_db/    # Chroma database storage
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/caslip/Learn-RAG.git
   cd Learn-RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   (Note: A requirements.txt file should be created with all necessary dependencies)

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_password
   MYSQL_DATABASE=rag_db
   SERPER_API_KEY=your_serper_api_key
   LLM_MODEL_NAME=qwen3:8b
   LLM_BASE_URL=http://localhost:11434
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   EMBEDDING_MODEL_DEVICE=cpu
   ```

## Usage

### 1. Database Setup

First, create the necessary database tables:
```bash
python db/create.py
```

### 2. Document Upload and Processing

Upload documents to the vector store:

```python
from chroma_db import vector_store

# Upload a document (supports .txt, .pdf, .docx)
vector_store.upload_document("./path/to/your/document.pdf")

# List all documents
vector_store.list_documents()

# Delete a document (logical deletion)
vector_store.delete_document("./path/to/your/document.pdf")
```

### 3. RAG Query

Use the RAG agent to answer questions:

```python
from rag import RAGAgent

agent = RAGAgent()
response = agent.invoke("What is RAG?")
print(response)
```

### 4. Using Different Vector Stores

The project supports both Chroma and FAISS vector stores:

```python
# Chroma (default)
from chroma_db import vector_store

# FAISS
from vector_db.faiss_db import vector_store
```

### 5. Evaluation

Evaluate the RAG system performance:

```python
from evaluation.dataset import dataset
# Use the dataset for evaluation metrics
```

## Key Components

### ChromaVectorStore (`chroma_db.py`)

- Manages document upload, update, and deletion
- Handles document chunking and vectorization
- Provides similarity search functionality
- Supports document versioning

### FaissVectorStore (`vector_db/faiss_db.py`)

- Alternative vector store using FAISS for efficient similarity search
- Manages document upload, update, and deletion
- Handles document chunking and vectorization
- Supports document versioning

### RAGAgent (`rag.py`)

- Implements a RAG system with Ollama LLM
- Combines retrieved context with web search capabilities
- Generates structured responses using Pydantic models
- Saves responses to JSON files

### Document Management (`doc_table.py`)

- Defines SQLAlchemy models for documents and chunks
- Handles database operations for document metadata
- Supports document versioning and logical deletion

## Configuration

The `config.py` file manages all configuration settings:

- MySQL database connection
- Vector database settings
- LLM model parameters
- Embedding model configuration
- RAG-specific parameters

## Testing

Run the test script to verify functionality:

```bash
python test.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
