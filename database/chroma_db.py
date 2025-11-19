from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import uuid
from ..doc_table import get_db_session, create_tables, Documents, DocumentChunks

class ChromaVectorStore:

    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Specify device if needed, e.g., 'cuda' for GPU
        )
        self._vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self._embeddings,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )

        self.session = get_db_session()
    
    def split_by_length(self, file_path):
        """
        Docstring
            The text splitter chunk size default setting = 1000,
            overlap = 200,
        :param self: None
        :param file_path: 
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.doc', '.docx']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        # 加载文档
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        print(f"Split blog post into {len(all_splits)} sub-documents.")

        document_ids = self._vector_store.add_documents(all_splits)
        print(document_ids[:3])
        # store it in database

    def retrivel(self, question : str, k : int =3):
        return self._vector_store.similarity_search_with_relevance_scores(question, k=k)
    
# Create an instance of ChromaVectorStore to be imported by other modules
vector_store = ChromaVectorStore()

if __name__ == "__main__":
    # Example usage (optional, can be removed or modified)
    pdf_file_path = os.path.join(os.path.dirname(__file__), "../tests/", "demo.txt")
    docs = vector_store.split_by_length(pdf_file_path)

    vector_store.split_by_length(docs)

    ret = vector_store.retrivel("Transformer")
    print(ret)
