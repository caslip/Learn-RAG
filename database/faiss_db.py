from langchain_text_splitters import RecursiveCharacterTextSplitter
# import weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
import faiss
import os

class FaissVectorStore:
    def __init__(self, model:str) -> None:

        self._embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cpu'} # Specify device if needed, e.g., 'cuda' for GPU
        )

        # Get the embedding dimension from the embeddings model
        embedding_dim = len(self._embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)

        self._vector_store = FAISS(
            embedding_function=self._embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

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
        return all_splits
    
    def store_splits(self, splits):
        document_ids = self._vector_store.add_documents(splits)
        print(document_ids[:3])

    def retrivel(self, question: str, filter: dict[str, str] = {'':''}):
        """
        retrivel 的 Docstring
        
        :param self: None
        :param question: your question
        :type question: str
        :param filter: According to your metadata to set the filter, such as {'category':'ai'}
        :type filter: dict[str, str]
        """
        return self._vector_store.similarity_search(question, filter=filter)

        
if __name__ == "__main__":
    vector_store = FaissVectorStore("all-MiniLM-L6-v2")

    # Corrected path to the PDF file
    pdf_file_path = os.path.join(os.path.dirname(__file__), "..", "pdf_scan.pdf")
    docs = vector_store.split_by_length(pdf_file_path)

    vector_store.store_splits(docs)

    ret = vector_store.retrivel("Transformer")
    print(ret)
