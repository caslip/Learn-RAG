from text_splitter import create_chunks_by_length, create_chunks_by_semantics
# import weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

# Initialize the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Specify device if needed, e.g., 'cuda' for GPU
)

# Get the embedding dimension from the embeddings model
embedding_dim = embeddings.client.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# texts = create_chunks_by_length("./demo.txt")
texts = create_chunks_by_semantics("../demo.txt")

vector_store.add_texts(texts)

results = vector_store.similarity_search(
    "Transformer",
    k=2,
)
print("*************************************************")
print(results)
for res in results:
    print(f"* {res.page_content}")
