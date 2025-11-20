from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from chroma_db import vector_store

@tool
def search(query:str):
    """Useful for when you need to search Google for up-to-date information or real-time events."""
    search = GoogleSerperAPIWrapper(k=10)
    result = search.results(query)

    return result

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.retrivel(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content} \n Relevance Score: {score}")
        for doc, score in retrieved_docs
    )
    return serialized, retrieved_docs