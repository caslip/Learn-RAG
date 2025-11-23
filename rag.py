import os
import json
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent

import sys
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vector_db.chroma_db import vector_store
from pydantic import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv

from tools import retrieve_context, search

load_dotenv()
os.getenv("SERPER_API_KEY")

class RAGResponse(BaseModel):
    """
    StandardOutput of RAGAgent 
    """
    answer: str = Field(description="The answer of user's question")
    score: float = Field(description="The score of retrieve context relevance," \
    " directly adopt the relevance score," \
    "if the score is not only one, get average score of each score")
    quotation: str = Field(description="The information of retrieval context, " \
    "including the metadata of the context")
    note:str = Field(description="The everything you want to remind user")

class RAGAgent:
    def __init__(self) -> None:
        self._llm = ChatOllama(
            model="qwen3:8b"
        )
        self.vector_store = vector_store

        # @tool
        # def search(query:str):
        #     """Useful for when you need to search Google for up-to-date information or real-time events."""
        #     search = GoogleSerperAPIWrapper(k=10)
        #     result = search.results(query)

        #     return result

        # @tool(response_format="content_and_artifact")
        # def retrieve_context(query: str):
        #     """Retrieve information to help answer a query."""
        #     retrieved_docs = vector_store.retrivel(query, k=2)
        #     serialized = "\n\n".join(
        #         (f"Source: {doc.metadata}\nContent: {doc.page_content} \n Relevance Score: {score}")
        #         for doc, score in retrieved_docs
        #     )
        #     return serialized, retrieved_docs

        tools = [retrieve_context, search]

        prompt = (
            "You have access to a tool that retrieves context from a blog post. "
            "Use the tool to help answer user queries."
            "Use the search tool when you don't know something" \
            "Set the output formular is json"

        )

        self._agent = create_agent(
            self._llm, 
            tools, 
            system_prompt=prompt, 
            response_format=RAGResponse,
        )

    def invoke(self, query: str):
        """直接收集流式输出的所有数据"""
        collected_messages = []
        
        for event in self._agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()

        # res = self._agent.invoke(
        #     {"messages": [{"role": "user", "content": query}]},
        #     stream_mode="values",
        # )   

        # structured = res["structured_response"]
        # print(structured)
        # print(type(structured))

        # # 将 RAGResponse 对象的所有属性保存到 JSON 文件
        # response_dict = structured.model_dump() # 将 Pydantic 模型转换为字典
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"output/rag_response_{timestamp}.json"
        
        # # 确保输出目录存在
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # with open(filename, 'w', encoding='utf-8') as f:
        #     json.dump(response_dict, f, ensure_ascii=False, indent=4)
            
        # print(f"Response saved to {filename}")

        # return structured


if __name__ == "__main__":
    query = "Transformer"
    agent = RAGAgent()

    # Example of how to use the output_parser (assuming agent.invoke returns a string)
    # For demonstration, let's simulate a direct call to output_parser
    # In a real scenario, you would get 'llm_output_string' from the agent's invocation
    agent.invoke(query)
