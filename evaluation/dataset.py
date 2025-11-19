from datasets import Dataset
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..rag import agent
from db import vector_store

questions = ['Transformer改变了什么?',
             '深度学习在哪个分支']
ground_truths = ['Transformer模型彻底改变了自然语言处理领域',
                 '深度学习是机器学习的一个重要分支。']
answers = []
contexts = []

for query in questions:
    for step in agent._agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        answers.append(step["messages"][-1])
    contexts = vector_store.similarity_search(query)

dataset = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

dataset = Dataset.from_dict(dataset)
