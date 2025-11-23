from datasets import Dataset
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import agent
from ..vector_db.chroma_db import vector_store

questions = ['What did the Transformer change?',
             'Which branch does deep learning belong to?']
ground_truths = ['The Transformer model has revolutionized the field of natural language processing',
                 'Deep learning is an important branch of machine learning.']
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
