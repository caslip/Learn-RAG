import os
from typing import Iterable, List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_splitter import SemanticTextSplitter


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_chunks_by_length(filename, max_length=200):
    print("Encoding texts by length...")
    all_splits: Iterable[str] = []
    if os.path.splitext(filename)[-1] == '.txt':
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_length,
                chunk_overlap=25,
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            )
            splitted_texts = text_splitter.split_text(text)
            # print(f"Total chunks created: {len(splitted_texts)}")
            # for i, chunk in enumerate(splitted_texts):
            #     print(f"Chunk {i}: {chunk}\n")
            
            splitted_texts = [split for split in splitted_texts if split.strip()]
            all_splits.extend(splitted_texts)
            return all_splits

    return all_splits
            # embeddings = embedding_model.encode(splitted_texts)
            # print(f"Embeddings type: {type(embeddings)}")
            # return embeddings


def create_chunks_by_semantics(filename):
    all_splits: Iterable[str] = []
    print("Encoding texts by semantics...")
    if os.path.splitext(filename)[-1] == '.txt':
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            print(text)
            text_splitter = SemanticTextSplitter(
                chunk_size=100,
                chunk_overlap=20,
            )
            splitted_texts = text_splitter.semantic_split(text, similarity_threshold=0.7)
            
            splitted_texts = [split for split in splitted_texts if split.strip()]
            all_splits.extend(splitted_texts)
            return all_splits
    return all_splits
            

if "__name__" == "__main__":
    # Encode the length-based chunks to get numerical embeddings
    length_based_chunks = create_chunks_by_length(".\\demo.txt")
    embedding_length = embedding_model.encode(length_based_chunks) if length_based_chunks else None

    embedding_semantics = create_chunks_by_semantics(".\\demo.txt")

    if embedding_length is not None and embedding_semantics is not None:
        # Ensure both are tensors/lists of numbers for similarity calculation
        # The encode method typically returns numpy arrays, which similarity can handle.
        # If they are already lists of lists of numbers, encode might not be strictly necessary
        # but encoding ensures they are in the correct format for the similarity function.
        # However, similarity function expects batch of vectors. If encode already gives that,
        # then direct use is fine. Let's assume encode gives the right format.
        
        # The similarity function from sentence_transformers.util.cos_sim
        # expects two 2D tensors (or arrays) where each row is an embedding.
        # embedding_model.encode() should produce this.
        
        similarity = embedding_model.similarity(embedding_length, embedding_semantics) # pyright: ignore[reportArgumentType, reportCallIssue]
        print(f"Similarity between length-based and semantic-based embeddings: {similarity}")
    else:
        print("Could not calculate similarity due to missing embeddings.")
