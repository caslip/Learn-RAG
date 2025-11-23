# Filename: chroma_vector_store.py
import os
import uuid
from datetime import datetime
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Note: The imported class names are plural!
from doc_table import get_db_session, Documents, DocumentChunks


class ChromaVectorStore:
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self._vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self._embeddings,
            persist_directory="./chroma_langchain_db",
        )
        self.session = get_db_session()

    # ====================== Upload/Update Document ======================
    def upload_document(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path = os.path.abspath(file_path)
        file_name = os.path.basename(file_path)

        # Load + chunk
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits: List[Document] = splitter.split_documents(docs)
        print(f"[{file_name}] split into {len(splits)} chunks")

        self._add_or_update_chunks(file_name, file_path, splits)

    def _add_or_update_chunks(self, file_name: str, file_path: str, splits: List[Document]):
        session = self.session

        # Query current valid document by file_name (update if exists)
        doc_record = session.query(Documents).filter(
            Documents.file_name == file_name,
        ).first()

        if doc_record:
            if doc_record.is_deleted:
                print(f"Found deleted document '{file_name}', resurrecting and updating...")
            doc_record.is_deleted = False
            # Update: version +1
            old_version = doc_record.version
            doc_record.version += 1
            # doc_record.updated_at = datetime.utcnow()
            session.flush()
            new_version = doc_record.version
            print(f"Updating document '{file_name}' v{old_version} → v{new_version}")
        else:
            # Add new
            doc_record = Documents(
                file_name=file_name,
                file_path=file_path,
                version=1,
                is_deleted=False
            )
            session.add(doc_record)
            session.flush()  # Get id
            new_version = 1
            print(f"Adding new document '{file_name}' v{new_version}")

        session.query(DocumentChunks).filter(
            DocumentChunks.document_id == doc_record.id
        ).delete(synchronize_session=False)  # Directly physically delete old chunk records
        # Generate deterministic vector_id and write
        texts, metadatas, vector_ids, chunk_records = [], [], [], []

        for i, split in enumerate(splits):
            vector_id = f"{doc_record.id}_v{new_version}_{i}"

            vector_ids.append(vector_id)
            texts.append(split.page_content)
            metadatas.append({
                "source": file_name,
                "file_name": file_name,
                "doc_id": doc_record.id,
                "version": new_version,
                "chunk_index": i,
                "start_index": split.metadata.get("start_index", 0),
            })

            chunk_records.append(DocumentChunks(
                document_id=doc_record.id,
                chunk_index=i,
                vector_id=vector_id
            ))

        # Write to Chroma (same ID automatically overwrites = upsert)
        self._vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=vector_ids
        )

        session.bulk_save_objects(chunk_records)
        session.commit()
        print(f"Successfully wrote {len(splits)} chunks (version v{new_version})")

    # ====================== Delete Document ======================
    def delete_document(self, file_path: str) -> bool:
        file_name = os.path.basename(file_path)
        session = self.session
        doc = session.query(Documents).filter(
            Documents.file_name == file_name,
            Documents.is_deleted == False
        ).first()

        if not doc:
            print(f"Document not found: {file_name}")
            return False

        doc.is_deleted = True
        # doc.updated_at = datetime.utcnow()
        session.commit()
        print(f"Document '{file_name}' has been logically deleted (current version v{doc.version})")
        return True

    # ====================== Search (automatically filters old versions and deleted) ======================
    def similarity_search(self, query: str, k: int = 6) -> List[Tuple[Document, float]]:
        candidates = self._vector_store.similarity_search_with_score(query, k=k*4)
        valid: List[Tuple[Document, float]] = []
        session = self.session

        for doc, score in candidates:
            meta = doc.metadata
            doc_id = meta.get("doc_id")
            version = meta.get("version")
            if not doc_id or version is None:
                continue

            current = session.query(Documents).filter(
                Documents.id == doc_id,
                Documents.is_deleted == False
            ).first()

            if current and current.version == version:
                valid.append((doc, score))
                if len(valid) >= k:
                    break
        return valid[:k]

    def retrivel(self, question: str, k: int = 6):
        return self.similarity_search(question, k=k)

    # ====================== Physically clean up old versions (run once a month) ======================
    def cleanup_old_versions(self):
        session = self.session
        old_chunks = session.query(DocumentChunks).join(Documents).filter(
            Documents.is_deleted == False,
            Documents.version != DocumentChunks.version  # Note: This is DocumentChunks.version (delete this line if this column doesn't exist)
        ).all()

        if old_chunks:
            ids = [c.vector_id for c in old_chunks]
            self._vector_store._collection.delete(ids=ids) # type: ignore
            print(f"Physically deleted {len(ids)} old version vectors")
        else:
            print("No old versions to clean up")

    # ====================== List current valid documents ======================
    def list_documents(self):
        docs = self.session.query(Documents).filter(Documents.is_deleted == False).all()
        for d in docs:
            print(f"{d.file_path} (v{d.version}) → {d.file_path}")


# Singleton export
vector_store = ChromaVectorStore()


# Test
if __name__ == "__main__":
    # vector_store.upload_document("./tests/demo.txt")
    # vector_store.upload_document("./tests/许三观卖血记.txt")  # Upload again → auto v2
    vector_store.delete_document("./tests/许三观卖血记.txt")  # Upload again → auto v2
    # vector_store.delete_document("demo.txt")
    # vector_store.delete_document("demo.txt")

    results = vector_store.similarity_search("What is RAG", k=4)
    for doc, score in results:
        print(f"Score: {score:.4f}\n{doc.page_content[:300]}\n{'-'*80}")
