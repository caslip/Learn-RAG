# 文件名: chroma_vector_store.py
import os
import uuid
from datetime import datetime
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 注意这里导入的是复数类名！
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

    # ====================== 上传/更新文档 ======================
    def upload_document(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_path = os.path.abspath(file_path)
        file_name = os.path.basename(file_path)

        # 加载 + 分块
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits: List[Document] = splitter.split_documents(docs)
        print(f"[{file_name}] 分割为 {len(splits)} 个 chunks")

        self._add_or_update_chunks(file_name, file_path, splits)

    def _add_or_update_chunks(self, file_name: str, file_path: str, splits: List[Document]):
        session = self.session

        # 按 file_name 查询当前有效文档（同名即更新）
        doc_record = session.query(Documents).filter(
            Documents.file_name == file_name,
        ).first()

        if doc_record:
            if doc_record.is_deleted:
                print(f"检测到已删除的文档「{file_name}」，正在复活并更新...")
            doc_record.is_deleted = False
            # 更新：版本 +1
            old_version = doc_record.version
            doc_record.version += 1
            # doc_record.updated_at = datetime.utcnow()
            session.flush()
            new_version = doc_record.version
            print(f"更新文档「{file_name}」 v{old_version} → v{new_version}")
        else:
            # 新增
            doc_record = Documents(
                file_name=file_name,
                file_path=file_path,
                version=1,
                is_deleted=False
            )
            session.add(doc_record)
            session.flush()  # 获取 id
            new_version = 1
            print(f"新增文档「{file_name}」 v{new_version}")

        session.query(DocumentChunks).filter(
            DocumentChunks.document_id == doc_record.id
        ).delete(synchronize_session=False)  # 直接物理删除旧 chunks 记录
        # 生成确定性 vector_id 并写入
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

        # 写入 Chroma（相同 ID 自动覆盖 = upsert）
        self._vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=vector_ids
        )

        session.bulk_save_objects(chunk_records)
        session.commit()
        print(f"成功写入 {len(splits)} 条 chunks（版本 v{new_version}）")

    # ====================== 删除文档 ======================
    def delete_document(self, file_path: str) -> bool:
        file_name = os.path.basename(file_path)
        session = self.session
        doc = session.query(Documents).filter(
            Documents.file_name == file_name,
            Documents.is_deleted == False
        ).first()

        if not doc:
            print(f"未找到文档: {file_name}")
            return False

        doc.is_deleted = True
        # doc.updated_at = datetime.utcnow()
        session.commit()
        print(f"文档「{file_name}」已逻辑删除（当前版本 v{doc.version}）")
        return True

    # ====================== 检索（自动过滤旧版本和已删除）======================
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

    # ====================== 物理清理旧版本（每月跑一次）======================
    def cleanup_old_versions(self):
        session = self.session
        old_chunks = session.query(DocumentChunks).join(Documents).filter(
            Documents.is_deleted == False,
            Documents.version != DocumentChunks.version  # 注意这里是 DocumentChunks.version（如果没这列就删掉这行）
        ).all()

        if old_chunks:
            ids = [c.vector_id for c in old_chunks]
            self._vector_store._collection.delete(ids=ids) # type: ignore
            print(f"物理删除 {len(ids)} 条旧版本向量")
        else:
            print("无旧版本可清理")

    # ====================== 列出当前有效文档 ======================
    def list_documents(self):
        docs = self.session.query(Documents).filter(Documents.is_deleted == False).all()
        for d in docs:
            print(f"{d.file_path} (v{d.version}) → {d.file_path}")


# 单例导出
vector_store = ChromaVectorStore()


# 测试
if __name__ == "__main__":
    # vector_store.upload_document("./tests/demo.txt")
    # vector_store.upload_document("./tests/许三观卖血记.txt")  # 再次上传 → 自动 v2
    vector_store.delete_document("./tests/许三观卖血记.txt")  # 再次上传 → 自动 v2
    # vector_store.delete_document("demo.txt")
    # vector_store.delete_document("demo.txt")

    results = vector_store.similarity_search("RAG 是什么", k=4)
    for doc, score in results:
        print(f"Score: {score:.4f}\n{doc.page_content[:300]}\n{'-'*80}")