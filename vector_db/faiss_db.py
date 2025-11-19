# 文件名: faiss_vector_store.py
import os
import uuid
from datetime import datetime
from typing import List, Tuple

import faiss
import numpy as np
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 与 chroma 版共用同一张表
from doc_table import get_db_session, Documents, DocumentChunks


class FaissVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )

        # FAISS 索引（L2 距离）
        embedding_dim = len(self._embeddings.embed_query("hello world"))
        self._index = faiss.IndexFlatL2(embedding_dim)
        self._index_to_id = {}           # faiss internal_id → vector_id (str)
        self._id_to_index = {}           # vector_id → faiss internal_id
        self._vectors = np.empty((0, embedding_dim), dtype=np.float32)  # 实际向量数组

        # 持久化路径
        self.persist_dir = "./faiss_langchain_db"
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.mapping_path = os.path.join(self.persist_dir, "faiss_mapping.npy")

        self._load_index()  # 启动时尝试加载已有索引

        self.session = get_db_session()

    # ====================== 持久化加载/保存 ======================
    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self._index = faiss.read_index(self.index_path)
            mapping = np.load(self.mapping_path, allow_pickle=True).item()
            self._index_to_id = mapping["index_to_id"]
            self._id_to_index = {v: k for k, v in self._index_to_id.items()}
            # 重建向量数组（faiss 读取后可直接访问）
            if self._index.ntotal > 0:
                self._vectors = faiss.vector_to_array(self._index.xb)
            print(f"加载 FAISS 索引成功，包含 {self._index.ntotal} 条向量")
        else:
            print("未发现已有 FAISS 索引，从零开始")

    def _save_index(self):
        faiss.write_index(self._index, self.index_path)
        mapping = {
            "index_to_id": self._index_to_id,
        }
        np.save(self.mapping_path, mapping)
        print(f"FAISS 索引已保存 → {self.index_path}")

    # ====================== 上传/更新文档 ======================
    def upload_document(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_path = os.path.abspath(file_path)
        file_name = os.path.basename(file_path)

        # 1. 加载 + 分块
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

        # 2. 查询或创建文档记录
        doc_record = session.query(Documents).filter(
            Documents.file_name == file_name
        ).first()

        if doc_record:
            if doc_record.is_deleted:
                print(f"检测到已删除的文档「{file_name}」，正在复活并更新...")
            doc_record.is_deleted = False
            old_version = doc_record.version
            doc_record.version += 1
            session.flush()
            new_version = doc_record.version
            print(f"更新文档「{file_name}」 v{old_version} → v{new_version}")
        else:
            doc_record = Documents(
                file_name=file_name,
                file_path=file_path,
                version=1,
                is_deleted=False
            )
            session.add(doc_record)
            session.flush()
            new_version = 1
            print(f"新增文档「{file_name}」 v{new_version}")

        # 3. 删除旧版本的所有 chunks（SQL + FAISS）
        old_chunks = session.query(DocumentChunks).filter(
            DocumentChunks.document_id == doc_record.id
        ).all()

        old_vector_ids = [c.vector_id for c in old_chunks]
        if old_vector_ids:
            # 从 FAISS 中删除
            old_internal_ids = [self._id_to_index[vid] for vid in old_vector_ids if vid in self._id_to_index]
            if old_internal_ids:
                # FAISS 删除需要重建索引（Flat 索引只能 rebuild）
                mask = np.ones(self._index.ntotal, dtype=bool)
                mask[old_internal_ids] = False
                remain_vectors = self._vectors[mask]
                # 重建索引
                new_index = faiss.IndexFlatL2(self._vectors.shape[1])
                new_index.add(remain_vectors)
                self._index = new_index
                self._vectors = remain_vectors

                # 重建映射
                new_index_to_id = {}
                new_id_to_index = {}
                cur = 0
                for old_idx in range(len(mask)):
                    if mask[old_idx]:
                        old_vid = self._index_to_id[old_idx]
                        new_index_to_id[cur] = old_vid
                        new_id_to_index[old_vid] = cur
                        cur += 1
                self._index_to_id = new_index_to_id
                self._id_to_index = new_id_to_index

            # 删除 SQL 记录
            session.query(DocumentChunks).filter(
                DocumentChunks.document_id == doc_record.id
            ).delete(synchronize_session=False)

        # 4. 添加新版本 chunks
        texts = []
        metadatas = []
        vector_ids = []
        chunk_records = []

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
                vector_id=vector_id,
                version=new_version  # 如果你的 DocumentChunks 表有 version 列的话
            ))

        # 计算 embeddings 并加入 FAISS
        embeddings = self._embeddings.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype('float32')

        # FAISS add
        start_idx = self._index.ntotal
        self._index.add(embeddings_np)
        self._vectors = np.vstack((self._vectors, embeddings_np)) if self._vectors.size else embeddings_np

        # 更新映射
        for offset, vid in enumerate(vector_ids):
            internal_id = start_idx + offset
            self._index_to_id[internal_id] = vid
            self._id_to_index[vid] = internal_id

        # 写入 SQL
        session.bulk_save_objects(chunk_records)
        session.commit()
        self._save_index()
        print(f"成功写入 {len(splits)} 条 chunks（版本 v{new_version}）")

    # ====================== 删除文档（逻辑删除） ======================
    def delete_document(self, file_path: str) -> bool:
        file_name = os.path.basename(file_path)
        doc = self.session.query(Documents).filter(
            Documents.file_name == file_name,
            Documents.is_deleted == False
        ).first()

        if not doc:
            print(f"未找到文档: {file_name}")
            return False

        doc.is_deleted = True
        self.session.commit()
        print(f"文档「{file_name}」已逻辑删除（当前版本 v{doc.version}）")
        return True

    # ====================== 检索（只返回最新有效版本） ======================
    def similarity_search(self, query: str, k: int = 6) -> List[Tuple[Document, float]]:
        if self._index.ntotal == 0:
            return []

        query_vec = self._embeddings.embed_query(query)
        query_np = np.array([query_vec]).astype('float32')
        distances, indices = self._index.search(query_np, k * 4)  # 多取一点再过滤

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            vector_id = self._index_to_id.get(idx)
            if not vector_id:
                continue

            # 从 SQL 查最新版本
            parts = vector_id.split('_v')
            if len(parts) < 2:
                continue
            doc_id_str = parts[0]
            version_str = parts[1].split('_')[0]
            try:
                doc_id = int(doc_id_str)
                version = int(version_str)
            except ValueError:
                continue

            current_doc = self.session.query(Documents).filter(
                Documents.id == doc_id,
                Documents.is_deleted == False
            ).first()

            if current_doc and current_doc.version == version:
                # 重新构造 Document
                chunk = self.session.query(DocumentChunks).filter(
                    DocumentChunks.vector_id == vector_id
                ).first()
                if not chunk:
                    continue

                # 这里我们没有存原始 page_content，所以只能从 metadata 里拿部分信息
                # 实际项目中建议把 page_content 也存进 DocumentChunks 表，或者接受只能返回 metadata
                # 下面用一个占位方式，你可以自行扩展表结构
                doc = Document(
                    page_content=f"[Chunk {chunk.chunk_index} from {current_doc.file_name}]",
                    metadata={
                        "source": current_doc.file_name,
                        "file_name": current_doc.file_name,
                        "doc_id": doc_id,
                        "version": version,
                        "chunk_index": chunk.chunk_index,
                    }
                )
                candidates.append((doc, float(dist)))

            if len(candidates) >= k:
                break

        return candidates[:k]

    def retrivel(self, question: str, k: int = 6):
        return self.similarity_search(question, k=k)

    # ====================== 物理清理旧版本 ======================
    def cleanup_old_versions(self):
        """删除已被新版本取代的向量（每月跑一次）"""
        # 这里逻辑和 Chroma 版几乎一样，只是删除方式不同
        # 由于我们已经在上传时即时删除了旧向量，这里其实已经没有垃圾
        print("FAISS 版在上传时已即时删除旧版本，无需额外清理")

    # ====================== 列出当前有效文档 ======================
    def list_documents(self):
        docs = self.session.query(Documents).filter(Documents.is_deleted == False).all()
        for d in docs:
            print(f"{d.file_path} (v{d.version})")


# 单例导出（项目中统一导入这个实例）
vector_store = FaissVectorStore()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # vector_store.upload_document("./tests/demo.pdf")
    # vector_store.upload_document("./tests/demo.pdf")  # 再次上传 → v2
    # vector_store.delete_document("./tests/demo.pdf")

    results = vector_store.similarity_search("Transformer 模型", k=5)
    for doc, score in results:
        print(f"Score: {score:.4f} | Source: {doc.metadata.get('source')}")
        print(f"Content preview: {doc.page_content[:200]}\n{'-'*80}")