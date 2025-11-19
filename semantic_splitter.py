from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import re

class SemanticTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化语义文本分割器
        
        Args:
            model_name: 使用的句子Transformer模型
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        self.model.eval()  # 设置为评估模式
        
    def get_sentence_embeddings(self, sentences):
        """获取句子的嵌入向量"""
        embeddings = self.model.encode(sentences)
        return embeddings
    
    def mean_pooling(self, model_output, attention_mask):
        """平均池化获取句子表示"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def split_sentences(self, text):
        """将文本分割成句子"""
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            )
        sentences = text_splitter.split_text(text)
        return sentences
    
    def semantic_split(self, text, similarity_threshold=0.7, min_chunk_size=2) -> list[str]:
        """
        基于语义相似度的文本分割
        
        Args:
            text: 输入文本
            similarity_threshold: 相似度阈值，低于此值则分割
            min_chunk_size: 最小块大小（句子数）
        """
        # 1. 分割成句子
        sentences = self.split_sentences(text)
        if len(sentences) <= min_chunk_size:
            print("text too short to split, returning original text.")
            return [text]
        
        # 2. 获取句子嵌入
        sentence_embeddings = self.get_sentence_embeddings(sentences)
        
        # 3. 计算相邻句子的相似度
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # 计算当前句子与前一句的相似度
            similarity = cosine_similarity(
                sentence_embeddings[i-1].reshape(1, -1),
                sentence_embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                # 相似度高，合并到当前块
                current_chunk.append(sentences[i])
            else:
                # 相似度低，开始新块
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
        
        # 添加最后一个块
        if len(current_chunk) >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# 使用示例
def main():
    # 示例文本
    sample_text = """
    人工智能是计算机科学的一个分支，它试图理解智能的本质并生产出新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。深度学习是机器学习的一个重要分支。
    它使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。近年来，深度学习在语音识别和计算机视觉领域取得了显著成果。
    
    另一方面，气候变化是当今世界面临的最严峻挑战之一。全球变暖导致极端天气事件频发，海平面上升威胁沿海城市。
    各国政府正在采取行动减少温室气体排放，推动可再生能源发展。太阳能和风能等清洁能源技术成本持续下降，使得可再生能源更具竞争力。
    
    回到技术话题，Transformer模型彻底改变了自然语言处理领域。它的自注意力机制能够捕捉长距离依赖关系。
    BERT、GPT等基于Transformer的模型在各种NLP任务上取得了突破性进展。这些模型通过预训练和微调范式，大大降低了NLP应用的门槛。
    """
    
    # 初始化分割器
    splitter = SemanticTextSplitter(100,20)
    
    # 进行语义分割
    chunks = splitter.semantic_split(sample_text, similarity_threshold=0.5)
    

    print(chunks)
    # 输出结果
    print("语义分割结果:")
    print("-" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"块 {i} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 50)

if __name__ == "__main__":
    main()