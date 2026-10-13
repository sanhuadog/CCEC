# build_commonsense_vector_db.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_vector_store():
    corpus_path = "data/commonsense_corpus.jsonl"
    index_out = "data/commonsense.index"
    map_out = "data/commonsense_map.json"
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"未找到常识文本文件：{corpus_path}，请先运行 download_corpus.py！")
    
    # 💡 替换为专门用于英文的 SOTA 轻量 Embedding 模型
    # BAAI/bge-small-en-v1.5 为 384 维向量
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"1. 加载英文 Embedding 模型 ({model_name})...")
    
    # 自动识别 GPU/CPU 进行加载
    embedder = SentenceTransformer(model_name)
    
    texts = []
    ids = []
    
    print("2. 读取本地常识文本数据...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.append(item["id"])
            texts.append(item["text"])
            
    print(f"3. 开始对 {len(texts)} 条英文常识数据生成向量 (Batch Encoding)...")
    # batch_size=256 保证显存利用率，normalize_embeddings=True 确保生成的向量长度为 1
    embeddings = embedder.encode(
        texts, 
        batch_size=256, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    
    dimension = embeddings.shape[1] # bge-small-en-v1.5 为 384 维
    print(f"向量维度: {dimension}")
    
    print("4. 构建 Faiss 高性能向量索引...")
    # IndexFlatIP 表示 Inner Product（内积），搭配单位归一化向量即为余弦相似度（Cosine Similarity）
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    # 5. 保存索引文件与 ID-文本 映射表到本地
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, index_out)
    
    # 保存下标（0, 1, 2...）与原文本的映射字典，检索时通过下标快速查出具体文本
    id_map = {str(idx): {"id": ids[idx], "text": texts[idx]} for idx in range(len(texts))}
    with open(map_out, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 英文常识向量数据库构建完成！")
    print(f" - 向量索引文件: {index_out}")
    print(f" - 文本映射字典: {map_out}")

if __name__ == "__main__":
    build_vector_store()
