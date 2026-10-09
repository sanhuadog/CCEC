# multi_source_kb/REAP_multi_source_retriever.py
import json
import os

# 强制 Hugging Face 和 Transformers 使用本地已下载的模型，禁用联网校验
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import hashlib
import tempfile
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

# 引入网络与学术检索子模块 (确保 search_network.py 和 search_academic.py 在同一目录下)
try:
    from .search_network import search_network_context
    from .search_academic import search_academic_paper
except ImportError:
    # 兼容直接单文件运行或绝对路径导入
    from search_network import search_network_context
    from search_academic import search_academic_paper

# 💡 动态获取当前脚本所在文件夹的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 💡 本地知识库数据路径
DEFAULT_INDEX_PATH = os.path.join(BASE_DIR, "data", "commonsense.index")
DEFAULT_MAP_PATH = os.path.join(BASE_DIR, "data", "commonsense_map.json")
DEFAULT_CACHE_PATH = os.path.join(BASE_DIR, "data", "reap_retriever_cache.json")

# 本地 Embedder 向量模型路径 (默认切换为英文版 BAAI/bge-small-en-v1.5)
LOCAL_EMBEDDER_PATH = os.getenv("EMBEDDER_MODEL_PATH", "BAAI/bge-small-en-v1.5")


class MultiSourceRetriever:
    """
    REAP 框架专用多源异构知识检索器（英文适配版）：
    融合 本地向量知识库 (FAISS)、学术文献 (CrossRef/Semantic Scholar/ArXiv) 与 网络语境 (DuckDuckGo/SerpAPI)。
    """
    def __init__(self, 
                 index_path: str = DEFAULT_INDEX_PATH, 
                 map_path: str = DEFAULT_MAP_PATH,
                 cache_path: str = DEFAULT_CACHE_PATH):
        
        print("Initializing REAP Multi-Source Heterogeneous Knowledge Retriever...")
        
        # 1. 确保 data 目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        self.index_path = index_path
        self.map_path = map_path
        self.cache_path = cache_path
        
        # 2. 读取 FAISS 索引（增加二进制流与临时文件跨平台兼容）
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                print(f"  └─ Successfully loaded REAP FAISS Index: {index_path}")
            except Exception:
                print("  └─ Path compatibility issue detected, loading FAISS index via system temp file...")
                try:
                    with open(index_path, "rb") as f:
                        index_bytes = f.read()
                    
                    temp_dir = tempfile.gettempdir()
                    temp_index_path = os.path.join(temp_dir, "temp_reap_faiss.index")
                    with open(temp_index_path, "wb") as f:
                        f.write(index_bytes)
                    
                    self.index = faiss.read_index(temp_index_path)
                    print("  └─ ✅ Successfully loaded FAISS index via temp file!")
                except Exception as inner_e:
                    print(f"  └─ ❌ REAP FAISS Index Load Failed: {inner_e}")
                    self.index = None
        else:
            self.index = None
            print(f"  └─ ⚠️ Warning: FAISS index file not found: {index_path}, skipping vector retrieval.")

        # 3. 读取 ID Map 映射字典
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
            print(f"  └─ Successfully loaded Commonsense Map ({len(self.id_map)} entries)")
        else:
            self.id_map = {}
            print(f"  └─ ⚠️ Warning: Map file not found: {map_path}")

        # 4. 加载向量编码器 (使用英文 Embedding 模型)
        print(f"  └─ Loading Embedding Model: {LOCAL_EMBEDDER_PATH}")
        try:
            self.embedder = SentenceTransformer(LOCAL_EMBEDDER_PATH)
        except Exception as e:
            print(f"  └─ ⚠️ Local model load failed, attempting online fallback to English model: {e}")
            self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # 5. 读取本地磁盘缓存
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"  └─ ⚠️ Cache read error: {e}, initializing empty cache.")
                self.cache = {}
                
        print("✅ REAP English Multi-Source Retriever Initialized Successfully!\n")

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Cache saving error: {e}")

    def _get_cache_key(self, task_type: str, query: str) -> str:
        return hashlib.md5(f"reap_en_{task_type}_{query}".encode("utf-8")).hexdigest()

    def search_local_kb(self, query: str, top_k: int = 3) -> list:
        """从本地向量知识库检索支持证据"""
        results = []
        if self.index is None or not self.id_map:
            return results
            
        try:
            query_vec = self.embedder.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(np.array(query_vec).astype("float32"), k=top_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and str(idx) in self.id_map:
                    text = self.id_map[str(idx)]["text"]
                    results.append(f"[Local Wiki Knowledge / Confidence 0.85] (Similarity: {score:.2f}) {text}")
        except Exception as e:
            print(f"⚠️ REAP Local KB retrieval error: {e}")
            
        return results

    def retrieve_for_reap_node(self, task_type: str, query: str, max_chars: int = 1500) -> list:
        """
        根据 REAP 节点的任务类型 (LEAF_REALITY_CHECK / LEAF_CONTEXT_CHECK 等) 执行路由多源检索
        """
        cache_key = self._get_cache_key(task_type, query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        evidence_list = []

        # 针对 Task Type 进行策略路由多线程并发
        with ThreadPoolExecutor(max_workers=3) as executor:
            if task_type in ["LEAF_REALITY_CHECK", "COMMON_SENSE_CHECK"]:
                # 现实常识校验：主攻学术文献与本地知识库，网络补位
                future_academic = executor.submit(search_academic_paper, query, 2)
                future_local = executor.submit(self.search_local_kb, query, 2)
                future_net = executor.submit(search_network_context, query, 1)

                academic_raw = future_academic.result() or []
                local_ev = future_local.result() or []
                net_raw = future_net.result() or []

                academic_weighted = [
                    f"[Academic Literature / Confidence 0.90] {item}"
                    if "rate limit" not in str(item).lower()
                    else f"[Academic Literature / Low Confidence (Rate Limited)] {item}" 
                    for item in academic_raw
                ]
                net_weighted = [f"[Web Context / Confidence 0.55] {item}" for item in net_raw]

                evidence_list.extend(academic_weighted)
                evidence_list.extend(local_ev)
                evidence_list.extend(net_weighted)

            elif task_type in ["LEAF_CONTEXT_CHECK", "CONTEXT_CHECK"]:
                # 语境/壳核验：主攻网络检索与本地知识库
                future_net = executor.submit(search_network_context, query, 2)
                future_local = executor.submit(self.search_local_kb, query, 2)

                net_raw = future_net.result() or []
                local_ev = future_local.result() or []

                net_weighted = [f"[Web Context / Confidence 0.75] {item}" for item in net_raw]

                evidence_list.extend(net_weighted)
                evidence_list.extend(local_ev)

            else:
                # 默认三路并发保底
                future_academic = executor.submit(search_academic_paper, query, 1)
                future_local = executor.submit(self.search_local_kb, query, 1)
                future_net = executor.submit(search_network_context, query, 1)

                academic_raw = future_academic.result() or []
                local_ev = future_local.result() or []
                net_raw = future_net.result() or []

                academic_weighted = [f"[Academic Literature / Confidence 0.90] {item}" for item in academic_raw]
                net_weighted = [f"[Web Context / Confidence 0.55] {item}" for item in net_raw]

                evidence_list.extend(academic_weighted)
                evidence_list.extend(local_ev)
                evidence_list.extend(net_weighted)

        # 长度截断保护 (针对英文扩展至 1500 字符)
        total_text = "\n".join(evidence_list)
        if len(total_text) > max_chars:
            total_text = total_text[:max_chars] + "...[Truncated]"
            evidence_list = [total_text]

        # 写入缓存
        self.cache[cache_key] = evidence_list
        self._save_cache()

        return evidence_list


# ==========================================
# 💡 导出统一接口 (完美契合 core.py / 外部系统)
# ==========================================
_GLOBAL_RETRIEVER_INSTANCE = None

def get_retriever_instance() -> MultiSourceRetriever:
    """获取单例检索器对象"""
    global _GLOBAL_RETRIEVER_INSTANCE
    if _GLOBAL_RETRIEVER_INSTANCE is None:
        _GLOBAL_RETRIEVER_INSTANCE = MultiSourceRetriever()
    return _GLOBAL_RETRIEVER_INSTANCE


def search_knowledge_base(query: str, task_type: str = "LEAF_REALITY_CHECK", top_k: int = 3, **kwargs) -> list:
    """
    核心暴露接口：供 core.py 中的 retrieve_context 调用
    """
    retriever = get_retriever_instance()
    
    # 优先路由至特化对撞检索函数
    raw_evidences = retriever.retrieve_for_reap_node(task_type=task_type, query=query)
    
    # 组装为 core.py 期望的标准字典格式或列表格式
    formatted_hits = []
    for idx, ev in enumerate(raw_evidences):
        formatted_hits.append({
            "text": ev,
            "source": "REAP_MultiSource_KB",
            "score": 0.85
        })
        
    return formatted_hits