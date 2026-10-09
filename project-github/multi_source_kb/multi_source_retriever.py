# stage1-retriever.py
import json
import os
# 强制 Hugging Face 和 Transformers 使用本地已下载的模型，禁用联网校验
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import hashlib
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 💡 动态获取当前脚本所在文件夹 (multi_source_kb) 的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 💡 拼接绝对路径
DEFAULT_INDEX_PATH = os.path.join(BASE_DIR, "data", "commonsense.index")
DEFAULT_MAP_PATH = os.path.join(BASE_DIR, "data", "commonsense_map.json")
DEFAULT_CACHE_PATH = os.path.join(BASE_DIR, "data", "retriever_cache.json")

# 💡 切换为英文 Embedding 模型路径
LOCAL_EMBEDDER_PATH = os.getenv("EMBEDDER_MODEL_PATH", "BAAI/bge-small-en-v1.5")


class MultiSourceRetriever:
    def __init__(self, 
                 index_path: str = DEFAULT_INDEX_PATH, 
                 map_path: str = DEFAULT_MAP_PATH,
                 cache_path: str = DEFAULT_CACHE_PATH):
        
        print("正在初始化英文纯本地离线常识检索器 (英文Wiki向量库 + 缓存)...")
        
        # 1. 确保 data 目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        self.index_path = index_path
        self.map_path = map_path
        self.cache_path = cache_path
        
        # 2. 读取 FAISS 索引（使用 tempfile 解决跨平台路径兼容性）
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                print(f"  └─ 成功加载 FAISS 索引: {index_path}")
            except Exception:
                print(f"  └─ 检测到路径兼容性问题，正在通过临时文件加载 FAISS 索引...")
                try:
                    with open(index_path, "rb") as f:
                        index_bytes = f.read()
                    
                    temp_dir = tempfile.gettempdir()
                    temp_index_path = os.path.join(temp_dir, "temp_faiss_commonsense.index")
                    with open(temp_index_path, "wb") as f:
                        f.write(index_bytes)
                    
                    self.index = faiss.read_index(temp_index_path)
                    print(f"  └─ ✅ 通过临时文件成功加载 FAISS 索引！")
                except Exception as inner_e:
                    print(f"  └─ ❌ FAISS 索引加载失败: {inner_e}")
                    self.index = None
        else:
            self.index = None
            print(f"  └─ ⚠️ 警告: 未找到 FAISS 索引文件: {index_path}，将跳过本地向量检索。")

        # 3. 读取 ID Map 映射字典
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
            print(f"  └─ 成功加载常识 Map 映射 ({len(self.id_map)} 条)")
        else:
            self.id_map = {}
            print(f"  └─ ⚠️ 警告: 未找到 Map 映射文件: {map_path}")

        # 4. 加载本地英文向量编码器
        print(f"  └─ 正在加载英文 Embedding 模型: {LOCAL_EMBEDDER_PATH}")
        self.embedder = SentenceTransformer(LOCAL_EMBEDDER_PATH)
        
        # 5. 读取本地磁盘缓存
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"  └─ ⚠️ 缓存读取异常: {e}，初始化为空缓存。")
                self.cache = {}
                
        print("✅ 英文纯本地常识检索器准备就绪！(已完全屏蔽外网请求)\n")

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存缓存出错: {e}")

    def _get_cache_key(self, task_type: str, query: str) -> str:
        return hashlib.md5(f"{task_type}_{query}".encode("utf-8")).hexdigest()

    def _search_local_commonsense(self, query: str, top_k: int = 3) -> list:
        """从本地英文 Wiki 向量库检索"""
        results = []
        if self.index is None or not self.id_map:
            return results
            
        try:
            query_vec = self.embedder.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(np.array(query_vec).astype("float32"), k=top_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and str(idx) in self.id_map:
                    text = self.id_map[str(idx)]["text"]
                    results.append(f"【[Local English KB/Score:{score:.2f}]】{text}")
        except Exception as e:
            print(f"⚠️ 本地常识库检索出错: {e}")
            
        return results

    def retrieve_by_task(self, task_type: str, query: str, max_chars: int = 1200) -> list:
        # 1. 命中缓存直接返回
        cache_key = self._get_cache_key(task_type, query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        evidence_list = []

        # 2. 纯本地分流检索逻辑
        if task_type == "COMMON_SENSE_CHECK":
            # 常识/物理断言检查：提取 Top-2 本地知识
            local_ev = self._search_local_commonsense(query=query, top_k=2)
            evidence_list.extend(local_ev)

        elif task_type == "CONTEXT_CHECK":
            # 语境上下文检查：提取 Top-3 本地知识，增大检索覆盖面
            local_ev = self._search_local_commonsense(query=query, top_k=3)
            evidence_list.extend(local_ev)

        else:
            # 默认保底逻辑
            local_ev = self._search_local_commonsense(query=query, top_k=2)
            evidence_list.extend(local_ev)

        # 3. 结果截断与格式化处理
        total_text = "\n".join(evidence_list)
        if len(total_text) > max_chars:
            total_text = total_text[:max_chars] + "...[Truncated]"
            evidence_list = [total_text]

        # 4. 写入本地缓存
        self.cache[cache_key] = evidence_list
        self._save_cache()
        
        return evidence_list