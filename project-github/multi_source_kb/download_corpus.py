# download_corpus.py
import os
import json
from datasets import load_dataset

print("正在加载 Hugging Face 官方英文维基百科常识库 (20231101.en)...")
# 使用 wikimedia 官方维护的英文维基百科切片
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
print(f"数据加载完成，共计 {len(dataset)} 条数据。")

# 打印第一条数据结构
sample = dataset[0]
print("【数据样例结构】:", sample)

output_dir = "data"
output_path = os.path.join(output_dir, "commonsense_corpus.jsonl")
os.makedirs(output_dir, exist_ok=True)

valid_count = 0
with open(output_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        # 官方数据集主要使用 "text" 字段存放主体内容
        text = item.get("text") or item.get("completion") or item.get("output") or ""
        
        # 备用拼接逻辑
        if not text:
            text = " ".join([str(v) for v in item.values() if isinstance(v, str)])
            
        # 英文以字符数/词数过滤，少于 10 个字符丢弃
        if len(text.strip()) < 10:
            continue
            
        record = {
            "id": f"wiki_en_cs_{i:06d}",
            "text": text.strip()[:500]  # 英文建议切取前 500 个字符（约 80-100 个单词），信息量更完整
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        valid_count += 1

print(f"✅ 转换完成！成功写入 {valid_count} 条英文常识数据存入 {output_path}")