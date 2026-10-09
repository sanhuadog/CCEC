# search_academic.py
import re
import requests

def search_academic_paper(query: str, max_results: int = 2) -> list:
    """
    学术文献知识库接口 (基于 CrossRef API - 高可用免 Key 极速响应)
    :param query: 科学断言/专业术语 (如 "fish respiration gills mechanics")
    :param max_results: 返回论文摘要条数
    :return: 学术论文标题与摘要列表
    """
    results = []
    
    # 构造请求头 (遵守 CrossRef 的 Polite Pool 规范，附带 User-Agent)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    url = f"https://api.crossref.org/works?query={query}&rows={max_results}"

    try:
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            for item in items:
                # 提取标题
                titles = item.get("title", [])
                title = titles[0] if titles else "未知标题"
                
                # 提取摘要并清洗 XML/JATS 标签
                raw_abstract = item.get("abstract", "")
                if raw_abstract:
                    # 使用正则剥离所有 JATS/XML 标签（如 <jats:p>, <jats:bold> 等）
                    clean_abstract = re.sub(r"<[^>]+>", "", raw_abstract).strip()
                else:
                    clean_abstract = "（已在 CrossRef 索引集中匹配到该学术条目，但源出版商未公开摘要文本）"
                
                if title:
                    results.append(f"【学术文献-CrossRef】标题: {title}\n摘要: {clean_abstract[:300]}...")
            
            if results:
                return results

        else:
            print(f"⚠️ CrossRef 响应状态码异常: {response.status_code}")

    except Exception as e:
        print(f"⚠️ CrossRef 学术接口请求异常: {e}")

    # 优雅降级兜底
    return [f"【学术文献】已发起对“{query}”的 CrossRef 索引校验，未成功检索到匹配文献摘要。"]


if __name__ == "__main__":
    # 测试学术检索
    print("正在测试 CrossRef 学术知识库检索...")
    test_res = search_academic_paper(query="fish respiration mechanics", max_results=2)
    print("\n学术知识库测试输出：")
    for r in test_res:
        print(r)