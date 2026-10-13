# search_network.py
from ddgs import DDGS

def search_network_context(query: str, context_anchor: str = "", max_results: int = 2) -> list:
    """
    网络知识搜索接口 (适配最新 ddgs 库)
    :param query: 核心查询词 (如 "鱼类用鳃呼吸")
    :param context_anchor: 语境锚点 (如 "朋友圈养生")
    :param max_results: 返回的最佳摘要条数
    :return: 网络检索得到的上下文摘要列表
    """
    if context_anchor:
        search_term = f"{context_anchor} {query}"
    else:
        search_term = query

    results = []
    
    try:
        # 使用 ddgs 库最新语法：直接将 search_term 作为第一个位置参数传入
        with DDGS(timeout=5) as ddgs:
            raw_results = list(ddgs.text(
                search_term, 
                region="cn-zh", 
                max_results=max_results
            ))
            
            if raw_results:
                for r in raw_results:
                    body = r.get("body", "").strip()
                    title = r.get("title", "").strip()
                    if body:
                        results.append(f"【网络网页】{title}：{body}")
            else:
                results.append(f"【网络网页】未找到关于“{search_term}”的相关网络语境。")
    except Exception as e:
        print(f"⚠️ 网络知识库搜索超时或异常: {e}")
        results.append(f"【网络网页】网络检索暂时不可用，搜索词：“{search_term}”。")
        
    return results

if __name__ == "__main__":
    test_res = search_network_context(query="鱼类用鳃呼吸", context_anchor="朋友圈养生")
    print("网络知识库测试输出：")
    for r in test_res:
        print(r)