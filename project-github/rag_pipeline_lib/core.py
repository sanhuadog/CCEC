# rag_pipeline_lib/core.py
import config
from rag_pipeline_lib import llm_adapter
from multi_source_kb.REAP_multi_source_retriever import search_knowledge_base
from typing import List, Dict, Any, Union
import re
import json
import copy

def retrieve_context(query: str, task_type: str = "LEAF_REALITY_CHECK") -> List[Dict[str, Any]]:
    """
    Retrieves context using REAP Multi-Source Knowledge Retriever (Wiki KB / Academic / Web).
    Routes retrieval based on task_type (LEAF_CONTEXT_CHECK or LEAF_REALITY_CHECK).
    """
    print(f"\n--- [Stage: Retrieval] Multi-source retrieving: '{query}' (Task Type: {task_type}) ---")
    try:
        top_k = getattr(config, "TOP_K", 3)
        search_hits = search_knowledge_base(
            query=query,
            task_type=task_type,
            top_k=top_k
        )
        print(f"--- Retrieval completed, hit {len(search_hits)} highly relevant evidence items ---")
        return search_hits
    except Exception as e:
        print(f"REAP multi-source retrieval error: {e}")
        return []

def decompose_reasoning_tree(s0_input: Union[Dict[str, Any], str], context_shell: str = "", pure_assertion: str = "") -> List[Dict[str, Any]]:
    """
    Tier 1: Reasoning Tree Decomposition.
    Calls LLM (RL-Decomposer) to construct a bottom-up 3-layer Boolean Reasoning Tree.
    """
    print(f"\n--- [Stage: Decompose] Constructing 3-Layer Boolean Reasoning Tree ---")
    
    if isinstance(s0_input, dict):
        print(f"Target Sentence Y_orig: {s0_input.get('Y_orig', '')}")
        print(f"Number of Groups: {len(s0_input.get('groups', []))}")
    else:
        print(f"Sentence: {s0_input}")
    try:
        tree_json_str = llm_adapter.decompose_tree(
            s0_input=s0_input,
            context_shell=context_shell,
            pure_assertion=pure_assertion
        )
        match = re.search(r'\[.*\]', tree_json_str, re.DOTALL)
        if not match:
            print("Error: No JSON array found in LLM decomposition output.")
            return []
        
        tree_nodes = json.loads(match.group(0))
        
        print(f"--- Tree constructed successfully, generated {len(tree_nodes)} nodes ---")
        for node in tree_nodes:
            print(f" - [{node.get('task_type')}] {node.get('id')}: {node.get('query')}")
        print("---------------------------------------------")
        
        return tree_nodes
    except Exception as e:
        print(f"Unknown error occurred during tree decomposition: {e}")
        return []

def retrieve_and_extract_leaf_facts(search_query: str, active_leaf_node: dict, known_facts: dict) -> dict:
    node_id = active_leaf_node.get('id', 'UNKNOWN')
    task_type = active_leaf_node.get('task_type', 'LEAF_REALITY_CHECK')
    
    print(f"\n--- [Stage: Fact & Vector Extraction] Node {node_id} ---")
    print(f"Search Probe: '{search_query}' | Task Type: {task_type}")
    
    search_hits = retrieve_context(query=search_query, task_type=task_type)
    
    retrieved_documents_str = "No relevant context found."
    if search_hits:
        docs_list = []
        for i, hit in enumerate(search_hits):
            content = hit.get('text', str(hit))
            source = hit.get('source', 'Unknown_Source')
            score = hit.get('score', 0.0)
            docs_list.append(
                f'<document id={i+1} source="{source}" score={score:.2f}>\n{content}\n</document>'
            )
        retrieved_documents_str = "\n\n".join(docs_list)
    
    active_requirement_str = json.dumps(active_leaf_node, indent=2, ensure_ascii=False)
    facts_list_str = json.dumps(known_facts, indent=2, ensure_ascii=False)
    try:
        facts_json_str = llm_adapter.extract_facts_and_vector(
            query=search_query,
            active_requirement=active_requirement_str,
            known_facts=facts_list_str,
            retrieved_documents=retrieved_documents_str
        )
        match = re.search(r'\{.*\}', facts_json_str, re.DOTALL)
        if not match:
            return {}
        
        extracted_data = json.loads(match.group(0))
        
        # 兼容性归一化向量命名
        if "efficacy_vector" not in extracted_data:
            if "validity_vector" in extracted_data:
                extracted_data["efficacy_vector"] = extracted_data["validity_vector"]
            else:
                extracted_data["efficacy_vector"] = {"v_context": 0, "v_reality": 0, "v_universal": 0}
                
        v_ctx = extracted_data["efficacy_vector"].get("v_context", 0)
        v_real = extracted_data["efficacy_vector"].get("v_reality", 0)
        v_univ = extracted_data["efficacy_vector"].get("v_universal", 0)
        
        # 论文对齐 1: 判定 PartialClue 状态下的 80% 一致性阈值
        consistency_ratio = extracted_data.get("consistency_ratio", 1.0)
        label_type = extracted_data.get("label_type", "DirectAnswer")
        if label_type == "PartialClue" and consistency_ratio < 0.8:
            print(f"⚠️ [PartialClue] Consistency ratio ({consistency_ratio:.2f}) < 0.8. Requires further retrieval.")
            extracted_data["requires_more_evidence"] = True
        else:
            extracted_data["requires_more_evidence"] = False

        print(f"--- Fact Extraction Success | v_context={v_ctx}, v_reality={v_real}, v_universal={v_univ} ---")
        return extracted_data
    except Exception as e:
        print(f"Error during fact extraction: {e}")
        return {}

def update_tree_execution_plan(original_query: str, completed_node_results: dict, pending_nodes: list) -> dict:
    print(f"\n--- [Stage: Plan Updater] Tree State Propagation & Scheduling ---")
    
    completed_str = json.dumps(completed_node_results, indent=2, ensure_ascii=False)
    pending_str = json.dumps(pending_nodes, indent=2, ensure_ascii=False)
    try:
        update_json_str = llm_adapter.update_plan(
            query=original_query,
            completed_node_results=completed_str,
            pending_requirements=pending_str
        )
        match = re.search(r'\{.*\}', update_json_str, re.DOTALL)
        if not match:
            return {}
        
        update_result = json.loads(match.group(0))
        return update_result
    except Exception as e:
        print(f"Error occurred during tree state scheduling: {e}")
        return {}

def evaluate_tree_boolean_logic(tree_nodes: list, node_results: dict) -> dict:
    """
    100% 对齐论文: 自底向上 (Bottom-up) 三维真值传递与逻辑聚合
    - v_context: 决定 Context Safety (Safe/Conflict), 指导路径剪枝
    - v_reality: 储备真实性属性 (用于根因分析)
    - v_universal: 通过逻辑与 (AND Aggregation) 自底向上递归传递至 Root
    """
    execution_tree = copy.deepcopy(tree_nodes)
    node_map = {n["id"]: n for n in execution_tree}
    
    # 1. 填入叶节点的三维真值 v_context, v_reality, v_universal
    for node_id, data in node_results.items():
        if node_id in node_map:
            vec = data.get("efficacy_vector", data.get("validity_vector", {}))
            v_ctx = vec.get("v_context", 0)
            v_real = vec.get("v_reality", 0)
            v_univ = vec.get("v_universal", 0)
            
            # 论文逻辑：节点上下文安全判定 (Safe/Conflict)
            node_status = "SAFE" if v_ctx == 1 else "CONFLICT"
            
            node_map[node_id]["result"] = {
                "status": "COMPLETED",
                "extracted_fact": data.get("extracted_fact", data.get("statement", "")),
                "efficacy_vector": {
                    "v_context": v_ctx,
                    "v_reality": v_real,
                    "v_universal": v_univ
                },
                "node_status": node_status,
                "boolean_value": bool(v_univ)
            }

    # 2. 按 Group 进行冲突检测与自底向上 Group 级逻辑与 (AND) 聚合
    groups = {}
    for n in execution_tree:
        gid = n.get("group_id")
        if gid and gid != "global":
            groups.setdefault(gid, []).append(n)
            
    global_has_clash = False
    
    for gid, g_nodes in groups.items():
        ctx_nodes = [n for n in g_nodes if n.get("task_type") == "LEAF_CONTEXT_CHECK"]
        real_nodes = [n for n in g_nodes if n.get("task_type") == "LEAF_REALITY_CHECK"]
        clash_nodes = [n for n in g_nodes if n.get("task_type") == "GROUP_CLASH_NODE"]
        
        len_ctx, len_real = len(ctx_nodes), len(real_nodes)
        m_j = min(len_ctx, len_real)
        n_isolated = abs(len_ctx - len_real)
        n_confused = 0
        
        v_ctx_val = ctx_nodes[0]["result"]["efficacy_vector"].get("v_context", 0) if (ctx_nodes and "result" in ctx_nodes[0]) else 0
        v_real_val = real_nodes[0]["result"]["efficacy_vector"].get("v_reality", 0) if (real_nodes and "result" in real_nodes[0]) else 0
        
        # 论文定义: 没有语境保护且存在真实性违背，则触发 Clash
        has_ctx_protection = (v_ctx_val == 1)
        has_real_violation = (v_real_val == 0)
        has_clash = (not has_ctx_protection) and has_real_violation
        
        if has_clash:
            global_has_clash = True
            
        for c_node in clash_nodes:
            c_node["result"] = {
                "status": "EVALUATED",
                "has_clash": has_clash,
                "clash_type": "REALITY_CONFLICT" if has_clash else "CONSISTENT",
                "explanation": f"v_context={v_ctx_val}, v_reality={v_real_val} -> {'Clash Triggered' if has_clash else 'Consistent'}",
                "rl_stats": {
                    "M_j": m_j,
                    "N_isolated_j": n_isolated,
                    "N_confused_j": n_confused
                }
            }

    # 3. 论文核心: 自底向上逻辑与 (AND Aggregation) 传播至 ROOT 节点
    # 只有当所有叶节点的 v_universal 均为 1 且全局未检测出事实冲突时，全局句式才是 TRUE
    all_leaf_universal_valid = all(
        n.get("result", {}).get("efficacy_vector", {}).get("v_universal", 0) == 1
        for n in execution_tree
        if n.get("task_type") in ["LEAF_CONTEXT_CHECK", "LEAF_REALITY_CHECK"]
    ) if execution_tree else False

    sentence_boolean_truth = all_leaf_universal_valid and (not global_has_clash)

    # 4. 回填 ROOT 节点真值
    for n in execution_tree:
        if n.get("task_type") in ["ROOT_TEXT_FUSION", "ROOT_SENTENCE_FUSION"]:
            n["result"] = {
                "status": "EVALUATED",
                "sentence_boolean_truth": sentence_boolean_truth,
                "v_universal_root": 1 if sentence_boolean_truth else 0,
                "summary": "Logically consistent and universally valid." if sentence_boolean_truth else "Commonsense conflict or universal validity violation detected."
            }

    return {
        "sentence_boolean_truth": sentence_boolean_truth,
        "global_clash_detected": global_has_clash,
        "execution_tree": execution_tree
    }