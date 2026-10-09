# rag_pipeline_lib/batch_processor.py

import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from rag_pipeline_lib import core
from rag_pipeline_lib.rewriter import TreeGuidedRewriter


class BatchTreePipeline:
    """
    REAP 三层布尔常识对撞树 - 批量处理与受控改写大模块
    支持对 JSONL 数据集进行单条或批量推理，输出带线索(clue)的搜索树、最终布尔真值以及修正后的句子。
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # 初始化基于对撞树引导的句子改写器
        self.rewriter = TreeGuidedRewriter()

    @staticmethod
    def build_s0_dict(sample: dict) -> dict:
        """从阶段1/2的样本解析 Stage 0 初始输入结构"""
        full_sentence = sample.get("full_sentence", "")
        stage1_tuple = sample.get("stage1_tuple", [])
        conflict_eval = sample.get("conflict_evaluation", {})

        groups = []
        if len(stage1_tuple) >= 2 and isinstance(stage1_tuple[1], list):
            for idx, item in enumerate(stage1_tuple[1]):
                group_id = f"group_{idx}"
                initial_check = item.get("initial_check") or {}
                
                intuition_I0 = {
                    "coarse_label": conflict_eval.get("coarse_label", ""),
                    "focus_guidance": conflict_eval.get("focus_guidance", ""),
                    "cs_evidence": str(initial_check.get("cs_evidence", ""))[:200],
                    "ctx_evidence": str(initial_check.get("ctx_evidence", ""))[:200]
                }

                groups.append({
                    "group_id": group_id,
                    "context_shell": item.get("full_context_shell", ""),
                    "context_nature": item.get("context_nature", ""),
                    "pure_assertion": item.get("pure_assertion", ""),
                    "intuition_I0": intuition_I0
                })

        if not groups:
            groups.append({
                "group_id": "group_0",
                "context_shell": sample.get("context_shell", ""),
                "context_nature": "默认语境",
                "pure_assertion": sample.get("pure_assertion", ""),
                "intuition_I0": {}
            })

        return {
            "Y_orig": full_sentence,
            "groups": groups
        }

    def process_sample(self, sample: dict) -> Dict[str, Any]:
        """
        处理单条样本的核心流水线（包含树解构、对撞计算、病根剪枝定位与受控改写）
        Returns:
            Dict 包含：id, full_sentence, gold_label, sentence_boolean_truth, 
                 global_clash_detected, corrected_sentence, execution_tree
        """
        s0_dict = self.build_s0_dict(sample)
        full_sentence = s0_dict["Y_orig"]
        sample_id = sample.get("id", "unknown_id")
        golden_label = sample.get("gold_label", sample.get("global_judge", "UNKNOWN"))

        # 1. Stage 1: RL-Decomposer 树形展开
        tree_nodes = core.decompose_reasoning_tree(s0_input=s0_dict)
        if not tree_nodes:
            return {
                "id": sample_id,
                "full_sentence": full_sentence,
                "gold_label": golden_label,
                "sentence_boolean_truth": "UNKNOWN",
                "global_clash_detected": False,
                "corrected_sentence": full_sentence,
                "execution_tree": [],
                "error": "Decomposition Failed"
            }

        # 2. Stage 2 & 3: 叶子节点检索、向量抽取与线索(clue)注入
        node_results = {}
        known_facts = {}
        leaf_nodes = [n for n in tree_nodes if n.get("task_type") in ["LEAF_CONTEXT_CHECK", "LEAF_REALITY_CHECK"]]

        for leaf in leaf_nodes:
            node_id = leaf.get("id")
            search_query = leaf.get("query", full_sentence)

            # 调用多源检索与事实抽取
            fact_result = core.retrieve_and_extract_leaf_facts(
                search_query=search_query,
                active_leaf_node=leaf,
                known_facts=known_facts
            )

            # 提取 Top-2 证据注入到节点下作为 clue 字段（供后续句子修改参考）
            clue_list = fact_result.get("clue", [])
            if not clue_list and fact_result.get("extracted_fact"):
                clue_list = [fact_result["extracted_fact"]]
            
            # 显式挂载到叶子节点字典中
            leaf["clue"] = clue_list[:2]
            
            node_results[node_id] = fact_result
            if fact_result.get("extracted_fact"):
                known_facts[node_id] = fact_result.get("extracted_fact")

        # 3. Stage 4: 自底向上汇聚与布尔代数对撞判定
        final_tree_output = core.evaluate_tree_boolean_logic(tree_nodes, node_results)
        execution_tree = final_tree_output.get("execution_tree", [])

        # 4. Stage 5: 逆向路径剪枝定位病根与受控语义改写
        corrected_sentence = full_sentence
        try:
            corrected_sentence = self.rewriter.rewrite_sentence(
                full_sentence=full_sentence,
                execution_tree=execution_tree
            )
        except Exception as e:
            if self.verbose:
                print(f"[Warning] 改写模块发生异常，退回原句: {e}")

        return {
            "id": sample_id,
            "full_sentence": full_sentence,
            "gold_label": golden_label,
            "raw_gold_label": sample.get("raw_gold_label", ""),
            "sentence_boolean_truth": final_tree_output.get("sentence_boolean_truth", "UNKNOWN"),
            "global_clash_detected": final_tree_output.get("global_clash_detected", False),
            "corrected_sentence": corrected_sentence,  # 受控修改后的新句子
            "execution_tree": execution_tree,
            "error": None
        }

    def run_file(self, input_file_path: str, output_file_path: Optional[str] = None, delay_sec: float = 0.2) -> List[Dict[str, Any]]:
        """
        批量处理 JSONL 文件
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"找不到输入文件: {input_file_path}")

        samples = []
        with open(input_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

        print(f"\n[BatchTreePipeline] 开始批量处理，共计 {len(samples)} 条样本...")
        
        all_results = []
        out_f = open(output_file_path, "w", encoding="utf-8") if output_file_path else None

        try:
            for sample in tqdm(samples, desc="Executing REAP Pipeline"):
                try:
                    res = self.process_sample(sample)
                    all_results.append(res)

                    if out_f:
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        out_f.flush()

                    if delay_sec > 0:
                        time.sleep(delay_sec)

                except Exception as e:
                    print(f"\n[Error] 处理样本 {sample.get('id')} 时异常: {str(e)}")
                    err_res = {
                        "id": sample.get("id"),
                        "full_sentence": sample.get("full_sentence"),
                        "error": str(e)
                    }
                    all_results.append(err_res)
                    if out_f:
                        out_f.write(json.dumps(err_res, ensure_ascii=False) + "\n")

        finally:
            if out_f:
                out_f.close()

        print(f"[BatchTreePipeline] 处理完成！成功生成 {len(all_results)} 条结构化推理树与修正句子。")
        return all_results