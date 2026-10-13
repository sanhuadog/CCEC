# rag_pipeline_lib/pipeline.py
import json
import copy
from typing import Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
from functools import partial

from rag_pipeline_lib import core as rag_core
from rag_pipeline_lib import rewriter

tracer_context: ContextVar = ContextVar('tracer_context', default=None)


class Tracer:
    def __init__(self):
        self.log = []
        self.iteration_count = 0
        self._pending_log = []

    def record_llm_call(self, adapter_function_name, inputs, output):
        trace_entry = {
            "adapter_function_name": adapter_function_name,
            "llm_inputs": copy.deepcopy(inputs),
            "llm_output": copy.deepcopy(output)
        }
        self._pending_log.append(trace_entry)

    def commit_pending(self):
        self.log.extend(self._pending_log)
        self._pending_log.clear()

    def discard_pending(self):
        self._pending_log.clear()


def run_tree_pipeline(
    s0_input: Union[Dict[str, Any], str],
    context_shell: str = "",
    pure_assertion: str = "",
    verbose: bool = True,
    trace_collector: Tracer = None,
    enable_rewrite: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete pipeline for the REAP 3-Layer Boolean Reasoning Tree.
    Flow: Decompose -> Multi-source Retrieve & Extract -> Bottom-up Truth Evaluation -> Tree-Guided Rewriting
    """
    current_tracer = tracer_context.get()
    if isinstance(s0_input, dict):
        full_sentence = s0_input.get("Y_orig", "")
    else:
        full_sentence = s0_input

    if verbose:
        print(f"\n🚀 Launching REAP Boolean Reasoning Tree Pipeline")
        print(f"Target Sentence: '{full_sentence}'")

    # ==========================================
    # Stage 1: Build Topology Tree (Decompose Tree)
    # ==========================================
    tree_nodes = []
    max_analysis_retries = 3
    for attempt in range(max_analysis_retries):
        try:
            if verbose and attempt > 0:
                print(f"🔄 Retrying tree decomposition... ({attempt + 1}/{max_analysis_retries})")
            
            if current_tracer: 
                current_tracer.discard_pending()
            
            tree_nodes = rag_core.decompose_reasoning_tree(
                s0_input=s0_input,
                context_shell=context_shell,
                pure_assertion=pure_assertion
            )
            
            if not tree_nodes:
                raise ValueError("Failed to retrieve valid tree nodes from LLM decomposition.")
            
            if current_tracer: 
                current_tracer.commit_pending()
            if verbose: 
                print("✅ Boolean reasoning tree constructed successfully.")
            break 
        except Exception as e:
            if verbose: 
                print(f"❌ Tree construction failed (Attempt {attempt + 1}): {e}")
            if attempt == max_analysis_retries - 1:
                if current_tracer: 
                    current_tracer.discard_pending()
                return {"error": f"Pipeline Error: Failed to build reasoning tree after {max_analysis_retries} attempts."}

    if not tree_nodes:
        return {"error": "Pipeline Error: Unable to obtain valid topological tree."}

    pending_nodes = list(tree_nodes)
    completed_node_results = {}
    max_iterations = 6
    iteration_count = 0

    # ==========================================
    # Stage 2 & 3: Dynamic Execution & Scheduling
    # ==========================================
    while iteration_count < max_iterations:
        if not pending_nodes:
            if verbose: 
                print("\n✅ All nodes converged. Entering algebraic evaluation stage.")
            break
        
        if current_tracer:
            current_tracer.iteration_count = iteration_count + 1
            current_tracer.discard_pending()

        if verbose:
            print(f"\n--- Iteration {iteration_count + 1}/{max_iterations} ---")
            print(f"📝 Pending Nodes: {len(pending_nodes)}")

        try:
            decision_result = rag_core.update_tree_execution_plan(
                original_query=full_sentence,
                completed_node_results=completed_node_results,
                pending_nodes=pending_nodes
            )
            decision = decision_result.get("decision", {})
            next_step = decision.get("next_step", decision.get("action"))

            if next_step in ["EVALUATE_TREE", "EVALUATE_TREE_ROOT"]:
                if verbose: 
                    print("✅ Scheduler determined sufficient vectors exist. Proceeding directly to boolean evaluation.")
                if current_tracer: 
                    current_tracer.commit_pending()
                break
            
            if next_step != "CONTINUE_TREE_SEARCH":
                # 兜底：如果调度指令不识别，默认打断进入评估
                if verbose:
                    print(f"⚠️ Unknown scheduling instruction '{next_step}', falling back to evaluation.")
                break

            executable_ids = decision.get("executable_node_ids", [])
            if not executable_ids:
                if verbose: 
                    print("⚠️ Scheduler failed to allocate executable nodes. Breaking out to prevent deadlock.")
                break

            executable_nodes = [n for n in pending_nodes if n["id"] in executable_ids]
            if verbose: 
                print(f"\n🔎 Executing retrieval and vector extraction in parallel for {len(executable_nodes)} nodes...")

            iteration_success_ids = []
            with ThreadPoolExecutor() as executor:
                future_to_node = {
                    executor.submit(
                        partial(copy_context().run, rag_core.retrieve_and_extract_leaf_facts), 
                        search_query=node.get("query", full_sentence), 
                        active_leaf_node=node, 
                        known_facts=completed_node_results
                    ): node
                    for node in executable_nodes
                }

                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    node_id = node["id"]
                    try:
                        extracted_data = future.result()
                        eff_vec = extracted_data.get("efficacy_vector", extracted_data.get("validity_vector"))
                        
                        if not extracted_data or not eff_vec:
                            raise ValueError(f"Node {node_id} failed to return a valid efficacy_vector")
                        
                        completed_node_results[node_id] = extracted_data
                        
                        # 如果不需要补采证据，则标记该节点在当前轮次处理完成
                        if not extracted_data.get("requires_more_evidence", False):
                            iteration_success_ids.append(node_id)
                        
                        if verbose:
                            v_ctx = eff_vec.get("v_context", 0)
                            v_real = eff_vec.get("v_reality", 0)
                            v_univ = eff_vec.get("v_universal", 0)
                            print(f"  - 📝 Node {node_id} complete | v_context={v_ctx}, v_reality={v_real}, v_universal={v_univ}")

                    except Exception as exc:
                        if verbose: 
                            print(f"  - ❌ Node {node_id} extraction failed: {exc}")

            pending_nodes = [n for n in pending_nodes if n["id"] not in iteration_success_ids]
            if current_tracer: 
                current_tracer.commit_pending()
            iteration_count += 1

        except Exception as e:
            if verbose: 
                print(f"\n❌ Exception in Iteration {iteration_count + 1}: {e}")
                print("🔄 Discarding current iteration state and preparing for next attempt.")
            if current_tracer: 
                current_tracer.discard_pending()
            iteration_count += 1
            continue

    if verbose and pending_nodes: 
        print("\n⚠️ Warning: Reached max iterations with pending nodes remaining. Forcing evaluation with existing evidence.")

    # ==========================================
    # Stage 4: Logical Algebraic Evaluation
    # ==========================================
    if verbose:
        print("\n--- Final Stage: Bottom-Up Algebraic Truth Aggregation ---")
    if current_tracer: 
        current_tracer.discard_pending()

    final_truth_result = rag_core.evaluate_tree_boolean_logic(
        tree_nodes=tree_nodes, 
        node_results=completed_node_results
    )

    # ==========================================
    # Stage 5: Tree-Guided Rewriting (修正改写)
    # ==========================================
    corrected_sentence = full_sentence
    if enable_rewrite:
        if verbose:
            print("\n--- Final Stage: Tree-Guided Minimal-Edit Rewriting ---")
        rw_engine = rewriter.TreeGuidedRewriter()
        corrected_sentence = rw_engine.rewrite_sentence(
            full_sentence=full_sentence,
            execution_tree=final_truth_result.get("execution_tree", [])
        )
        final_truth_result["predicted_rewrite"] = corrected_sentence
        final_truth_result["corrected_sentence"] = corrected_sentence

    if current_tracer: 
        current_tracer.commit_pending()

    return final_truth_result