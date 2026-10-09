# rag_pipeline_lib/rewriter.py
import json
import re
from typing import Dict, Any, List
from rag_pipeline_lib import llm_adapter

# ---------------------------------------------------------------------
# Prompts aligned with paper's Tree-Guided Rewriter specification
# ---------------------------------------------------------------------
SYSTEM_PROMPT_REWRITER = """You are an expert precision text corrector. Your task is to perform targeted, minimal-edit error correction on an input text Y_orig, guided by conflict paths and evidence T_clue from the Factual Conflict Tree.

### Correction Rules:
1. Soft Semantic Masking:
   - Freeze all error-free clauses and context frames in Y_orig (Frozen Safe Regions).
   - Confine edits strictly to designated conflict spans (Editable Slots, forming Y_masked).
2. Categorized Root-Cause Correction:
   - Factual Bias: If the assertion is factually wrong in objective reality, revise only the entity/predicate to match objective facts.
   - Contextual Conflict: If the assertion is valid but conflicts with context anchors, adapt only the contextual constraints while keeping the factual assertion intact.
3. Dual-Objective Optimization:
   - Maximize Factual Alignment S_fact: Align strictly with T_clue.
   - Minimize Edit Distance D_edit: Strictly preserve the original text. Only replace the exact erroneous word or phrase without rephrasing any other part.

### Output Constraints:
- Output strictly in valid JSON format.
- Do NOT include markdown code blocks, conversational filler, or explanations outside the JSON object."""


class TreeGuidedRewriter:
    """
    Sentence rewriter based on reasoning tree path pruning and soft semantic masking.
    """
    def locate_root_cause_and_clues(self, execution_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        clash_groups = []
        
        # Get all group nodes and leaf nodes
        group_nodes = [n for n in execution_tree if n.get("task_type") == "GROUP_CLASH_NODE"]
        
        for g_node in group_nodes:
            group_id = g_node.get("group_id")
            
            # 1. Compatible with multiple has_clash evaluation formats (bool, str, or explicit check)
            node_has_clash = g_node.get("has_clash", False)
            if isinstance(node_has_clash, str):
                node_has_clash = node_has_clash.lower() == "true"

            # Find corresponding leaf nodes for the current group
            leaf_nodes = [
                n for n in execution_tree 
                if n.get("parent_group_id") == group_id or (group_id and str(n.get("id", "")).startswith(str(group_id)))
            ]
            if not leaf_nodes:
                # Fallback: get all leaf nodes
                leaf_nodes = [n for n in execution_tree if n.get("task_type") in ["LEAF_CONTEXT_CHECK", "LEAF_REALITY_CHECK"]]

            failing_aspects = []
            group_clues = []
            for leaf in leaf_nodes:
                # Compatible with truth values extracted from node_result or efficacy_vector
                eff_vec = leaf.get("efficacy_vector") or leaf.get("vector") or {}
                v_ctx = eff_vec.get("v_context", 1)
                v_real = eff_vec.get("v_reality", 1)
                
                if v_real == 0:
                    failing_aspects.append("Factual Bias (v_reality = 0)")
                if v_ctx == 0:
                    failing_aspects.append("Contextual Conflict (v_context = 0)")

                # Collect clues
                clues = leaf.get("clue", [])
                if isinstance(clues, list):
                    group_clues.extend(clues)
                elif isinstance(clues, str) and clues:
                    group_clues.append(clues)

            # Fallback mark: if conflict is detected or any vector element is 0, mark as requiring edit
            if node_has_clash or len(failing_aspects) > 0:
                clash_groups.append({
                    "group_id": group_id,
                    "context_shell": g_node.get("context_shell", ""),
                    "pure_assertion": g_node.get("pure_assertion", ""),
                    "failing_aspects": list(set(failing_aspects)) if failing_aspects else ["Factual Bias (v_reality = 0)"],
                    "clues": list(set(group_clues))
                })

        return {
            "has_conflict": len(clash_groups) > 0,
            "clash_groups": clash_groups
        }

    def rewrite_sentence(self, full_sentence: str, execution_tree: List[Dict[str, Any]]) -> str:
        # 1. Locate conflicts from the execution tree
        localization = self.locate_root_cause_and_clues(execution_tree)
        
        # 2. Fallback mechanism: if no clash_groups found in tree, construct a fallback conflict object
        clash_info = localization["clash_groups"]
        if not clash_info:
            clash_info = [{
                "context_shell": "",
                "pure_assertion": full_sentence,
                "failing_aspects": ["Factual Bias (v_reality = 0)"],
                "clues": []
            }]

        # 3. Assemble formatted clash info block (T_clue) aligned with paper user prompt
        clash_info_block = ""
        for idx, item in enumerate(clash_info):
            clash_info_block += f"Conflict Path {idx+1}:\n"
            if item['context_shell']:
                clash_info_block += f"- Context Shell (C): \"{item['context_shell']}\"\n"
            if item['pure_assertion']:
                clash_info_block += f"- Target Slot: Assertion = \"{item['pure_assertion']}\"\n"
            clash_info_block += f"- Conflict Category: {', '.join(item['failing_aspects'])}\n"
            if item['clues']:
                clash_info_block += f"- Clues (T_clue): \"{'; '.join(item['clues'])}\"\n"
            clash_info_block += "\n"

        # 4. Construct User Prompt with explicit JSON format instruction
        user_prompt = f"""[INPUT TEXT]
Y_orig: "{full_sentence}"

[PATH-PRUNED CONFLICT DIAGNOSTICS & RETRIEVED CLUES (T_clue)]
{clash_info_block.strip()}

[INSTRUCTION]
Apply categorized root-cause correction on the editable slots of Y_masked. Maximize factual alignment with T_clue. Strictly preserve the original text. Only replace the exact erroneous word or phrase without rephrasing any other part.

Return strictly in the following JSON format:
{{
  "corrected_sentence": "Your finalized corrected text Y_hat here"
}}"""

        # 5. Call LLM for rewrite and safely parse JSON response
        try:
            raw_response = llm_adapter._call_ollama_chat(
                system_prompt=SYSTEM_PROMPT_REWRITER,
                user_prompt=user_prompt,
                temperature=0.0
            ).strip()

            corrected = ""
            # Attempt to parse as JSON first
            try:
                data = json.loads(raw_response)
                if isinstance(data, dict):
                    corrected = data.get("corrected_sentence", "").strip()
            except Exception:
                # Fallback: regex search for JSON object if json.loads fails
                match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        corrected = data.get("corrected_sentence", "").strip()
                    except Exception:
                        pass

            # Final cleanup on string wrappers/quotes
            if corrected:
                corrected = re.sub(r'^["‘“]|["’”]$', '', corrected).strip()

            # Return original sentence if extraction yielded empty or invalid JSON
            if not corrected or corrected in ["{}", "{ }"]:
                return full_sentence

            return corrected

        except Exception as e:
            print(f"[Rewriter Error] Failed to call rewrite model: {e}")
            return full_sentence