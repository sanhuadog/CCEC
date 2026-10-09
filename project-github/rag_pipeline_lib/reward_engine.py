# rag_pipeline_lib/reward_engine.py
import json
from typing import Dict, Any, List
from evaluate import load
from rag_pipeline_lib import llm_adapter


class REAPRewardEngine:
    def __init__(
        self, 
        base_r: float = 0.5, 
        alpha: float = 0.5,  # 论文对齐：alpha 设为 0.5
        theta1: float = 1.5,
        theta2: float = 0.5,
        theta3: float = 1.0
    ):
        """
        REAP Reward Engine strictly aligned with paper formulations.
        
        :param base_r: Base reward r in R_succ
        :param alpha: Weight scaling factor alpha for R_conf in R_total (alpha = 0.5)
        :param theta1: Weight for matched pairs M_j (theta1 = 1.5)
        :param theta2: Weight for isolated facts N_iso,j (theta2 = 0.5)
        :param theta3: Weight for confused facts N_cnf,j (theta3 = 1.0)
        """
        self.base_r = base_r
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        
        # Load BERTScore for English evaluation
        print("Loading BERTScore evaluation model for English...")
        self.bert_scorer = load("bertscore")
        print("BERTScore loaded successfully!")

    def compute_bert_score(self, reference: str, candidate: str) -> float:
        """Computes continuous semantic similarity S_bert(Y_hat, Y*) in [-1, 1]."""
        if not candidate.strip() or not reference.strip():
            return 0.0
        results = self.bert_scorer.compute(
            predictions=[candidate], 
            references=[reference], 
            lang="en",
            model_type="bert-base-uncased"
        )
        return float(results["f1"][0])

    def calculate_reward(
        self, 
        trajectory_data: Dict[str, Any], 
        golden_rewrite: str, 
        predicted_rewrite: str,
        z_logic: int = 1
    ) -> Dict[str, float]:
        """
        Calculates complete total reward R_total based on paper equations:
        R_succ = z_logic * (S_bert(Y_hat, Y*) + r)
        R_conf = (1/J) * sum_j(R_conf,j)
        R_total = R_succ + alpha * R_conf (alpha = 0.5)
        """
        # 1. Compute R_succ with soft logic gate
        s_bert = self.compute_bert_score(golden_rewrite, predicted_rewrite)
        
        # Binary logic correctness indicator z_logic in {0, 1}
        z_logic_val = 1 if bool(z_logic) else 0
        r_succ = z_logic_val * (s_bert + self.base_r)

        # 2. Compute R_conf per assertion group j
        execution_tree = trajectory_data.get("execution_tree", [])
        
        # Group leaf nodes and clash nodes by group_id
        groups_map = {}
        for node in execution_tree:
            gid = node.get("group_id")
            if gid and gid != "global":
                groups_map.setdefault(gid, []).append(node)

        r_conf_list = []
        
        if not groups_map:
            # Fallback if no explicit groups found
            groups_map = {"group_0": execution_tree}

        for gid, g_nodes in groups_map.items():
            # Check context indicator I_ctx(j): 1 if context shell exists, else 0
            has_context = any(bool(n.get("context_shell", "")) for n in g_nodes)
            i_ctx = 1.0 if has_context else 0.0
            
            # Extract statistics for group j
            ctx_nodes = [n for n in g_nodes if n.get("task_type") == "LEAF_CONTEXT_CHECK"]
            real_nodes = [n for n in g_nodes if n.get("task_type") == "LEAF_REALITY_CHECK"]

            # M_j: matched valid pairs count
            m_j = min(len(ctx_nodes), len(real_nodes))
            
            # N_iso,j: isolated facts count
            n_iso_j = 0
            for leaf in ctx_nodes + real_nodes:
                result = leaf.get("result", {})
                if not result.get("extracted_fact") and not leaf.get("clue"):
                    n_iso_j += 1
            
            # N_cnf,j: confused facts count (v_reality = 1 and v_context = 1)
            n_cnf_j = 0
            for leaf in ctx_nodes + real_nodes:
                vec = leaf.get("result", {}).get("efficacy_vector", leaf.get("efficacy_vector", {}))
                if vec.get("v_reality") == 1 and vec.get("v_context") == 1:
                    n_cnf_j += 1

            # Non-contextual operator Psi(F_j)
            has_context_pollution = any(
                leaf.get("result", {}).get("efficacy_vector", {}).get("v_context") == 1
                for leaf in g_nodes if "LEAF_" in leaf.get("task_type", "")
            )
            psi_fj = -1.0 if has_context_pollution else 1.0

            # Compute R_conf,j
            r_conf_j = i_ctx * (
                self.theta1 * m_j - self.theta2 * n_iso_j - self.theta3 * n_cnf_j
            ) + (1.0 - i_ctx) * psi_fj
            
            r_conf_list.append(r_conf_j)

        # Total R_conf = average across all groups
        j_count = max(len(r_conf_list), 1)
        r_conf = sum(r_conf_list) / j_count

        # 3. Total Reward R_total = R_succ + 0.5 * R_conf
        r_total = r_succ + self.alpha * r_conf

        return {
            "r_total": round(r_total, 4),
            "r_succ": round(r_succ, 4),
            "r_conf": round(r_conf, 4),
            "s_bert": round(s_bert, 4),
            "z_logic": z_logic_val
        }


# Global singleton reward engine
_global_reward_engine = None

def compute_decomposer_reward(
    action_text: str, 
    s0_dict: dict, 
    golden_rewrite: str,
    execution_tree: List[Dict[str, Any]] = None,
    z_logic: int = 1
) -> float:
    """
    Adapter function for Decomposer PPO reward computation.
    """
    global _global_reward_engine
    if _global_reward_engine is None:
        _global_reward_engine = REAPRewardEngine(alpha=0.5)

    # 1. Format and validity check on generated Action JSON
    try:
        cleaned_action = llm_adapter.clean_deepseek_reasoning_output(action_text)
        action_tree = json.loads(cleaned_action)
        if not isinstance(action_tree, list):
            return -1.0
    except Exception:
        return -1.0

    # 2. Topology check (presence of Root and Leaf nodes)
    has_root = any(node.get("task_type") in ["ROOT_TEXT_FUSION", "ROOT_SENTENCE_FUSION"] for node in action_tree)
    has_leaf = any("LEAF_" in node.get("task_type", "") for node in action_tree)
    if not (has_root and has_leaf):
        return -0.5

    # 3. Construct trajectory data and compute reward
    trajectory_data = {
        "Y_orig": s0_dict.get("Y_orig", ""),
        "execution_tree": execution_tree if execution_tree else action_tree
    }

    res = _global_reward_engine.calculate_reward(
        trajectory_data=trajectory_data,
        golden_rewrite=golden_rewrite,
        predicted_rewrite=action_text,
        z_logic=z_logic
    )

    return res.get("r_total", 0.0)