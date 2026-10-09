# rag_pipeline_lib/trajectory_collector.py
import json
from typing import Dict, Any, List
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib.reward_engine import REAPRewardEngine


class TrajectoryCollector:
    def __init__(self, reward_engine: REAPRewardEngine):
        """
        REAP Trajectory Collector aligned with step-wise reward formulation in paper.
        :param reward_engine: Instantiated REAPRewardEngine
        """
        self.reward_engine = reward_engine

    def collect_single_trajectory(
        self, 
        s0_dict: Dict[str, Any], 
        golden_rewrite: str,
        pipeline_runner_func
    ) -> Dict[str, Any]:
        """
        Executes a single complete rollout and collects trajectory data.
        """
        y_orig = s0_dict.get("Y_orig", "")
        state = {
            "Y_orig": y_orig,
            "groups": s0_dict.get("groups", [])
        }

        # 1. Action (a_t): Decomposer model constructs tree
        decomposer_raw_output = llm_adapter.decompose_tree(s0_dict)
        cleaned_action = llm_adapter.clean_deepseek_reasoning_output(decomposer_raw_output)
        
        try:
            decomposer_tree = json.loads(cleaned_action)
            if not isinstance(decomposer_tree, list):
                decomposer_tree = []
        except Exception:
            decomposer_tree = []

        # 2. Environment Step: Execute full RAG pipeline
        pipeline_result = pipeline_runner_func(s0_dict, decomposer_tree)
        predicted_rewrite = pipeline_result.get("predicted_rewrite", pipeline_result.get("corrected_sentence", ""))
        execution_tree = pipeline_result.get("execution_tree", decomposer_tree)
        z_logic = 1 if pipeline_result.get("sentence_boolean_truth", False) else 0

        trajectory_data = {
            "Y_orig": y_orig,
            "state": state,
            "action_raw": decomposer_raw_output,
            "action_tree": decomposer_tree,
            "execution_tree": execution_tree,
            "predicted_rewrite": predicted_rewrite
        }

        # 3. Reward Calculation using REAPRewardEngine
        reward_info = self.reward_engine.calculate_reward(
            trajectory_data=trajectory_data,
            golden_rewrite=golden_rewrite,
            predicted_rewrite=predicted_rewrite,
            z_logic=z_logic
        )
        
        trajectory_data["reward_info"] = reward_info
        return trajectory_data

    def collect_batch(
        self, 
        batch_samples: List[Dict[str, Any]], 
        pipeline_runner_func
    ) -> List[Dict[str, Any]]:
        trajectories = []
        for sample in batch_samples:
            s0_dict = sample.get("s0_dict", sample)
            golden_rewrite = sample.get("golden_rewrite", sample.get("corrected_full_sentence", ""))
            
            traj = self.collect_single_trajectory(
                s0_dict=s0_dict, 
                golden_rewrite=golden_rewrite, 
                pipeline_runner_func=pipeline_runner_func
            )
            trajectories.append(traj)
            
        return trajectories