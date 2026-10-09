import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 动态挂载项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline_lib import prompts, core, rewriter, llm_adapter

# ==========================================
# 路径与设备配置
# ==========================================
# PPO 微调保存的最佳 Decomposer 模型路径
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "decomposer_ppo_final", "best_checkpoint")
# 如果没有跑出 best_checkpoint，自动降级加载基座模型或普通 checkpoint
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "decomposer_ppo_final")
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.environ.get(
        "LOCAL_MODEL_DIR", 
        os.path.join(PROJECT_ROOT, "models", "DeepSeek-R1-Distill-Qwen-7B")
    )

TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_splits", "test_set_processed_stage1_2.jsonl")
OUTPUT_RESULT_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_splits", "test_set_results.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_s0_dict(sample: dict) -> dict:
    """提取标准化 s0_dict 输入结构"""
    if "stage1_tuple" in sample:
        stage1_data = sample["stage1_tuple"]
        y_orig = sample.get("full_sentence", stage1_data[0] if len(stage1_data) > 0 else "")
        raw_groups = stage1_data[1] if len(stage1_data) > 1 and isinstance(stage1_data[1], list) else []
        parsed_groups = []
        for idx, g in enumerate(raw_groups):
            parsed_groups.append({
                "group_id": f"group_{idx}",
                "context_shell": g.get("full_context_shell", ""),
                "pure_assertion": g.get("pure_assertion", "")
            })
        return {"Y_orig": y_orig, "groups": parsed_groups}
    elif "s0_dict" in sample:
        return sample["s0_dict"]
    else:
        return {
            "Y_orig": sample.get("Y_orig", sample.get("full_sentence", "")),
            "groups": sample.get("groups", [])
        }


def build_decomposer_prompt(s0_dict: dict) -> str:
    """组装 Decomposer 的输入 Prompt"""
    y_orig = s0_dict.get("Y_orig", "")
    groups = s0_dict.get("groups", [])
    
    groups_detail = ""
    for g in groups:
        gid = g.get('group_id', 'group_0')
        cshell = g.get('context_shell', '')
        passert = g.get('pure_assertion', '')
        
        groups_detail += f"- Group ID: {gid}\n"
        groups_detail += f"  * Context Shell (C): {cshell}\n"
        groups_detail += f"  * Pure Assertion (A): {passert}\n\n"

    user_text = prompts.DECOMPOSER_USER_PROMPT.format(Y_orig=y_orig, groups_detail=groups_detail)
    prompt_text = (
        f"<|im_start|>system\n{prompts.DECOMPOSER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt_text


def predict_decomposer_tree(model, tokenizer, prompt_text: str) -> list:
    """使用微调后的 Decomposer 生成 Action Tree JSON"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # 推理阶段使用确定性采样 (Greedy Search)
            pad_token_id=tokenizer.eos_token_id
        )

    action_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
    cleaned_json_str = llm_adapter.clean_deepseek_reasoning_output(action_text)

    try:
        action_tree = json.loads(cleaned_json_str)
        if isinstance(action_tree, list):
            return action_tree
        return []
    except Exception:
        return []


def process_test_set():
    print("==================================================")
    print(" 🚀 开始对测试集执行 REAP 端到端推理与评估 ")
    print("==================================================")
    print(f"👉 加载 Decomposer 权重 : {CHECKPOINT_PATH}")
    print(f"👉 读取测试集路径       : {TEST_DATA_PATH}")
    print(f"👉 结果保存路径         : {OUTPUT_RESULT_PATH}")
    print("--------------------------------------------------\n")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 错误: 未找到测试集文件: {TEST_DATA_PATH}")
        return

    # 1. 加载微调后的 Decomposer 模型与分词器
    print("1️⃣ [模型加载] 正在加载微调后的 Decomposer 神经网络与 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=target_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("   ✅ 模型加载成功！\n")

    # 2. 读取测试集
    test_samples = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))

    print(f"2️⃣ [测试集处理] 锁定 {len(test_samples)} 条测试样本，开始批量推演...")
    
    os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)
    rewriter_engine = rewriter.TreeGuidedRewriter()

    success_count = 0
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(test_samples):
            sample_id = sample.get("id", idx)
            s0_dict = extract_s0_dict(sample)
            y_orig = s0_dict.get("Y_orig", "")

            # Step A: Decomposer 生成拓扑逻辑树
            prompt_text = build_decomposer_prompt(s0_dict)
            decomposer_tree = predict_decomposer_tree(model, tokenizer, prompt_text)

            # Step B: 执行叶子节点检索与事实抽取
            node_results = {}
            known_facts = {}
            leaf_nodes = [
                n for n in decomposer_tree 
                if isinstance(n, dict) and n.get("task_type") in ["LEAF_CONTEXT_CHECK", "LEAF_REALITY_CHECK"]
            ]

            for leaf in leaf_nodes:
                node_id = leaf.get("id")
                search_query = leaf.get("query", y_orig)
                fact_result = core.retrieve_and_extract_leaf_facts(
                    search_query=search_query,
                    active_leaf_node=leaf,
                    known_facts=known_facts
                )
                node_results[node_id] = fact_result
                if fact_result.get("extracted_fact"):
                    known_facts[node_id] = fact_result.get("extracted_fact")

            # Step C: 三层布尔逻辑对撞判别
            final_tree_output = core.evaluate_tree_boolean_logic(decomposer_tree, node_results)
            boolean_result = bool(final_tree_output.get("sentence_boolean_truth", False))

            # Step D: 树引导精准改写 (Tree-Guided Rewriter)
            corrected_sentence = rewriter_engine.rewrite_sentence(
                full_sentence=y_orig,
                execution_tree=final_tree_output.get("execution_tree", [])
            )

            # Step E: 构造清爽格式（不含中间树结构）落盘
            result_entry = {
                "id": sample_id,
                "original_sentence": y_orig,
                "is_correct": boolean_result,          # 最终逻辑真值: true / false
                "corrected_sentence": corrected_sentence # 修改后的句子
            }

            out_f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            out_f.flush()

            success_count += 1
            print(f"   - [{success_count}/{len(test_samples)}] Sample ID {sample_id} | Logic Verdict: {boolean_result} | Done!")

    print("\n==================================================")
    print(f" 🎉 测试集推理完成！共处理 {success_count} 条数据。")
    print(f" 📁 干净格式的结果已成功落盘至:")
    print(f" 👉 {OUTPUT_RESULT_PATH}")
    print("==================================================")


if __name__ == "__main__":
    process_test_set()