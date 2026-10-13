# train_ppo_decomposer.py
import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# 1. 动态挂载项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline_lib import prompts, core, rewriter, llm_adapter
from rag_pipeline_lib.reward_engine import REAPRewardEngine

# ==========================================
# 全局配置 (严格完全对齐论文)
# ==========================================
MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_DIR", 
    os.path.join(PROJECT_ROOT, "models", "DeepSeek-R1-Distill-Qwen-7B")
)
SAVED_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "decomposer_ppo_final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_splits", "warmup_data.jsonl")
PPO_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_splits", "train_set_processed_stage1_2.jsonl")
PPO_DEV_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_splits", "dev_set_processed_stage1_2.jsonl")

WARMUP_EPOCHS = 3
SFT_LR = 2e-4
POLICY_LR = 2e-5       # \beta_policy = 2e-5
VALUE_LR = 1e-4        # \beta_value = 1e-4
GAMMA = 0.98            # \gamma = 0.98
LAMBDA_GAE = 0.95       # GAE \lambda = 0.95
ENTROPY_COEF = 0.05    # \mu = 0.05
CLIP_EPS = 0.2         # \epsilon = 0.2
VALUE_LOSS_WEIGHT = 0.5# \lambda_v = 0.5
MAX_PPO_STEPS = 5000
PPO_EPOCHS_PER_STEP = 3# 采样一次后内层 PPO 更新 Epoch 次数
BATCH_SIZE = 1
EVAL_INTERVAL = 200    # 每 200 step 评估一次
PATIENCE = 3           # 连续 3 次验证集未提升触发早停


# ==========================================
# 1. Critic 价值网络类
# ==========================================
class ValueHead(nn.Module):
    """
    Critic Network V_\phi(S_t) for estimating expected state-values.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        values = self.value_head(last_hidden_state).squeeze(-1)
        return values


# ==========================================
# 2. Dataset 与通用 Prompt 构建
# ==========================================
class GenericJsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            print(f"⚠️ [Warning] File not found: {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_s0_and_rewrite(sample):
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
            
        s0_dict = {"Y_orig": y_orig, "groups": parsed_groups}
        golden_rewrite = sample.get("corrected_full_sentence", "")
    elif "s0_dict" in sample:
        s0_dict = sample["s0_dict"]
        golden_rewrite = sample.get("golden_rewrite", "")
    else:
        s0_dict = {
            "Y_orig": sample.get("Y_orig", sample.get("full_sentence", "")),
            "groups": sample.get("groups", [])
        }
        golden_rewrite = sample.get("golden_rewrite", sample.get("corrected_full_sentence", ""))

    if isinstance(s0_dict["Y_orig"], list) and len(s0_dict["Y_orig"]) > 0:
        s0_dict["Y_orig"] = s0_dict["Y_orig"][0]
    if isinstance(golden_rewrite, list) and len(golden_rewrite) > 0:
        golden_rewrite = golden_rewrite[0]

    return s0_dict, golden_rewrite


def build_prompt_text(s0_dict):
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
    return user_text


def pipeline_runner_func(s0_dict, decomposer_tree):
    """Executes downstream RAG verification logic for reward settlement."""
    node_results = {}
    known_facts = {}
    full_sentence = s0_dict.get("Y_orig", "")
    
    leaf_nodes = [
        n for n in decomposer_tree 
        if isinstance(n, dict) and n.get("task_type") in ["LEAF_CONTEXT_CHECK", "LEAF_REALITY_CHECK"]
    ]
    
    for leaf in leaf_nodes:
        node_id = leaf.get("id")
        search_query = leaf.get("query", full_sentence)
        fact_result = core.retrieve_and_extract_leaf_facts(
            search_query=search_query,
            active_leaf_node=leaf,
            known_facts=known_facts
        )
        node_results[node_id] = fact_result
        if fact_result.get("extracted_fact"):
            known_facts[node_id] = fact_result.get("extracted_fact")

    final_tree_output = core.evaluate_tree_boolean_logic(decomposer_tree, node_results)
    
    rw = rewriter.TreeGuidedRewriter()
    corrected_sentence = rw.rewrite_sentence(
        full_sentence=full_sentence,
        execution_tree=final_tree_output.get("execution_tree", [])
    )

    return {
        "sentence_boolean_truth": final_tree_output.get("sentence_boolean_truth", False),
        "execution_tree": final_tree_output.get("execution_tree", []),
        "corrected_sentence": corrected_sentence
    }


def compute_gae(rewards, values, gamma=0.98, lam=0.95):
    """
    Generalized Advantage Estimation (GAE)
    """
    seq_len = len(rewards)
    advantages = torch.zeros(seq_len, device=DEVICE)
    last_gae_lam = 0.0
    
    for t in reversed(range(seq_len)):
        next_val = values[t + 1] if t + 1 < seq_len else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * last_gae_lam
        
    returns = advantages + values
    return advantages, returns


# ==========================================
# 3. Stage 1: Warm-up SFT
# ==========================================
def run_warmup_phase(actor_model, tokenizer, dataloader, optimizer):
    print("\n🔥 === Stage 1: Warm-up Supervised Fine-Tuning (SFT Eq. 1) === 🔥")
    actor_model.train()
    
    for epoch in range(WARMUP_EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            sample = {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in batch.items()}
            s0_dict, _ = extract_s0_and_rewrite(sample)
            
            golden_tree = sample.get("golden_action_tree", [])
            golden_tree_str = json.dumps(golden_tree, ensure_ascii=False)
            
            user_text = build_prompt_text(s0_dict)
            
            prompt_part = (
                f"<|im_start|>system\n{prompts.DECOMPOSER_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            target_part = f"{golden_tree_str}<|im_end|>"
            
            prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
            target_ids = tokenizer.encode(target_part, add_special_tokens=False)
            
            MAX_LEN = 1536
            if len(prompt_ids) + len(target_ids) > MAX_LEN:
                max_prompt_len = MAX_LEN - len(target_ids)
                if max_prompt_len > 0:
                    prompt_ids = prompt_ids[-max_prompt_len:]
                else:
                    target_ids = target_ids[:MAX_LEN - 100]
                    prompt_ids = prompt_ids[:100]

            input_ids_list = prompt_ids + target_ids
            labels_list = [-100] * len(prompt_ids) + target_ids
            
            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(DEVICE)
            labels = torch.tensor([labels_list], dtype=torch.long).to(DEVICE)
            
            outputs = actor_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if (step + 1) % 5 == 0 or step == len(dataloader) - 1:
                print(f"   [Warmup Epoch {epoch+1} | Step {step+1}/{len(dataloader)}] SFT Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"✅ Warmup Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")


# ==========================================
# 4. Stage 2: Online Actor-Critic PPO
# ==========================================
def evaluate_dev_set(actor_model, tokenizer, dev_loader, reward_engine):
    """验证集评估函数，返回平均 Reward"""
    actor_model.eval()
    total_rewards = []
    
    with torch.no_grad():
        for batch in dev_loader:
            sample = {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in batch.items()}
            s0_dict, golden_rewrite = extract_s0_and_rewrite(sample)

            user_text = build_prompt_text(s0_dict)
            prompt_text = (
                f"<|im_start|>system\n{prompts.DECOMPOSER_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

            generated_ids = actor_model.generate(
                **prompt_inputs,
                max_new_tokens=512,
                do_sample=False,  # 验证集使用 greedy search 确定性输出
                pad_token_id=tokenizer.eos_token_id
            )

            prompt_len = prompt_inputs.input_ids.shape[1]
            action_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
            cleaned_action = llm_adapter.clean_deepseek_reasoning_output(action_text)

            try:
                decomposer_tree = json.loads(cleaned_action)
                if not isinstance(decomposer_tree, list): decomposer_tree = []
            except Exception:
                decomposer_tree = []

            rag_output = pipeline_runner_func(s0_dict, decomposer_tree)
            reward_dict = reward_engine.calculate_reward(
                trajectory_data={"Y_orig": s0_dict.get("Y_orig", ""), "execution_tree": rag_output.get("execution_tree", [])},
                golden_rewrite=golden_rewrite,
                predicted_rewrite=rag_output.get("corrected_sentence", ""),
                z_logic=1 if rag_output.get("sentence_boolean_truth") else 0
            )
            total_rewards.append(reward_dict.get("r_total", 0.0))

    actor_model.train()
    return sum(total_rewards) / max(len(total_rewards), 1)


def run_ppo_phase(actor_model, critic_model, ref_model, tokenizer, ppo_loader, dev_loader, actor_opt, critic_opt, reward_engine):
    print("\n🚀 === Stage 2: Online PPO Reinforcement Learning === 🚀")
    actor_model.train()
    critic_model.train()
    ref_model.eval()

    global_step = 0
    best_eval_reward = -float("inf")
    no_improve_count = 0

    for epoch in range(100):
        for step, batch in enumerate(ppo_loader):
            if global_step >= MAX_PPO_STEPS:
                break

            sample = {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in batch.items()}
            s0_dict, golden_rewrite = extract_s0_and_rewrite(sample)

            user_text = build_prompt_text(s0_dict)
            prompt_text = (
                f"<|im_start|>system\n{prompts.DECOMPOSER_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            prompt_len = prompt_inputs.input_ids.shape[1]

            # 1. Rollout Policy Sampling a_t ~ pi_\theta
            with torch.no_grad():
                generated_ids = actor_model.generate(
                    **prompt_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            action_ids = generated_ids[:, prompt_len:]
            action_text = tokenizer.decode(action_ids[0], skip_special_tokens=True)
            seq_len = action_ids.shape[1]

            if seq_len == 0:
                continue

            # 记录旧策略 \pi_old 的 log_probs (锁死用于 Importance Sampling 比率)
            with torch.no_grad():
                old_logits = actor_model(generated_ids).logits[:, prompt_len - 1 : -1, :]
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                old_token_log_probs = torch.gather(old_log_probs, 2, action_ids.unsqueeze(-1)).squeeze(-1)[0]

            # 2. Execute RAG Environment Step and compute R_total
            cleaned_action = llm_adapter.clean_deepseek_reasoning_output(action_text)
            try:
                decomposer_tree = json.loads(cleaned_action)
                if not isinstance(decomposer_tree, list): decomposer_tree = []
            except Exception:
                decomposer_tree = []

            rag_output = pipeline_runner_func(s0_dict, decomposer_tree)
            reward_dict = reward_engine.calculate_reward(
                trajectory_data={"Y_orig": s0_dict.get("Y_orig", ""), "execution_tree": rag_output.get("execution_tree", [])},
                golden_rewrite=golden_rewrite,
                predicted_rewrite=rag_output.get("corrected_sentence", ""),
                z_logic=1 if rag_output.get("sentence_boolean_truth") else 0
            )
            r_total = reward_dict.get("r_total", 0.0)

            # 单步密集奖励构建
            rewards_k = torch.zeros(seq_len, device=DEVICE)
            rewards_k[-1] = r_total

            # 获取 Critic Values
            with torch.no_grad():
                values = critic_model(generated_ids)[:, prompt_len - 1 : -1][0]

            # 计算 GAE 优势与 Returns
            advantages, returns = compute_gae(rewards_k, values, gamma=GAMMA, lam=LAMBDA_GAE)
            if advantages.numel() > 1 and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. PPO 内层 Epoch 多次梯度更新
            for ppo_epoch in range(PPO_EPOCHS_PER_STEP):
                actor_opt.zero_grad()
                critic_opt.zero_grad()

                # 当前策略对数概率 \pi_\theta
                actor_logits = actor_model(generated_ids).logits[:, prompt_len - 1 : -1, :]
                actor_log_probs = F.log_softmax(actor_logits, dim=-1)
                curr_token_log_probs = torch.gather(actor_log_probs, 2, action_ids.unsqueeze(-1)).squeeze(-1)[0]

                # 重要性采样比率 r_t(\theta)
                ratio = torch.exp(curr_token_log_probs - old_token_log_probs)

                # Clipped Policy Loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
                l_clip = -torch.min(surr1, surr2).mean()

                # 策略熵正则化
                probs = F.softmax(actor_logits, dim=-1)
                entropy = -(probs * actor_log_probs).sum(dim=-1).mean()
                l_policy = l_clip - ENTROPY_COEF * entropy

                # Critic 价值损失
                curr_values = critic_model(generated_ids)[:, prompt_len - 1 : -1][0]
                l_value = F.mse_loss(curr_values, returns)

                # 联合损失与梯度反传
                total_loss = l_policy + VALUE_LOSS_WEIGHT * l_value

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)

                actor_opt.step()
                critic_opt.step()

            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                print(f" [PPO Step {global_step}/{MAX_PPO_STEPS}] Total Reward: {r_total:.4f} | Policy Loss: {l_policy.item():.4f} | Value Loss: {l_value.item():.4f} | Entropy: {entropy.item():.4f}")

            # 4. 定期验证集评估与早停触发
            if global_step % EVAL_INTERVAL == 0 and len(dev_loader) > 0:
                print(f"\n🔍 [Evaluation Checkpoint @ Step {global_step}] Evaluating on Validation Set...")
                eval_reward = evaluate_dev_set(actor_model, tokenizer, dev_loader, reward_engine)
                print(f"   📊 Mean Dev Reward: {eval_reward:.4f} (Best: {best_eval_reward:.4f})")

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    no_improve_count = 0
                    # 保存最佳模型
                    best_saved_dir = os.path.join(SAVED_CHECKPOINT_DIR, "best_checkpoint")
                    os.makedirs(best_saved_dir, exist_ok=True)
                    actor_model.save_pretrained(best_saved_dir)
                    tokenizer.save_pretrained(best_saved_dir)
                    print(f"   ✨ Improved! Saved best model to: {best_saved_dir}")
                else:
                    no_improve_count += 1
                    print(f"   ⚠️ No improvement. Patience count: {no_improve_count}/{PATIENCE}")

                if no_improve_count >= PATIENCE:
                    print(f"\n🛑 Triggering Early Stopping! Mean validation reward failed to improve across {PATIENCE} consecutive evaluation checkpoints.")
                    return

            torch.cuda.empty_cache()

    print("\n🎉 Online Actor-Critic PPO Training Complete!")


# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    print(f"Using Device: {DEVICE}")
    print(f"Loading Base Model: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # Actor Model \pi_\theta
    actor_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=target_dtype, trust_remote_code=True
    ).to(DEVICE)
    if hasattr(actor_model, "gradient_checkpointing_enable"):
        actor_model.gradient_checkpointing_enable()

    # Reference Model \pi_ref (Frozen copy)
    ref_model = copy.deepcopy(actor_model)
    for param in ref_model.parameters():
        param.requires_grad = False

    # Critic Model V_\phi
    critic_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=target_dtype, trust_remote_code=True
    ).to(DEVICE)
    if hasattr(critic_base, "gradient_checkpointing_enable"):
        critic_base.gradient_checkpointing_enable()
    critic_model = ValueHead(critic_base).to(DEVICE)

    # 优化器
    actor_opt = AdamW(actor_model.parameters(), lr=POLICY_LR)
    critic_opt = AdamW(critic_model.parameters(), lr=VALUE_LR)
    sft_opt = AdamW(actor_model.parameters(), lr=SFT_LR)

    reward_engine = REAPRewardEngine(alpha=0.5)

    # 1. Warm-up SFT Phase
    warmup_dataset = GenericJsonlDataset(WARMUP_DATA_PATH)
    if len(warmup_dataset) > 0:
        warmup_loader = DataLoader(warmup_dataset, batch_size=BATCH_SIZE, shuffle=True)
        run_warmup_phase(actor_model, tokenizer, warmup_loader, sft_opt)

    # 2. Online PPO Phase
    ppo_dataset = GenericJsonlDataset(PPO_DATA_PATH)
    dev_dataset = GenericJsonlDataset(PPO_DEV_PATH)
    
    if len(ppo_dataset) > 0:
        ppo_loader = DataLoader(ppo_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(dev_dataset) > 0 else []
        
        run_ppo_phase(
            actor_model, critic_model, ref_model, tokenizer, ppo_loader, dev_loader,
            actor_opt, critic_opt, reward_engine
        )

    # 3. Save Final Checkpoint
    os.makedirs(SAVED_CHECKPOINT_DIR, exist_ok=True)
    print(f"\n💾 Saving final fine-tuned Decomposer model to: {SAVED_CHECKPOINT_DIR}")
    actor_model.save_pretrained(SAVED_CHECKPOINT_DIR)
    tokenizer.save_pretrained(SAVED_CHECKPOINT_DIR)
    print("🎉 All training stages complete! Model saved successfully.")


if __name__ == "__main__":
    main()