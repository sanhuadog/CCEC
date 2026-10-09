# config.py
"""
Configuration settings for the RAG pipeline & PPO Training.
Loads sensitive keys and custom environment variables from .env file or environment variables.
"""
import os
from dotenv import load_dotenv

# Load .env environment variables if present
load_dotenv()

# --- AI Provider Selection (Options: local, vllm, qwen, zhipu, openai) ---
AI_PROVIDER = os.getenv('AI_PROVIDER', 'local').lower()

# --- Local & Checkpoint Paths (Anonymous relative paths) ---
LOCAL_MODEL_PATH = os.getenv(
    'LOCAL_MODEL_PATH', 
    os.path.join(".", "models", "DeepSeek-R1-Distill-Qwen-7B")
)
SAVED_CHECKPOINT_DIR = os.getenv(
    'SAVED_CHECKPOINT_DIR', 
    os.path.join(".", "checkpoints", "decomposer_ppo_final")
)

# --- API Keys (Leave blank for anonymous public distribution) ---
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')

# --- LLM Model Names ---
LOCAL_LLM_MODEL = "DeepSeek-R1-Distill-Qwen-7B"
ZHIPU_LLM_MODEL = os.getenv('ZHIPU_LLM_MODEL', 'glm-4-flash')
QWEN_LLM_MODEL = os.getenv('QWEN_LLM_MODEL', 'qwen2.5-7b-instruct')
OPENAI_LLM_MODEL = os.getenv('OPENAI_LLM_MODEL', 'gpt-4o-mini')

# --- VLLM Settings (Optional acceleration) ---
VLLM_HOST = os.getenv('VLLM_HOST', "http://localhost")
VLLM_LLM_MODEL = os.getenv('VLLM_LLM_MODEL', "DeepSeek-R1-Distill-Qwen-7B")
VLLM_LLM_PORT = int(os.getenv('VLLM_LLM_PORT', 8000))
VLLM_USE_DEDICATED_MODELS = True

# --- Search Service Configuration ---
SEARCH_SERVICE_HOST = os.getenv('SEARCH_SERVICE_HOST', '127.0.0.1')
SEARCH_SERVICE_PORT = int(os.getenv('SEARCH_SERVICE_PORT', 8090))

# --- RAG Parameters & Controls ---
TOP_K = int(os.getenv('TOP_K', 5))
SHOW_RETRIEVED_CONTEXT = True

# --- LLM Model Name Dynamic Assignment ---
if AI_PROVIDER == 'local':
    LLM_MODEL_NAME = LOCAL_MODEL_PATH
elif AI_PROVIDER == 'vllm':
    LLM_MODEL_NAME = VLLM_LLM_MODEL
elif AI_PROVIDER == 'qwen':
    LLM_MODEL_NAME = QWEN_LLM_MODEL
elif AI_PROVIDER == 'zhipu':
    LLM_MODEL_NAME = ZHIPU_LLM_MODEL
elif AI_PROVIDER == 'openai':
    LLM_MODEL_NAME = OPENAI_LLM_MODEL
else:
    LLM_MODEL_NAME = LOCAL_MODEL_PATH