# Commonsense Error Correction Based on Hierarchical Traceability Reasoning

Official repository for the **anonymous submission** to ARR. This repository contains the source code, evaluation tools, prompt templates, and dataset for **CCEC** (Contextual Commonsense Error Correction) and the **CCECD** benchmark.

**Note on Anonymity:** To adhere to double-blind review policies, all author details, institutional affiliations, and specific server/local path signatures have been removed or anonymized.
Anonymous link: https://anonymous.4open.science/r/CCEC-0FF0/

---

##  Overview

Large Language Models (LLMs) often struggle with commonsense error correction when complex contextual constraints, carrier attributes, and historical/domain assumptions intersect. This repository provides a complete implementation of our proposed Hierarchical Traceability Reasoning Framework, which addresses this challenge through:

1. **Tree-Structured Disentanglement:** Decomposing complex text into contextual constraints, physical reality, and universal commonsense.
2. **Dual-Agent Retrieval:** Querying specialized knowledge bases (academic/commonsense) and open-web/scientific sources with non-uniform confidence weighting.
3. **Bottom-Up Logic Aggregation & Local Mask Correction:** Performing precise error localization and minimal edit revisions via tree-guided logic pruning and reinforcement learning (PPO).

##  Dataset Notice (CCECD Benchmark)

To ensure full reproducibility while adhering to the anonymous submission guidelines, this repository currently includes a **curated subset of 100 sample instances for each split** (`Warmup`, `PPO_Train`, `Dev`, and `Test`) located in `data/dataset_splits/`. 

* **Sample Data Access:** These representative subsets allow reviewers to immediately verify the schema, hierarchical annotations, and inspect prompt-execution pipelines.
* **Full Benchmark Release:** The complete **CCECD** dataset (comprising all generated trajectories, evidence chains, and golden annotations) along with pre-processed vector indices will be fully released to the public upon official publication.

## Reproducibility Notes
1. Source Code:
The core pipeline library `rag_pipeline_lib` is packaged in an encrypted archive.
Publicly accessible files include training entry, evaluation scripts and metric calculation code.
These public files are sufficient to verify the experimental setup and results described in our paper.

2. Model Weights:
No model checkpoints are provided in this repository.
Base model weights can be downloaded from their official public releases following the codes we provide.

3. Dataset:
Only a sampled subset of 100 test examples is released here for demonstration purposes.
The full dataset construction procedure is described in our paper.


---

##  Environment Setup & Quick Start

### Prerequisites

* **OS**: Linux / Windows
* **Python**: >= 3.10
* **GPU**: NVIDIA GPU with CUDA >= 11.8 (CUDA 12.1 recommended, 8GB+ VRAM)
* **Ollama**: Installed and running locally for downstream Fact Extractor & Rewriter services

### 1. Environment Setup

Run the following unified setup commands to configure the Conda environment, install PyTorch with CUDA support, pull required dependencies, download NLTK data, and prepare the local Ollama service:

```bash
# 1. Create & activate environment, install PyTorch and dependencies
conda create -n reap python=3.10 -y && conda activate reap
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Initialize NLTK evaluation assets & start Ollama DeepSeek model service
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
ollama pull deepseek-r1:7b

```

## 2. Commonsense Knowledge Base Setup

Initialize and build the multi-source knowledge base (incorporating Wikipedia, academic, and web corpora) with dense vector embeddings for retrieval:

```bash
# 1. Enter the knowledge base directory, download the raw corpus, and build vector DB
cd multi_source_kb
python download_corpus.py
python build_commonsense_vector_db.py

# 2. Return to the project root directory
cd ..

```

> 💡 **Note**: `build_commonsense_vector_db.py` uses `FAISS` and `SentenceTransformers` to encode multi-source documents into dense vector indexes. Ensure your GPU or CPU has sufficient memory during the embedding generation process.

## 3. Model Weights Download
Download the DeepSeek-R1-Distill-Qwen-7B local model weights into the models/ directory prior to running state initialization, PPO training, and evaluation:

```Bash
# Execute model download script
python download.py
```
💡 Note: Ensure the weights are saved to models/DeepSeek-R1-Distill-Qwen-7B/. The system will automatically detect this directory for local offline inference and RL fine-tuning.

## 4. Input Diagnostic Analysis & Initial State Construction

Run `State_Init.py` to parse raw input text, execute diagnostic reasoning, and construct structured initial environment states ($S_0$) containing context shells and assertions.

### Usage Instructions

Open `State_Init.py` and modify line 11 (`DATASET_PATH`) to point to your target dataset split:

```python
# Line 11 in State_Init.py
DATASET_PATH = "data/dataset_splits/train_set.jsonl"  # Replace with dev_set.jsonl or text_set.jsonl as needed

```

Then execute:

```bash
python State_Init.py

```

The parsed output will be saved under `data/dataset_splits/` with the `_processed_stage1_2.jsonl` suffix (e.g., `train_set_processed_stage1_2.jsonl`).


## 5. Decomposer PPO Reinforcement Learning Training

Train the Decomposer agent using PPO (Proximal Policy Optimization) guided by bottom-up logical AND aggregation rewards, contextual safety signals, and minimal-edit penalties.

### Training Execution

Run the main PPO training script:

```bash
python train_ppo_decomposer.py

```

> 💡 **Training Details & Checkpoints**:
> * **Input Data**: Automatically loads the processed training split `data/dataset_splits/train_set_processed_stage1_2.jsonl`.
> * **Base Model**: Loads local weights from `models/DeepSeek-R1-Distill-Qwen-7B/`.
> * **Checkpoint Saving**: Model checkpoints during training and the final best Decomposer model will be saved to:
> ```text
> checkpoints/decomposer_ppo_final/best_checkpoint/
> 
> ```
> 
> 
> 
>

## 6. Evaluation

Evaluate the pipeline on the test split using the trained PPO Decomposer checkpoint, perform multi-source retrieval verification and tree-guided rewriting, and compute quantitative paper metrics.

### Step 1: Run Full-Pipeline Inference on Test Set

Execute `evaluate_test_set.py` to process the stage-1 processed test split (`text_set_processed_stage1_2.jsonl`). It automatically loads the best PPO checkpoint, performs bottom-up boolean verification, applies minimal-edit sentence rewriting, and saves clean boolean verdicts and corrected sentences:

```bash
python evaluate_test_set.py

```

* **Input**: `data/dataset_splits/test_set_processed_stage1_2.jsonl`
* **Checkpoint**: `checkpoints/decomposer_ppo_final/best_checkpoint/`
* **Output Results**: Saved to `data/dataset_splits/test_set_results.jsonl`

---

### Step 2: Compute Quantitative Performance Metrics

Calculate both error detection ACC/F1 metrics and standard generation/rewriting metrics (ROUGE-L, METEOR, BERTScore-F1, SARI) on the inference results:

```bash
python compute_metrics.py
```
