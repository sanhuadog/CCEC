# compute_metrics.py
import os
import sys
import json
import argparse
import warnings
import nltk
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# 屏蔽无用警告
warnings.filterwarnings("ignore")

# 动态挂载项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================
# SARI 标准英文计算函数 (Xu et al. 2016)
# ==========================================
def sari_compute_one(system_output: str, source: str, references: list) -> dict:
    """
    SARI metric from Xu et al. 2016 (Standard English implementation).
    """
    def tokenize(sent: str):
        return word_tokenize(sent.strip().lower())

    def ngram(tok_list, n=1):
        if len(tok_list) < n:
            return set()
        return set(zip(*[tok_list[i:] for i in range(n)]))

    def avg(arr):
        return sum(arr) / len(arr) if arr else 0.0

    source_toks = tokenize(source)
    pred_toks = tokenize(system_output)
    ref_tok_list = [tokenize(r) for r in references]

    # 1. Keep
    src_ngram = ngram(source_toks, n=1)
    pred_ngram = ngram(pred_toks, n=1)
    ref_ngrams = [ngram(r, n=1) for r in ref_tok_list]

    keep_recall_list = []
    for ref_ng in ref_ngrams:
        keep_overlap = src_ngram & pred_ngram & ref_ng
        keep_gold = src_ngram & ref_ng
        keep_recall = len(keep_overlap) / len(keep_gold) if len(keep_gold) > 0 else 0.0
        keep_recall_list.append(keep_recall)
    keep = avg(keep_recall_list)

    # 2. Add
    add_recall_list = []
    pred_only = pred_ngram - src_ngram
    for ref_ng in ref_ngrams:
        add_overlap = pred_only & ref_ng
        add_gold = ref_ng - src_ngram
        add_recall = len(add_overlap) / len(add_gold) if len(add_gold) > 0 else 0.0
        add_recall_list.append(add_recall)
    add = avg(add_recall_list)

    # 3. Delete
    del_recall_list = []
    src_only = src_ngram - pred_ngram
    for ref_ng in ref_ngrams:
        del_gold = src_ngram - ref_ng
        del_overlap = src_only & del_gold
        del_recall = len(del_overlap) / len(del_gold) if len(del_gold) > 0 else 0.0
        del_recall_list.append(del_recall)
    delete = avg(del_recall_list)

    sari = (keep + add + delete) / 3.0
    return {"sari": sari, "keep": keep, "add": add, "delete": delete}


# ==========================================
# 辅助工具函数
# ==========================================
def read_jsonl(file_path: str) -> list:
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def bool_convert(x) -> bool:
    """防错转换：将 'True'/'False' 字符串或布尔值统一转成 Python 布尔"""
    if isinstance(x, str):
        s = x.strip().lower()
        return s == "true"
    return bool(x)


def get_field_val(entry: dict, keys: list, default=""):
    """多关键字兜底提取字段，兼容不同测试结果文件的命名约定"""
    for k in keys:
        if k in entry and entry[k] is not None:
            return entry[k]
    return default


# ==========================================
# 主评估程序
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="REAP Evaluation Metrics Script for English Data")
    parser.add_argument("--gold_path", type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "dataset_splits", "text_set_processed_stage1_2.jsonl"),
                        help="Ground Truth JSONL Path")
    parser.add_argument("--pred_path", type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "dataset_splits", "test_set_results.jsonl"),
                        help="Model Predictions JSONL Path")
    parser.add_argument("--start_idx", type=int, default=0, help="Evaluation start index")
    parser.add_argument("--end_idx", type=int, default=None, help="Evaluation end index (None for EOF)")

    args = parser.parse_args()

    # 1. 下载英文 NLTK 必要资源
    for res in ["wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

    # 2. 读取数据
    print("==================================================")
    print(" 📊 REAP English Text Verification & Rewriting Metrics Evaluation")
    print("==================================================")
    print(f"👉 Ground Truth Path : {args.gold_path}")
    print(f"👉 Predictions Path  : {args.pred_path}")

    gold_data = read_jsonl(args.gold_path)
    pred_data = read_jsonl(args.pred_path)

    if not gold_data or not pred_data:
        print("❌ Dataset is empty. Cannot compute metrics!")
        return

    # 切片区间选取
    end_idx = args.end_idx if args.end_idx is not None else len(gold_data)
    gold_slice = gold_data[args.start_idx:end_idx]
    pred_slice = pred_data[args.start_idx:end_idx]

    use_len = min(len(gold_slice), len(pred_slice))
    gold_data = gold_slice[:use_len]
    pred_data = pred_slice[:use_len]

    print(f"👉 Slice Range       : [{args.start_idx}:{end_idx}]")
    print(f"👉 Total Samples     : {use_len}\n")

    # 3. 收集判断标签与英文改写文本
    y_gold_judge = []
    y_pred_judge = []
    gold_correction_list = []
    pred_correction_list = []
    source_text_list = []

    for g, p in zip(gold_data, pred_data):
        gj = bool_convert(get_field_val(g, ["is_correct", "pred_judge", "sentence_boolean_truth"], True))
        pj = bool_convert(get_field_val(p, ["is_correct", "global_judge", "pred_judge"], True))
        
        gc = str(get_field_val(g, ["corrected_full_sentence", "golden_rewrite", "corrected_sentence", "pred_correction"], ""))
        pc = str(get_field_val(p, ["corrected_sentence", "correction", "pred_correction"], ""))
        src_sent = str(get_field_val(g, ["full_sentence", "original_sentence", "Y_orig"], ""))

        y_gold_judge.append(1 if gj else 0)
        y_pred_judge.append(1 if pj else 0)
        gold_correction_list.append(gc)
        pred_correction_list.append(pc)
        source_text_list.append(src_sent)

    # ==========================================
    # 指标 1: 分类指标 (ACC & F1)
    # ==========================================
    acc = accuracy_score(y_gold_judge, y_pred_judge)
    f1 = f1_score(y_gold_judge, y_pred_judge, average="binary", pos_label=1)

    # ==========================================
    # 指标 2: ROUGE-L (英文标准按空格/Stemmer)
    # ==========================================
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl_scores = []
    for ref, pred in zip(gold_correction_list, pred_correction_list):
        s = scorer.score(ref, pred)
        rl_scores.append(s["rougeL"].fmeasure)
    rougeL = sum(rl_scores) / max(len(rl_scores), 1)

    # ==========================================
    # 指标 3: BERTScore-F1 (英文指定 lang="en")
    # ==========================================
    P, R, F1_bert = bert_score_func(pred_correction_list, gold_correction_list, lang="en", verbose=False)
    bert_f1 = float(F1_bert.mean())

    # ==========================================
    # 指标 4: SARI (英文版)
    # ==========================================
    sari_scores = []
    for src, pred, ref in zip(source_text_list, pred_correction_list, gold_correction_list):
        res = sari_compute_one(system_output=pred, source=src, references=[ref])
        sari_scores.append(res["sari"])
    sari_avg = sum(sari_scores) / max(len(sari_scores), 1)

    # ==========================================
    # 指标 5: METEOR (英文 NLTK Tokenize)
    # ==========================================
    meteor_list = []
    for ref, pred in zip(gold_correction_list, pred_correction_list):
        ref_tok = word_tokenize(ref.lower())
        pred_tok = word_tokenize(pred.lower())
        m = meteor_score([ref_tok], pred_tok)
        meteor_list.append(m)
    meteor_avg = sum(meteor_list) / max(len(meteor_list), 1)

    # ==========================================
    # 结果汇总控制台打印与 Markdown 输出
    # ==========================================
    print("===== 1. Error Detection Metrics (Classification) =====")
    print(f"Accuracy (ACC) : {acc * 100:.2f} %")
    print(f"F1-score       : {f1 * 100:.2f} %")

    print("\n===== 2. Sentence Rewriting Metrics (Generation) =====")
    print(f"ROUGE-L        : {rougeL * 100:.2f} %")
    print(f"METEOR         : {meteor_avg * 100:.2f} %")
    print(f"BERTScore-F1   : {bert_f1 * 100:.2f} %")
    print(f"SARI           : {sari_avg * 100:.2f} %")

    print("\n===== 3. Markdown Table =====")
    print("| ACC (%) | F1 (%) | ROUGE-L (%) | METEOR (%) | BERTScore-F1 (%) | SARI (%) |")
    print("|:-------:|:------:|:-----------:|:----------:|:----------------:|:--------:|")
    print(f"| {acc * 100:.2f} | {f1 * 100:.2f} | {rougeL * 100:.2f} | {meteor_avg * 100:.2f} | {bert_f1 * 100:.2f} | {sari_avg * 100:.2f} |")
    print("==================================================")


if __name__ == "__main__":
    main()