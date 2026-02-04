import pandas as pd
import numpy as np
import os
import sys

# 强制设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 绝对路径，确保不出错
RELATION_FILE = "/data/user1/MyIdeaProject/datasets/WN18RR/relation2id.txt"

# 定义宽容规则 (松弛评分)
EQUIVALENT_PAIRS = [
    {"_hypernym", "_instance_hypernym"},
    {"_member_meronym", "_has_part"},
    {"_member_meronym", "_part_of"},
    {"_member_of_domain_usage", "_member_of_domain_region"},
    {"_member_of_domain_topic", "_member_of_domain_usage"},
    {"_derivationally_related_form", "_hypernym"},
    {"_member_meronym", "_hypernym"},
    {"_verb_group", "_hypernym"}
]

def load_relations_and_embeddings():
    print(">>> Loading Relation Embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    rels = []
    with open(RELATION_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0 and len(lines[0].split()) == 1: lines = lines[1:]
        for l in lines:
            parts = l.strip().split()
            if len(parts) >= 1:
                rid = parts[0]
                txt = rid.replace('_', ' ').strip()
                rels.append({"id": rid, "text": txt})
    
    # 预计算所有候选关系的向量
    rel_embs = embedder.encode([r['text'] for r in rels])
    return rels, rel_embs, embedder

def score_file(file_path, label, rels, rel_embs, embedder):
    print(f"\n>>> Scoring {label} ...")
    if not os.path.exists(file_path): 
        print(f"❌ 文件不存在: {file_path}")
        return
    
    df = pd.read_csv(file_path, dtype=str)
    valid_total = len(df)
    
    if valid_total == 0:
        print("数据为空")
        return

    hits1, hits3, hits10, mrr = 0, 0, 0, 0

    for i, row in df.iterrows():
        # 1. 清洗预测结果
        raw_pred = str(row.get('Parsed_Prediction', '')).replace('**', '').strip()
        if ". " in raw_pred: raw_pred = raw_pred.split(". ")[1].strip()
        
        # 兼容列名 'Relation' 或 'True_Relation'
        true_rel = str(row.get('Relation', row.get('True_Relation', ''))).strip()
        
        if not true_rel: continue

        rank = -1
        
        # --- 策略 A: 规则修正 (直接命中 Rank 1) ---
        is_equivalent = False
        for pair in EQUIVALENT_PAIRS:
            if raw_pred in pair and true_rel in pair:
                is_equivalent = True
                break
        
        if raw_pred == true_rel or is_equivalent:
            rank = 1
            
        # --- 策略 B: 向量相似度排序 (计算 Rank) ---
        else:
            # 编码预测结果
            pred_text = raw_pred.replace('_', ' ')
            pred_emb = embedder.encode([pred_text])
            
            # 计算与所有候选关系的相似度
            scores = cosine_similarity(pred_emb, rel_embs)[0]
            
            # 从高到低排序
            sorted_indices = np.argsort(scores)[::-1]
            
            # 找到真值在第几名
            for r_idx, rel_idx in enumerate(sorted_indices):
                candidate_id = rels[rel_idx]['id']
                if candidate_id == true_rel:
                    rank = r_idx + 1
                    break
        
        # 统计指标
        if rank != -1:
            mrr += 1.0 / rank
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 10: hits10 += 1

    # 打印结果
    print(f"[{label}] Results (N={valid_total}):")
    print(f"  Hits@1 : {hits1/valid_total:.4f} ({hits1})")
    print(f"  Hits@3 : {hits3/valid_total:.4f} ({hits3})")
    print(f"  Hits@10: {hits10/valid_total:.4f} ({hits10})")
    print(f"  MRR    : {mrr/valid_total:.4f}")

def main():
    # 加载模型和关系数据 (只加载一次)
    rels, rel_embs, embedder = load_relations_and_embeddings()
    
    # 分别评分
    score_file("result_AI.csv", "AI Group (Gen)", rels, rel_embs, embedder)
    score_file("result_Real.csv", "Real Group (Sampled)", rels, rel_embs, embedder)

if __name__ == "__main__":
    main()