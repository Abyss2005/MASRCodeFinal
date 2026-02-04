import pandas as pd
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 必备

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fb15k_loader import FB15kLoader

# 使用 Loader 自动读取关系，不需要手动解析 relation2id.txt
KG_ROOT = "/data/user1/MyIdeaProject/datasets/fb15k-237/get_neighbor"

def clean_text(text):
    # FB15k 专用清洗: /film/actor -> film actor
    return str(text).replace('/', ' ').replace('_', ' ').replace('.', ' ').strip()

def get_metrics(file_path, rel_embs, rel_ids, embedder):
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path, dtype=str)
    
    hits1, hits3, hits10, mrr = 0, 0, 0, 0
    valid = 0
    
    for _, row in df.iterrows():
        raw_pred = str(row.get('Parsed_Prediction', '')).strip()
        true_rel = str(row['True_Relation']).strip()
        if not true_rel: continue
        
        rank = -1
        # 1. 精确匹配
        if raw_pred == true_rel:
            rank = 1
        # 2. 向量匹配
        else:
            pred_text = clean_text(raw_pred)
            pred_emb = embedder.encode([pred_text])
            scores = cosine_similarity(pred_emb, rel_embs)[0]
            sorted_idx = np.argsort(scores)[::-1]
            
            for r_idx, idx in enumerate(sorted_indices := sorted_idx):
                if rel_ids[idx] == true_rel:
                    rank = r_idx + 1
                    break
        
        if rank != -1:
            valid += 1
            mrr += 1.0 / rank
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 10: hits10 += 1
            
    return {"H1": hits1/valid, "H3": hits3/valid, "H10": hits10/valid, "MRR": mrr/valid}

def main():
    print(">>> [Score] 初始化 SBERT 和 关系列表...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 从 Loader 获取标准关系列表 (过滤 _inv)
    loader = FB15kLoader(KG_ROOT)
    rel_ids = [r for r in loader.relation2id.keys() if not r.endswith('_inv')]
    # 预计算 Embeddings
    rel_texts = [clean_text(r) for r in rel_ids]
    rel_embs = embedder.encode(rel_texts)
    
    print("\n" + "="*40)
    
    # 评分 AI 组
    res_ai = get_metrics("result_AI.csv", rel_embs, rel_ids, embedder)
    if res_ai:
        print(f"[AI Group] (Gen Images)")
        print(f"  Hits@1 : {res_ai['H1']:.4f}")
        print(f"  Hits@3 : {res_ai['H3']:.4f}")
        print(f"  Hits@10: {res_ai['H10']:.4f}")
        print(f"  MRR    : {res_ai['MRR']:.4f}")
    
    print("-" * 40)
    
    # 评分 Real 组
    res_real = get_metrics("result_Real.csv", rel_embs, rel_ids, embedder)
    if res_real:
        print(f"[Real Group] (Real Images)")
        print(f"  Hits@1 : {res_real['H1']:.4f}")
        print(f"  Hits@3 : {res_real['H3']:.4f}")
        print(f"  Hits@10: {res_real['H10']:.4f}")
        print(f"  MRR    : {res_real['MRR']:.4f}")
        
    print("="*40)

if __name__ == "__main__":
    main()