import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config.config import get_config
from utils.common_utils import clean_relation_text


def run_eval(cfg):
    print(f">>> 评估开始: {cfg['output_file']}")
    if not os.path.exists(cfg['output_file']):
        print("错误: 找不到结果文件")
        return

    embedder = SentenceTransformer(cfg['sbert_model'])

    # 1. 加载所有标准关系候选
    rels = []
    with open(cfg['rel_path'], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines[0].split()) == 1: lines = lines[1:]
        for l in lines:
            parts = l.strip().split()
            rid = parts[0] if not parts[0].isdigit() else parts[1]
            rels.append(rid)

    rel_embs = embedder.encode([clean_relation_text(r) for r in rels])

    # 2. 计算指标
    df = pd.read_csv(cfg['output_file'], dtype=str)
    mrr, h1, h3, h10 = 0, 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        true_r = str(row['True_Relation']).strip()
        pred_r = str(row['Predicted_ID']).strip()

        if pred_r == true_r:
            rank = 1
        else:
            # 语义排名：计算预测文本与所有候选关系的相似度
            p_text = clean_relation_text(pred_r)
            p_emb = embedder.encode([p_text])
            scores = cosine_similarity(p_emb, rel_embs)[0]
            # 找到正确答案在预测结果中的排名
            sort_idx = np.argsort(scores)[::-1]
            rank = np.where(np.array(rels)[sort_idx] == true_r)[0][0] + 1

        mrr += 1.0 / rank
        if rank <= 1: h1 += 1
        if rank <= 3: h3 += 1
        if rank <= 10: h10 += 1

    n = len(df)
    res_str = f"Hits@1: {h1 / n:.4f} | Hits@3: {h3 / n:.4f} | Hits@10: {h10 / n:.4f} | MRR: {mrr / n:.4f}"
    print(res_str)

    # 保存结果报告
    report_path = cfg['output_file'].replace('.csv', '_metrics.txt')
    with open(report_path, 'w') as f:
        f.write(res_str)