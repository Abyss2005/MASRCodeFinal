import pandas as pd
import os
import random
from wn18rr_loader import WN18RRLoader

# ================= 路径配置 =================
KG_ROOT = "/data/user1/MyIdeaProject/datasets/WN18RR"
TEST_FILE = os.path.join(KG_ROOT, "test2id.txt")
# 真实图片路径
REAL_IMG_ROOT = "/data/user1/MyIdeaProject/ai_image_generation/real_images_full/wn18rr"

OUTPUT_CSV = "sampled_50_triples.csv"
# ===========================================

def has_real_img(eid):
    """检查真实图片是否存在 (兼容 n 前缀)"""
    # 路径 1: root/eid
    if os.path.exists(os.path.join(REAL_IMG_ROOT, eid)) and len(os.listdir(os.path.join(REAL_IMG_ROOT, eid))) > 0:
        return True
    # 路径 2: root/n+eid
    if os.path.exists(os.path.join(REAL_IMG_ROOT, "n" + eid)) and len(os.listdir(os.path.join(REAL_IMG_ROOT, "n" + eid))) > 0:
        return True
    return False

def main():
    print(">>> [Step 0] 正在筛选 50 条高质量三元组 (优先 Real 组有图)...")
    loader = WN18RRLoader(KG_ROOT)
    
    # 1. 读取所有测试题
    all_triples = []
    with open(TEST_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0 and len(lines[0].split()) == 1: lines = lines[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue
            h = loader.id2entity[int(parts[0])]
            t = loader.id2entity[int(parts[1])]
            r = loader.id2relation[int(parts[2])]
            all_triples.append({"Head": h, "Relation": r, "Tail": t})
            
    print(f"    测试集总数: {len(all_triples)}")
    
    # 2. 分级筛选
    tier1 = [] # 双端都有真实图片 (最公平的对决)
    tier2 = [] # 只有 Head 有真实图片
    tier3 = [] # 只有 Tail 有真实图片
    tier4 = [] # 都没有 (作为兜底)
    
    print("    正在扫描真实图片覆盖情况 (稍慢请等待)...")
    # 为了速度，随机打乱后扫描，扫够就停
    random.shuffle(all_triples)
    
    for item in all_triples:
        h_has = has_real_img(item['Head'])
        t_has = has_real_img(item['Tail'])
        
        if h_has and t_has:
            tier1.append(item)
        elif h_has:
            tier2.append(item)
        elif t_has:
            tier3.append(item)
        else:
            tier4.append(item)
            
        # 如果 Tier 1 + Tier 2 已经够多了，就提前结束扫描
        if len(tier1) + len(tier2) > 60:
            break
            
    print(f"    筛选结果分布: 双端有图={len(tier1)}, Head有图={len(tier2)}, Tail有图={len(tier3)}, 无图={len(tier4)}")
    
    # 3. 组合最终名单 (凑够 50 个)
    final_list = tier1 + tier2 + tier3 + tier4
    final_list = final_list[:50]
    
    pd.DataFrame(final_list).to_csv(OUTPUT_CSV, index=False)
    print(f">>> 最终考题生成: {OUTPUT_CSV} (共 {len(final_list)} 条)")
    print("    这组数据保证了 Real 组拥有尽可能多的图片，是真正的'巅峰对决'。")

if __name__ == "__main__":
    main()