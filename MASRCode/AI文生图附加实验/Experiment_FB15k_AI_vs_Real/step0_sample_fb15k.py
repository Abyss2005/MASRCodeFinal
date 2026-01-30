import pandas as pd
import os, random
from fb15k_loader import FB15kLoader

# ================= 配置 =================
KG_ROOT = "/data/user1/MyIdeaProject/datasets/fb15k-237/get_neighbor"
TEST_FILE = os.path.join(KG_ROOT, "test.txt") # FB15k通常是test.txt
REAL_IMG_ROOT = "/data/user1/MyIdeaProject/FB15k-images"
OUTPUT_CSV = "sampled_50_triples.csv"
# =======================================

def has_real_img(eid):
    # /m/010xjr -> m.010xjr
    folder = eid.lstrip('/').replace('/', '.')
    path = os.path.join(REAL_IMG_ROOT, folder)
    return os.path.exists(path) and len(os.listdir(path)) > 0

def main():
    print(">>> [Step 0] 筛选 FB15k-237 (优先 Real 有图)...")
    loader = FB15kLoader(KG_ROOT)
    
    all_triples = []
    # 尝试读取 test.txt 或 test2id.txt
    if os.path.exists(TEST_FILE):
        with open(TEST_FILE, 'r') as f:
            for l in f:
                p = l.strip().split()
                if len(p)>=3: all_triples.append({"Head": p[0], "Relation": p[2], "Tail": p[1]}) # FB15k 格式通常是 h t r 或者 h r t，请检查！
                # 修正：通常是 h t r，或者是 h r t。这里假设 data_loader 读取是对的。
                # 保险起见，我们假设文件是 h t r (原始FB15k) 或 h r t (OpenKE)
                # 你的 data_loader 似乎处理了 id 转换，这里我们直接读原始字符串方便查图
    
    # 重新读取原始 test.txt 确保 ID 是字符串 /m/xxx
    all_triples = []
    with open(TEST_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines[0].split()) == 1: lines = lines[1:]
        for l in lines:
            p = l.strip().split()
            if len(p) >= 3:
                # 假设格式: h t r (FB15k标准)
                # 如果你的文件是 ID，需要用 loader 转回字符串
                try:
                    # 尝试转换 ID -> String
                    h = loader.id2entity[int(p[0])]
                    t = loader.id2entity[int(p[1])]
                    r = loader.id2relation[int(p[2])]
                except:
                    # 本身就是 String
                    h, t, r = p[0], p[1], p[2]
                all_triples.append({"Head": h, "Tail": t, "Relation": r})

    print(f"    Total Test: {len(all_triples)}")
    
    # 分级筛选
    tier1, tier2, tier3, tier4 = [], [], [], []
    random.shuffle(all_triples)
    
    for item in all_triples:
        h_has = has_real_img(item['Head'])
        t_has = has_real_img(item['Tail'])
        
        if h_has and t_has: tier1.append(item)
        elif h_has: tier2.append(item)
        elif t_has: tier3.append(item)
        else: tier4.append(item)
        
        if len(tier1) + len(tier2) > 60: break
    
    final = (tier1 + tier2 + tier3 + tier4)[:50]
    pd.DataFrame(final).to_csv(OUTPUT_CSV, index=False)
    print(f">>> 采样完成: {len(final)} 条 (Tier1: {len(tier1)})")

if __name__ == "__main__":
    main()