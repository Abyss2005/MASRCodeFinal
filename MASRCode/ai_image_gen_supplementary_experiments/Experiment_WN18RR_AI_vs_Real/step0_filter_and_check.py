import json
import pandas as pd
import os
from wn18rr_loader import WN18RRLoader

# ================= 路径配置 =================
SAMPLED_JSON = "/data/user1/MyIdeaProject/ai_image_generation/data/wn18rr_sampled.json"
EXISTING_IMG_DIR = "/data/user1/MyIdeaProject/ai_image_generation/images/wn18rr"
KG_ROOT = "/data/user1/MyIdeaProject/datasets/WN18RR"
TEST_FILE = os.path.join(KG_ROOT, "test2id.txt")
DESC_FILE = os.path.join(KG_ROOT, "wordnet-definitions.txt")

OUTPUT_CSV = "sampled_50_triples.csv"
MISSING_LIST_JSON = "missing_tails_to_generate.json"
# ===========================================

def load_definitions():
    desc_map = {}
    with open(DESC_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2: desc_map[parts[0]] = parts[1]
    return desc_map

def main():
    print(">>> [Step 0] 筛选考题并检查图片覆盖率...")
    with open(SAMPLED_JSON, 'r') as f:
        sampled_data = json.load(f)
    head_ids = set([str(item['id']).strip() for item in sampled_data])
    
    desc_map = load_definitions()
    loader = WN18RRLoader(KG_ROOT)
    
    selected_triples = []
    missing_tails = {} 
    
    with open(TEST_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0 and len(lines[0].split()) == 1: lines = lines[1:]
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            
            h_id = loader.id2entity[h]
            t_id = loader.id2entity[t]
            r_name = loader.id2relation[r]
            
            if h_id in head_ids:
                # 检查 Tail 是否有图
                if not os.path.exists(os.path.join(EXISTING_IMG_DIR, f"{t_id}.jpg")):
                    missing_tails[t_id] = desc_map.get(t_id, "Concept")
                
                selected_triples.append({"Head": h_id, "Relation": r_name, "Tail": t_id})
                if len(selected_triples) >= 50: break
    
    pd.DataFrame(selected_triples).to_csv(OUTPUT_CSV, index=False)
    print(f">>> 考题生成完毕: {OUTPUT_CSV} (共 {len(selected_triples)} 条)")
    
    missing_list = [{"id": k, "desc": v} for k, v in missing_tails.items()]
    with open(MISSING_LIST_JSON, 'w') as f:
        json.dump(missing_list, f, indent=2)
    print(f">>> 待补全 Tail 图片数量: {len(missing_list)} (保存至 {MISSING_LIST_JSON})")

if __name__ == "__main__":
    main()