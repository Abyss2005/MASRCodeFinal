import os, json, time, base64, pandas as pd
from openai import OpenAI

# ================= 配置 =================
API_KEY = ""
BASE_URL = "https://35.aigcbest.top/v1"

# 图片目录
AI_IMAGE_DIR = "./AI_images"
# 【新增】原始定义文件路径 (必须和 Real 组保持一致)
DESC_FILE = "/data/user1/MyIdeaProject/datasets/WN18RR/wordnet-definitions.txt"
INPUT_CSV = "sampled_50_triples.csv"
OUTPUT_JSON = "desc_AI.json"
# =======================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_definitions():
    """加载原始文本定义"""
    d = {}
    if not os.path.exists(DESC_FILE):
        print(f"⚠️ 警告: 找不到定义文件 {DESC_FILE}")
        return d
    with open(DESC_FILE, 'r') as f:
        for l in f:
            p = l.strip().split('\t')
            if len(p) >= 2: d[p[0]] = p[1]
    return d

def get_b64(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

def main():
    print(f">>> [Group AI] 生成描述 (文本+视觉融合版)...")
    if not os.path.exists(INPUT_CSV): return
    
    df = pd.read_csv(INPUT_CSV, dtype=str)
    entities = set(df['Head'].tolist() + df['Tail'].tolist())
    desc_map = load_definitions() # 【新增】加载定义
    
    print(f">>> 待处理实体数: {len(entities)}")
    
    results = {}
    for i, eid in enumerate(entities):
        eid = str(eid).strip()
        # 【新增】获取原始文本
        origin_text = desc_map.get(eid, f"Concept {eid}")
        
        path = os.path.join(AI_IMAGE_DIR, f"{eid}.jpg")
        b64 = get_b64(path)
        
        if b64:
            try:
                print(f"[{i+1}/{len(entities)}] Analyzing {eid} (High Res)...")
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": [
                        # 【修改】Prompt 加入原始定义，指导模型看图
                        {"type": "text", "text": f"Definition: {origin_text}\nTask: Analyze this AI-generated image concisely based on the definition: Category, Components, Appearance."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
                    ]}], 
                    max_tokens=300
                )
                # 【修改】拼接：原始文本 + 视觉描述
                results[eid] = f"{origin_text}\n\n[AI Image Analysis (HD)]:\n{res.choices[0].message.content}"
            except Exception as e:
                print(f"Error: {e}")
                results[eid] = origin_text # 出错回退到纯文本
        else:
            print(f"Warning: No image found for {eid}")
            results[eid] = origin_text # 无图回退到纯文本
            
        time.sleep(0.2)
            
    with open(OUTPUT_JSON, 'w') as f: json.dump(results, f, indent=2)
    print(">>> AI 描述生成完毕。")

if __name__ == "__main__":
    main()