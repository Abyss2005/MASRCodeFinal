import pandas as pd
import os, json, time, requests, base64

# ================= 配置 (已修改为本地路径) =================
# 考题文件 (当前目录下)
INPUT_CSV = "sampled_50_triples.csv"
# 定义文件 (保持绝对路径读取)
DESC_FILE = "/data/user1/MyIdeaProject/datasets/WN18RR/wordnet-definitions.txt"

# 【核心修改】图片保存到当前实验文件夹下的 AI_images
OUTPUT_DIR = "./AI_images"

# 豆包 API (保持不变)
API_KEY = ""
LLM_ENDPOINT_ID = "ep-20260105104049-4nzh2"
IMG_ENDPOINT_ID = "ep-20260105103828-shfhj"
LLM_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
IMG_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
# =======================================================

def load_definitions():
    d = {}
    with open(DESC_FILE, 'r') as f:
        for l in f:
            p = l.strip().split('\t')
            if len(p)>=2: d[p[0]] = p[1]
    return d

def call_llm_optimize_prompt(entity_name, raw_desc):
    system_instruction = (
        "You are an expert scientific illustrator. "
        "Create a prompt for a clear, educational image that represents the given concept. "
        "Focus on the category, components, and distinguishing visual features to avoid ambiguity. "
        "Format: 'An encyclopedic illustration of [Concept], [Key Features], [Style: Macro photography/Clean background]'"
    )
    safe_desc = raw_desc[:1000]
    payload = {
        "model": LLM_ENDPOINT_ID,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Entity Name: {entity_name}\nRaw Description: {safe_desc}"}
        ],
        "temperature": 0.7 
    }
    try:
        response = requests.post(LLM_API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
    except: pass
    return None

def call_img_api(prompt, save_path):
    payload = {
        "model": IMG_ENDPOINT_ID, "prompt": prompt, "size": "2048x2048", 
        "optimize_prompt_options": {"mode": "standard"}, "response_format": "b64_json"
    }
    try:
        response = requests.post(IMG_API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload, timeout=120)
        if response.status_code == 200:
            b64_data = response.json()["data"][0].get("b64_json")
            if b64_data:
                with open(save_path, "wb") as f: f.write(base64.b64decode(b64_data))
                print(f"    ✅ Saved: {os.path.basename(save_path)}")
                return True
        print(f"    ❌ Gen Failed: {response.text}")
    except Exception as e: print(f"    ❌ Gen Error: {e}")
    return False

def main():
    print(f">>> [Step 0.5] 补全 AI 图片到本地目录: {OUTPUT_DIR} ...")
    if not os.path.exists(INPUT_CSV): return
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR) # 自动创建目录
    
    df = pd.read_csv(INPUT_CSV, dtype=str)
    target_entities = set(df['Head'].tolist() + df['Tail'].tolist())
    desc_map = load_definitions()
    
    needed = [eid for eid in target_entities if not os.path.exists(os.path.join(OUTPUT_DIR, f"{eid}.jpg"))]
    print(f">>> 需生成数量: {len(needed)}")
    
    for i, eid in enumerate(needed):
        desc = desc_map.get(eid, "concept")
        print(f"[{i+1}/{len(needed)}] Processing: {eid}")
        
        final_prompt = call_llm_optimize_prompt(eid, desc)
        if not final_prompt: final_prompt = f"An encyclopedic illustration of {eid}. {desc[:300]}"
        
        if call_img_api(final_prompt, os.path.join(OUTPUT_DIR, f"{eid}.jpg")):
            time.sleep(1)

if __name__ == "__main__":
    main()