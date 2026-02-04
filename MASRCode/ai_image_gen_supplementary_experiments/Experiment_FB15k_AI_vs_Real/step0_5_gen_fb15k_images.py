import pandas as pd
import os, json, time, requests, base64

# ================= 配置 =================
INPUT_CSV = "sampled_50_triples.csv"
DESC_FILE = "/data/user1/MyIdeaProject/datasets/fb15k-237/get_neighbor/FB15k_mid2description.txt"
OUTPUT_DIR = "./AI_images"

API_KEY = ""
LLM_ENDPOINT_ID = "ep-20260105104049-4nzh2"
IMG_ENDPOINT_ID = "ep-20260105103828-shfhj"
LLM_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
IMG_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
# =======================================

def load_definitions():
    d = {}
    if not os.path.exists(DESC_FILE): return d
    with open(DESC_FILE, 'r') as f:
        for l in f:
            p = l.strip().split('\t')
            if len(p)>=2: d[p[0]] = p[1]
    return d

def call_llm_optimize_prompt(entity_name, raw_desc):
    system_instruction = "You are a visual prompt engineer. Create a photorealistic image prompt."
    safe_desc = raw_desc[:1000]
    payload = {
        "model": LLM_ENDPOINT_ID,
        "messages": [{"role":"system","content":system_instruction},{"role":"user","content":f"{entity_name}: {safe_desc}"}],
        "temperature": 0.7 
    }
    try:
        response = requests.post(LLM_API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"    ⚠️ LLM API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"    ⚠️ LLM Exception: {e}")
    return None

def call_img_api(prompt, save_path):
    payload = {"model": IMG_ENDPOINT_ID, "prompt": prompt, "size": "2048x2048", "response_format": "b64_json"}
    try:
        response = requests.post(IMG_API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload, timeout=120)
        if response.status_code == 200:
            res_json = response.json()
            if "data" in res_json:
                with open(save_path, "wb") as f: 
                    f.write(base64.b64decode(res_json["data"][0]["b64_json"]))
                print(f"    ✅ Saved: {os.path.basename(save_path)}")
                return True
            else:
                print(f"    ❌ Response format error: {res_json}")
        else:
            print(f"    ❌ Image API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"    ❌ Image Network Exception: {e}")
    return False

def main():
    print(">>> [Step 0.5] 补全 FB15k AI 图片 (Debug Mode)...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    df = pd.read_csv(INPUT_CSV, dtype=str)
    ents = set(df['Head'].tolist() + df['Tail'].tolist())
    desc_map = load_definitions()
    
    needed = [e for e in ents if not os.path.exists(os.path.join(OUTPUT_DIR, f"{e.replace('/','_')}.jpg"))]
    print(f">>> 需生成: {len(needed)}")
    
    for i, eid in enumerate(needed):
        print(f"[{i+1}/{len(needed)}] Processing: {eid}")
        desc = desc_map.get(eid, f"Entity {eid}")
        
        # 1. 优化提示词
        prompt = call_llm_optimize_prompt(eid, desc)
        if not prompt:
            print("    ⚠️ LLM failed, using fallback prompt.")
            prompt = f"A photorealistic image of {eid}"
            
        # 2. 生成图片
        safe_name = eid.replace('/', '_')
        if call_img_api(prompt, os.path.join(OUTPUT_DIR, f"{safe_name}.jpg")):
            time.sleep(1) # 成功才休息
        else:
            print("    ❌ Failed to generate. Continuing...")

if __name__ == "__main__":
    main()