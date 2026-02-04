import requests, json, os, base64, time

# ================= 豆包 API 配置 =================
API_KEY = ""
LLM_ENDPOINT_ID = "ep-20260105104049-4nzh2"
IMG_ENDPOINT_ID = "ep-20260105103828-shfhj"
LLM_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
IMG_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

MISSING_FILE = "missing_tails_to_generate.json"
OUTPUT_DIR = "/data/user1/MyIdeaProject/ai_image_generation/images/wn18rr"
# ===============================================

def call_llm(name, desc):
    sys_prompt = "You are a scientific illustrator. Create a clear, encyclopedic image prompt based on the concept."
    try:
        res = requests.post(LLM_API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json={
            "model": LLM_ENDPOINT_ID,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Entity: {name}\nDesc: {desc[:500]}"}]
        }, timeout=30)
        return res.json()['choices'][0]['message']['content'] if res.status_code == 200 else None
    except: return None

def call_img(prompt, path):
    try:
        res = requests.post(IMG_API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json={
            "model": IMG_ENDPOINT_ID, "prompt": prompt, "size": "2048x2048", "response_format": "b64_json"
        }, timeout=120)
        if res.status_code == 200:
            b64 = res.json()["data"][0]["b64_json"]
            with open(path, "wb") as f: f.write(base64.b64decode(b64))
            print(f"    ✅ Saved: {os.path.basename(path)}")
            return True
        print(f"    ❌ Failed: {res.text}")
    except Exception as e: print(f"    ❌ Error: {e}")
    return False

def main():
    if not os.path.exists(MISSING_FILE): return
    with open(MISSING_FILE, 'r') as f: items = json.load(f)
    print(f">>> 开始补全 {len(items)} 张图片...")
    
    for i, item in enumerate(items):
        eid, desc = str(item['id']).strip(), item['desc']
        path = os.path.join(OUTPUT_DIR, f"{eid}.jpg")
        if os.path.exists(path): continue
        
        print(f"[{i+1}/{len(items)}] Gen: {eid}")
        prompt = call_llm(eid, desc) or f"A photo of {eid}"
        if call_img(prompt, path): time.sleep(1)

if __name__ == "__main__":
    main()