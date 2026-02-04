import os, json, time, base64, pandas as pd, random
from openai import OpenAI

API_KEY = ""
BASE_URL = "https://35.aigcbest.top/v1"
REAL_ROOT = "/data/user1/MyIdeaProject/FB15k-images"
DESC_FILE = "/data/user1/MyIdeaProject/datasets/fb15k-237/get_neighbor/FB15k_mid2description.txt"
INPUT_CSV = "sampled_50_triples.csv"
OUTPUT_JSON = "desc_Real.json"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_def():
    d = {}
    with open(DESC_FILE,'r') as f:
        for l in f:
            p = l.strip().split('\t')
            if len(p)>=2: d[p[0]] = p[1]
    return d

def get_b64(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

def find_folder(root, eid):
    # /m/01 -> m.01
    folder = eid.lstrip('/').replace('/', '.')
    path = os.path.join(root, folder)
    if os.path.isdir(path): return path
    return None

def main():
    print(">>> [Real Group] 生成描述...")
    df = pd.read_csv(INPUT_CSV, dtype=str)
    ents = set(df['Head'].tolist() + df['Tail'].tolist())
    desc_map = load_def()
    
    res = {}
    for i, eid in enumerate(ents):
        origin = desc_map.get(eid, "")
        target = find_folder(REAL_ROOT, eid)
        imgs = []
        if target:
            fs = [x for x in os.listdir(target) if x.endswith('.jpg')]
            if fs: imgs = [os.path.join(target, x) for x in random.sample(fs, min(5, len(fs)))]
            
        if imgs:
            print(f"[{i+1}/{len(ents)}] Analyzing {eid} ({len(imgs)} imgs)...")
            content = [{"type":"text","text":f"Definition: {origin}\nAnalyze: Category, Appearance."}]
            for p in imgs:
                b = get_b64(p)
                if b: content.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}","detail":"low"}})
            try:
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":content}])
                res[eid] = f"{origin}\n\n[Visual]:\n{r.choices[0].message.content}"
            except: res[eid] = origin
        else: res[eid] = origin
        
        if len(res)%5==0:
            with open(OUTPUT_JSON,'w') as f: json.dump(res,f,indent=2)

    with open(OUTPUT_JSON,'w') as f: json.dump(res,f,indent=2)

if __name__ == "__main__": main()