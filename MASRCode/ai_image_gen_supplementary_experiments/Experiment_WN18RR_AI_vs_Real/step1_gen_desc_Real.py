import os, json, time, base64, pandas as pd, random
from openai import OpenAI

# ================= 配置 =================
API_KEY = ""
BASE_URL = "https://35.aigcbest.top/v1"

# 真实图片根目录
REAL_IMAGE_ROOT = "/data/user1/MyIdeaProject/ai_image_generation/real_images_full/wn18rr"
# 原始定义文件 (新增：用于获取文本定义)
DESC_FILE = "/data/user1/MyIdeaProject/datasets/WN18RR/wordnet-definitions.txt"
# 考题文件
INPUT_CSV = "sampled_50_triples.csv"
# 输出结果
OUTPUT_JSON = "desc_Real.json"
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

def find_image_folder(root, eid):
    """智能查找文件夹：兼容纯数字ID和带n前缀的ID"""
    path1 = os.path.join(root, eid)
    if os.path.isdir(path1): return path1
    path2 = os.path.join(root, "n" + eid)
    if os.path.isdir(path2): return path2
    return None

def main():
    print(">>> [Group Real] 生成视觉描述 (5图采样 + 文本融合 + 自动保存)...")
    
    if not os.path.exists(REAL_IMAGE_ROOT):
        print(f"❌ 错误：找不到真实图片目录 {REAL_IMAGE_ROOT}")
        return

    # 1. 加载数据
    df = pd.read_csv(INPUT_CSV, dtype=str)
    entities = set(df['Head'].tolist() + df['Tail'].tolist())
    # 排序以保证处理顺序一致
    sorted_entities = sorted(list(entities))
    
    # 加载原始定义
    desc_map = load_definitions()
    
    print(f">>> 待处理实体数: {len(sorted_entities)}")
    
    # 断点续传支持
    results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r') as f: results = json.load(f)
            print(f">>> 已加载历史进度: {len(results)} 条")
        except: pass
    
    found_count = 0
    
    for i, eid in enumerate(sorted_entities):
        eid = str(eid).strip()
        
        # 如果已存在则跳过
        if eid in results:
            continue

        # 获取原始文本定义
        origin_text = desc_map.get(eid, f"Concept {eid}")
        
        # 2. 找图
        target_dir = find_image_folder(REAL_IMAGE_ROOT, eid)
        imgs = []
        
        if target_dir:
            files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            if files:
                # 随机采样最多 5 张
                sample_num = min(5, len(files))
                imgs = [os.path.join(target_dir, f) for f in random.sample(files, sample_num)]
                found_count += 1
                print(f"[{i+1}/{len(sorted_entities)}] Found {len(imgs)} images for {eid}")
            else:
                print(f"[{i+1}/{len(sorted_entities)}] Folder empty: {eid}")
        else:
            print(f"[{i+1}/{len(sorted_entities)}] No folder found: {eid}")
        
        # 3. 调用视觉模型
        if imgs:
            # Prompt 中加入原始定义，帮助模型更好理解模糊的真实图片
            prompt = f"Concept Definition: {origin_text}\nTask: Analyze these real-world images based on the definition. Describe Category, Components, and Appearance."
            
            content = [{"type": "text", "text": prompt}]
            
            # 循环上传最多 5 张
            for p in imgs:
                b = get_b64(p)
                if b: 
                    # 真实图片模糊，用 low 模式节省 token
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "low"}})
            
            try:
                res = client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[{"role": "user", "content": content}],
                    max_tokens=300
                )
                # 【核心】：拼接原始文本 + 视觉描述
                results[eid] = f"{origin_text}\n\n[Real Image Analysis ({len(imgs)} imgs)]:\n{res.choices[0].message.content}"
            except Exception as e: 
                print(f"    API Error: {e}")
                results[eid] = origin_text # 出错回退到纯文本
        else:
            # 没图，回退到纯文本
            results[eid] = origin_text
            
        time.sleep(0.2)

        # 自动保存 (每5条)
        if len(results) % 5 == 0:
            with open(OUTPUT_JSON, 'w') as f: json.dump(results, f, indent=2)
            print(f"    💾 Auto-saved {len(results)} items...")
            
    # 最终保存
    with open(OUTPUT_JSON, 'w') as f: json.dump(results, f, indent=2)
    print(f"\n>>> 真实组描述生成完毕。共找到图片: {found_count} 个实体。")

if __name__ == "__main__":
    main()