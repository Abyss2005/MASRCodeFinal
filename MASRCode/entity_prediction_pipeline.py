import os
import json
import random
import time
import re
from collections import defaultdict, deque
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# =========================================================
# 1. 安全配置与环境初始化
# =========================================================
# 建议在运行前设置：export OPENAI_API_KEY='sk-xxx'
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 强制开启 SBERT 离线模式（防止 GitHub 用户因网络问题报错）
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 模型与路径配置
STAGE_2A_MODEL = "gpt-4o-mini" # 用于 50 选 10
STAGE_2B_MODEL = "gpt-4o"      # 用于最终决策
PROJECT_ROOT = os.getenv("MASR_PATH", "./")

# 默认文件定义
INPUT_FILE = os.path.join(PROJECT_ROOT, "candidates_top200.json")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "summary_library_full_8583.json")
MID2NAME_FILE = os.path.join(PROJECT_ROOT, "FB15k_mid2name.txt")
TRAIN_FILE = os.path.join(PROJECT_ROOT, "train.txt")
OUTPUT_FILE = "final_rerank_results.json"

# 超参数
STAGE2A_TOP_K = 10
SC_ROUNDS = 3
TOP_K_PATHS = 4

# =========================================================
# 2. 核心算法组件
# =========================================================

def rel_to_text(rel_str):
    """规范化关系文本"""
    return rel_str.strip('/').replace('/', ' ').replace('_', ' ').strip()

def collect_paths(head, tail, graph, max_hop=3):
    """BFS 路径检索"""
    if head == tail: return []
    paths, queue, visited = [], deque([(head, [], 0)]), {head}
    while queue and len(paths) < 30:
        cur, edges, hop = queue.popleft()
        if hop >= max_hop: continue
        for r, nxt in graph.get(cur, []):
            if nxt == tail:
                paths.append(edges + [(r, nxt)])
            elif nxt not in visited and hop + 1 < max_hop:
                visited.add(nxt)
                queue.append((nxt, edges + [(r, nxt)], hop + 1))
    return paths

def get_best_paths_sbert(head_id, tail_id, rel_text, h_desc, t_desc, graph, mid2name, sbert):
    """利用 SBERT 进行路径语义重排"""
    query_text = f"{h_desc} {t_desc}".strip()
    raw_paths = collect_paths(head_id, tail_id, graph)
    
    if not raw_paths: return "No structural path found."
    
    path_texts = []
    for p_edges in raw_paths:
        t = mid2name.get(head_id, head_id)
        for r, nxt in p_edges:
            t += f" --({rel_to_text(r)})--> {mid2name.get(nxt, nxt)}"
        path_texts.append(t)

    query_emb = sbert.encode(query_text, convert_to_tensor=True)
    path_embs = sbert.encode(path_texts, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, path_embs)[0]
    
    scored = sorted(zip(sims.tolist(), path_texts), key=lambda x: x[0], reverse=True)
    return "\n    ".join([f"[sim={s:.3f}] {txt}" for s, txt in scored[:TOP_K_PATHS]])

def call_llm(prompt, model, is_json=False):
    """封装 API 调用"""
    for _ in range(3):
        try:
            params = {
                "model": model,
                "messages": [{"role": "system", "content": "You are a KG expert."},
                             {"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            if is_json: params["response_format"] = {"type": "json_object"}
            
            res = client.chat.completions.create(**params)
            return res.choices[0].message.content
        except Exception:
            time.sleep(2)
    return None

# =========================================================
# 3. 级联重排流程 (Stage 2A & 2B)
# =========================================================

def run_stage_2a(head_id, rel_name, h_desc, candidates, mid2name, summary_lib):
    """50 -> 10 粗略过滤"""
    shuffled = candidates.copy()
    random.shuffle(shuffled)
    
    block_lines = []
    for i, cid in enumerate(shuffled, 1):
        name = mid2name.get(cid, cid)
        prof = summary_lib.get(cid, "N/A")
        block_lines.append(f"[{i}] {name} ({cid}): {prof}")
    
    prompt = f"""Task: Select TOP-10 plausible candidates for ({mid2name.get(head_id, head_id)}, {rel_name}, ?).
Head Info: {h_desc}
Candidates:
{"\n".join(block_lines)}
Output ONLY JSON: {{"top_indices": [idx1, idx2, ...]}}"""

    res = call_llm(prompt, STAGE_2A_MODEL, is_json=True)
    try:
        idxs = json.loads(res).get("top_indices", [])
        return [shuffled[i-1] for i in idxs if 1 <= i <= len(shuffled)][:10]
    except:
        return shuffled[:10]

def run_stage_2b_sc(head_id, rel_name, h_desc, top10, graph, mid2name, summary_lib, sbert):
    """10 -> 1 证据驱动决策 + 自洽性投票"""
    votes = defaultdict(int)
    # 预准备证据块
    blocks = {}
    for cid in top10:
        p_evid = get_best_paths_sbert(head_id, cid, rel_name, h_desc, summary_lib.get(cid, ""), graph, mid2name, sbert)
        blocks[cid] = f"Name: {mid2name.get(cid, cid)}\nProfile: {summary_lib.get(cid, '')}\nEvidence: {p_evid}"

    for _ in range(SC_ROUNDS):
        current_cands = top10.copy()
        random.shuffle(current_cands)
        cand_str = "\n\n".join([f"[{i+1}] {blocks[cid]}" for i, cid in enumerate(current_cands)])
        
        prompt = f"""Select the single correct tail for ({mid2name.get(head_id, head_id)}, {rel_name}, ?).
Head Context: {h_desc}
Candidates with Structural Evidence:
{cand_str}
Reason step by step, then output: Final Answer: [index]"""

        output = call_llm(prompt, STAGE_2B_MODEL)
        match = re.search(r"Final Answer:\s*\[?(\d+)\]?", output or "")
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= len(current_cands): votes[current_cands[idx-1]] += 1
            
    return max(votes.items(), key=lambda x: x[1])[0] if votes else top10[0]

# =========================================================
# 4. 主程序：全量评测
# =========================================================

def main():
    # 数据加载
    with open(SUMMARY_JSON, "r") as f: summary_lib = json.load(f)
    with open(INPUT_FILE, "r") as f: all_q = json.load(f)
    mid2name = {}
    if os.path.exists(MID2NAME_FILE):
        for line in open(MID2NAME_FILE, "r"):
            p = line.strip().split("\t")
            if len(p)>=2: mid2name[p[0]] = p[1]
            
    graph = defaultdict(list)
    for line in open(TRAIN_FILE, "r"):
        p = line.strip().split()
        if len(p)==3: graph[p[0]].append((p[1], p[2]))
    
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 筛选有效样本 (Hits@50=True)
    targets = [it for item in all_q if (it:=item).get('recall_status',{}).get('in_top50')]
    results, hit = [], 0
    
    for item in tqdm(targets, desc="MASR Entity Prediction"):
        h, r, true_t = item['query']['head'], item['query']['relation'], item['true_tail']
        rel_t = rel_to_text(r)
        h_desc = summary_lib.get(h, "N/A")
        
        # 级联推理
        t10 = run_stage_2a(h, rel_t, h_desc, item['candidates_top50'], mid2name, summary_lib)
        pred = run_stage_2b_sc(h, rel_t, h_desc, t10, graph, mid2name, summary_lib, sbert)
        
        is_correct = (pred == true_t)
        if is_correct: hit += 1
        
        results.append({"head":h, "rel":r, "true":true_t, "pred":pred, "correct":is_correct})
        
        # 原子保存
        with open(OUTPUT_FILE + ".tmp", "w") as f: json.dump(results, f, indent=2)
        os.replace(OUTPUT_FILE + ".tmp", OUTPUT_FILE)

    print(f"Final Accuracy on Winnable Set: {hit/len(targets)*100:.2f}%")

if __name__ == "__main__":
    main()