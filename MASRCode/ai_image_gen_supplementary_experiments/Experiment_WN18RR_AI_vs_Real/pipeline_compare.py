import os
import json
from openai import OpenAI
from wn18rr_path_finder import WN18RRPathFinder

API_KEY = ""
BASE_URL = "https://35.aigcbest.top/v1"
KG_PATH = "/data/user1/MyIdeaProject/datasets/WN18RR"
DECISION_MODEL = "gpt-4o"

class DecisionPipeline:
    def __init__(self, desc_file):
        print(f">>> [Pipeline] Loading Description File: {desc_file}")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # 加载描述文件 (AI 或 Real)
        with open(desc_file, 'r') as f: 
            self.enhanced_descriptions = json.load(f)
        
        # 复用路径搜索器
        self.path_finder = WN18RRPathFinder(KG_PATH)
        
        self.rels = list(self.path_finder.kg.relation2id.keys())
        self.rels = [r for r in self.rels if not r.endswith('_inv')]
        self.candidate_str = "\n".join([f"{i+1}. {r}" for i, r in enumerate(self.rels)])

    def format_paths(self, paths):
        if paths == "none" or not paths: return "No direct path found."
        lines = []
        for i, p in enumerate(paths, 1):
            t = []
            for j in range(len(p['entity_names']) - 1):
                t.append(f"({p['entity_names'][j]}) --[{p['relation_names'][j]}]-->")
            t.append(f"({p['entity_names'][-1]})")
            lines.append(f"Path {i}: {' '.join(t)}")
        return "\n".join(lines)

    def predict(self, h, t):
        h_str = str(h).strip()
        t_str = str(t).strip()

        # 1. 获取描述
        desc_h = self.enhanced_descriptions.get(h_str, "No description.")
        desc_t = self.enhanced_descriptions.get(t_str, "No description.")
        
        # 2. 语义重排序搜索路径
        raw_paths = self.path_finder.run(h_str, t_str, h_desc=desc_h, t_desc=desc_t)
        paths_text = self.format_paths(raw_paths)
        
        # 3. 轮询邻居
        neigh_h = "; ".join(self.path_finder.get_top_neighbors(h_str))
        neigh_t = "; ".join(self.path_finder.get_top_neighbors(t_str))
        
        # 4. 【核心】使用“稳健改良版” Prompt (完全一致)
        prompt = f"""
You are a Lexical Knowledge Expert. Select the EXACT relation ID from the list.

### 1. Context
**Head:** {h_str}
- Def: {desc_h}
- Neighbors: {neigh_h}

**Tail:** {t_str}
- Def: {desc_t}
- Neighbors: {neigh_t}

### 2. Structural Paths
{paths_text}

### 3. Critical Definitions & Pitfalls (Read Carefully)
**A. Morphology (Derivation vs Hypernym):**
- `_derivationally_related_form`: Use this when Head and Tail share the **same linguistic root** but have different parts of speech (e.g., "destroy" vs "destruction").
- **Warning**: Do NOT use `_hypernym` if they are derivationally related forms.

**B. Taxonomy (Member vs Hypernym):**
- `_member_meronym`: Use this for **Group Membership**. Specifically in biology, a Genus is a MEMBER of a Family.
- `_hypernym`: Use this for **"Is-A" Type**. A Cat is a Mammal.
- **Warning**: Do NOT confuse being a "member of a group" with being a "type of object".

**C. Hierarchy (Instance vs Hypernym):**
- `_instance_hypernym`: Use ONLY for **Named Entities** (e.g., "New York", "Einstein") connecting to their category.
- `_hypernym`: Use for general classes (e.g., "City", "Scientist").

### 4. Candidates
{self.candidate_str}

### Reasoning Task
1. Analyze the semantic relationship.
2. Check for shared roots (Derivation).
3. Check for Group/Family membership (Meronym).
4. Select the most precise ID.

**Output Format:**
**Reasoning:** <Brief analysis>
**Predicted Relation:** <The Relation ID>
"""
        try:
            res = self.client.chat.completions.create(
                model=DECISION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"