import os, json
from openai import OpenAI
from fb15k_path_finder import FB15kPathFinder

# ================= 配置 =================
API_KEY = ""
BASE_URL = "https://35.aigcbest.top/v1"
KG_PATH = "/data/user1/MyIdeaProject/datasets/fb15k-237/get_neighbor"
DECISION_MODEL = "gpt-4o"
# =======================================

class DecisionPipeline:
    def __init__(self, desc_file):
        print(f">>> [Pipeline] Loading Description: {desc_file}")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        with open(desc_file, 'r') as f: 
            self.enhanced_descriptions = json.load(f)
        
        # 加载路径搜索器 (包含 SBERT)
        self.finder = FB15kPathFinder(KG_PATH)
        
        # 准备候选关系集 (过滤掉 _inv)
        self.rels = [r for r in self.finder.kg.relation2id.keys() if not r.endswith('_inv')]
        self.cand_str = "\n".join([f"- {r}" for r in self.rels])
        print(f">>> Loaded {len(self.rels)} candidate relations.")

    def format_paths(self, paths):
        if not paths or paths == "none": return "No direct path found."
        # FB15k 的路径格式化
        lines = []
        for i, p in enumerate(paths):
            chain = []
            for j in range(len(p['names']) - 1):
                chain.append(f"({p['names'][j]}) --[{p['r_names'][j]}]-->")
            chain.append(f"({p['names'][-1]})")
            lines.append(f"Path {i+1}: {' '.join(chain)}")
        return "\n".join(lines)

    def predict(self, h, t):
        # 1. 获取描述
        # FB15k ID 是 /m/xxx，JSON里可能存的是这个，也可能要注意格式
        dh = self.enhanced_descriptions.get(str(h), "No description")
        dt = self.enhanced_descriptions.get(str(t), "No description")
        
        # 2. 搜索路径 (SBERT Reranked)
        raw_paths = self.finder.run(h, t, h_desc=dh, t_desc=dt)
        paths_text = self.format_paths(raw_paths)
        
        # 3. 搜索邻居 (Round-Robin)
        nh = "; ".join(self.finder.get_top_neighbors(h))
        nt = "; ".join(self.finder.get_top_neighbors(t))
        
        # 4. 【SOTA Prompt】完全复刻你给我的那个版本
        prompt = f"""
You are a Knowledge Graph expert. Your goal is to select the **CORRECT relation ID** from the candidates to connect Head and Tail.

### 1. Context
**Head:** {h}
- Info: {dh}
- Neighbors: {nh}

**Tail:** {t}
- Info: {dt}
- Neighbors: {nt}

### 2. Structural Paths
{paths_text}

### 3. Learning form Examples (Few-Shot)
**Example 1 (Complex Schema):**
- Input: Head=Actor A, Tail=Movie B.
- Logic: "Acting in a movie" in FB15k-237 is usually `/film/actor/film` OR `/film/performance/film`.
- Correct ID: `/film/actor/film./film/performance/film`

**Example 2 (Inverse Relation):**
- Input: Head=English (Language), Tail=USA (Country).
- Logic: Language spoken in Country. Subject is Language.
- Correct ID: `/language/human_language/countries_spoken_in` (NOT `/location/country/languages_spoken`)

**Example 3 (Bias Correction):**
- Input: Head=Person A, Tail=City B.
- Logic: Person A lived in City B, but wasn't born there.
- Correct ID: `/people/person/places_lived./people/place_lived/location` (NOT `/people/person/place_of_birth`)

**Example 4 (Award Distinction):**
- Input: Head=Film A, Tail=Award B.
- Logic: If Tail is the award itself (e.g., Oscars), use `.../award`. If Tail is a category (e.g., Best Actor), use `.../nominated_for`.
- Correct ID: `/award/award_nominee/award_nominations./award/award_nomination/award`

### 4. Candidate Relations (Select ONE)
{self.cand_str}

### Reasoning Task
1. Analyze the semantic type of Head and Tail.
2. Check the direction (Head -> Tail).
3. Select the most precise ID from the list.

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