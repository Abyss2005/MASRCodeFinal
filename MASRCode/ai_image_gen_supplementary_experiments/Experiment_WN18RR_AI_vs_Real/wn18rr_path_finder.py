import torch
import random
from collections import deque, defaultdict
from sentence_transformers import SentenceTransformer, util
from wn18rr_loader import WN18RRLoader  # <--- 导入新的 Loader

class WN18RRPathFinder:
    def __init__(self, kg_path):
        """
        MASR 框架专用路径搜索器
        """
        # 1. 加载 SBERT (用于语义路径重排序)
        print(">>> [PathFinder] Loading SBERT model (all-MiniLM-L6-v2)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # 2. 加载数据
        self.kg = WN18RRLoader(kg_path)
        
        # 3. 构建邻接表
        print(">>> [PathFinder] Building Adjacency List...")
        self.entity_adj = defaultdict(list)
        for h, r, t in self.kg.triples:
            self.entity_adj[h].append((r, t))
        
        print(f">>> [PathFinder] Ready. Nodes: {len(self.entity_adj)}")

    def get_ent_name(self, ent_id):
        return self.kg.id2entity.get(ent_id, str(ent_id))

    def get_rel_name(self, rel_id):
        return self.kg.id2relation.get(rel_id, str(rel_id))

    # === 3.2.1.1 轮询邻居采样 (Round-Robin) ===
    def get_top_neighbors(self, entity_input, top_k=5):
        # 统一转为 int ID
        if isinstance(entity_input, str) and entity_input.isdigit():
            eid = int(entity_input)
        elif isinstance(entity_input, str):
            eid = self.kg.entity2id.get(entity_input)
        else:
            eid = entity_input

        neighbors = self.entity_adj.get(eid, [])
        if not neighbors: return []

        # 按关系类型分组
        rels_group = defaultdict(list)
        for r_id, t_id in neighbors:
            rels_group[r_id].append(t_id)
        
        selected = []
        unique_rels = list(rels_group.keys())
        random.shuffle(unique_rels) 
        
        # 轮询抽取
        while len(selected) < top_k and len(unique_rels) > 0:
            for r_id in list(unique_rels):
                if not rels_group[r_id]:
                    unique_rels.remove(r_id)
                    continue
                
                t_id = random.choice(rels_group[r_id])
                rels_group[r_id].remove(t_id)
                
                r_name = self.get_rel_name(r_id)
                t_name = self.get_ent_name(t_id)
                selected.append(f"{r_name} -> {t_name}")
                
                if len(selected) >= top_k: break
        
        return selected

    # === 3.2.1.2 语义路径重排序 (Semantic Reranking) ===
    def run(self, s_input, t_input, h_desc="", t_desc=""):
        # ID 转换
        s_id = int(s_input) if str(s_input).isdigit() else self.kg.entity2id.get(s_input)
        t_id = int(t_input) if str(t_input).isdigit() else self.kg.entity2id.get(t_input)

        if s_id is None or t_id is None: return "none"

        # 1. 自适应 BFS (优先 2-hop，不够找 3-hop)
        paths = self._bfs(s_id, t_id, max_hop=2)
        if len(paths) < 2:
            paths_3hop = self._bfs(s_id, t_id, max_hop=3)
            existing_sigs = set([tuple(p['entity_ids']) for p in paths])
            for p in paths_3hop:
                sig = tuple(p['entity_ids'])
                if sig not in existing_sigs:
                    paths.append(p)
        
        if not paths: return "none"

        # 2. 构造 Query 和 Path Text
        s_name = self.get_ent_name(s_id)
        t_name = self.get_ent_name(t_id)
        query = f"Relation path between {s_name} ({h_desc[:80]}) and {t_name} ({t_desc[:80]})"
        
        path_texts = []
        valid_paths = []
        for p in paths:
            try:
                chain = []
                e_names = p['entity_names']
                r_names = p['relation_names']
                for i in range(len(r_names)):
                    chain.append(f"{e_names[i]} --[{r_names[i]}]-->")
                chain.append(e_names[-1])
                path_texts.append(" ".join(chain))
                valid_paths.append(p)
            except:
                continue

        if not valid_paths: return "none"

        # 3. SBERT 相似度计算
        try:
            query_emb = self.model.encode(query, convert_to_tensor=True)
            path_embs = self.model.encode(path_texts, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, path_embs)[0]
            
            # 取 Top-8
            top_k = min(8, len(valid_paths))
            top_indices = torch.topk(scores, k=top_k).indices.tolist()
            final_paths = [valid_paths[i] for i in top_indices]
        except Exception as e:
            print(f"[Warning] SBERT failed: {e}. Using random.")
            final_paths = valid_paths[:8]
            
        return final_paths

    def _bfs(self, s_id, t_id, max_hop):
        queue = deque([(s_id, [s_id], [], 0)])
        paths = []
        max_paths_limit = 200 
        visited = set()
        
        while queue and len(paths) < max_paths_limit:
            curr, nodes, rels, hop = queue.popleft()
            if hop >= max_hop: continue
            if (curr, hop) in visited: continue
            visited.add((curr, hop))
            
            for r, neighbor in self.entity_adj.get(curr, []):
                if neighbor == t_id:
                    final_nodes = nodes + [t_id]
                    final_rels = rels + [r]
                    paths.append({
                        "hop": hop + 1,
                        "entity_ids": final_nodes,
                        "entity_names": [self.get_ent_name(e) for e in final_nodes],
                        "relation_names": [self.get_rel_name(rel) for rel in final_rels]
                    })
                    if len(paths) >= max_paths_limit: break
                elif neighbor not in nodes and len(nodes) < max_hop + 1:
                    queue.append((neighbor, nodes + [neighbor], rels + [r], hop + 1))
        return paths