import torch, random
from collections import deque, defaultdict
from sentence_transformers import SentenceTransformer, util
from fb15k_loader import FB15kLoader

class FB15kPathFinder:
    def __init__(self, kg_path):
        print(">>> [PathFinder] Loading SBERT & KG...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.kg = FB15kLoader(kg_path)
        
        self.entity_adj = defaultdict(list)
        for h, r, t in self.kg.triples:
            self.entity_adj[h].append((r, t))

    def get_ent_name(self, eid): return self.kg.id2entity.get(eid, str(eid))
    def get_rel_name(self, rid): return self.kg.id2relation.get(rid, str(rid))

    def get_top_neighbors(self, entity_str, top_k=5):
        # FB15k ID 格式 /m/xxx
        eid = self.kg.entity2id.get(entity_str)
        if eid is None: return []
        
        neighbors = self.entity_adj.get(eid, [])
        if not neighbors: return []

        rels_group = defaultdict(list)
        for r, t in neighbors: rels_group[r].append(t)
        
        selected = []
        keys = list(rels_group.keys())
        random.shuffle(keys)
        
        while len(selected) < top_k and keys:
            for r in list(keys):
                if not rels_group[r]: keys.remove(r); continue
                t = random.choice(rels_group[r])
                rels_group[r].remove(t)
                selected.append(f"{self.get_rel_name(r)} -> {self.get_ent_name(t)}")
                if len(selected) >= top_k: break
        return selected

    def run(self, s_str, t_str, h_desc="", t_desc=""):
        s_id = self.kg.entity2id.get(s_str)
        t_id = self.kg.entity2id.get(t_str)
        if s_id is None or t_id is None: return "none"

        paths = self._bfs(s_id, t_id, 2)
        if len(paths) < 2:
            paths += [p for p in self._bfs(s_id, t_id, 3) if p not in paths]
        if not paths: return "none"

        # Semantic Rerank
        s_name = self.get_ent_name(s_id)
        t_name = self.get_ent_name(t_id)
        query = f"Path between {s_name} ({h_desc[:50]}) and {t_name} ({t_desc[:50]})"
        
        path_txts = []
        for p in paths:
            chain = []
            for i in range(len(p['rels'])):
                chain.append(f"{p['names'][i]} --[{p['r_names'][i]}]-->")
            chain.append(p['names'][-1])
            path_txts.append(" ".join(chain))
            
        try:
            q_emb = self.model.encode(query, convert_to_tensor=True)
            p_embs = self.model.encode(path_txts, convert_to_tensor=True)
            scores = util.cos_sim(q_emb, p_embs)[0]
            top_idx = torch.topk(scores, k=min(8, len(paths))).indices.tolist()
            return [paths[i] for i in top_idx]
        except: return paths[:8]

    def _bfs(self, s, t, max_hop):
        q = deque([(s, [s], [], 0)])
        paths = []
        visited = set()
        while q and len(paths) < 50:
            curr, nodes, rels, hop = q.popleft()
            if hop >= max_hop: continue
            if (curr, hop) in visited: continue
            visited.add((curr, hop))
            
            for r, n in self.entity_adj.get(curr, []):
                if n == t:
                    final_nodes = nodes + [n]
                    final_rels = rels + [r]
                    paths.append({
                        'names': [self.get_ent_name(x) for x in final_nodes],
                        'r_names': [self.get_rel_name(x) for x in final_rels],
                        'rels': final_rels
                    })
                elif n not in nodes:
                    q.append((n, nodes+[n], rels+[r], hop+1))
        return paths