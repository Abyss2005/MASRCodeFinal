import torch
import random
from collections import deque, defaultdict
from sentence_transformers import SentenceTransformer, util


class MASR_Searcher:
    def __init__(self, loader, sbert_model_name='all-MiniLM-L6-v2'):
        self.loader = loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(sbert_model_name, device=self.device)

        # 构建邻接表
        self.adj = defaultdict(list)
        for h, r, t in self.loader.all_triples:
            self.adj[h].append((r, t))

    def get_top_neighbors(self, ent_id, top_k=5):
        """轮询邻居采样 (Round-Robin)：确保邻居的关系多样性"""
        neighbors = self.adj.get(ent_id, [])
        if not neighbors: return ""

        rels_group = defaultdict(list)
        for r, t in neighbors: rels_group[r].append(t)

        selected = []
        unique_rels = list(rels_group.keys())
        random.shuffle(unique_rels)

        while len(selected) < top_k and unique_rels:
            for r in list(unique_rels):
                if not rels_group[r]:
                    unique_rels.remove(r);
                    continue
                t = random.choice(rels_group[r])
                rels_group[r].remove(t)
                r_n = self.loader.id2relation.get(r, str(r))
                t_n = self.loader.id2entity.get(t, str(t))
                selected.append(f"{r_n} -> {t_n}")
                if len(selected) >= top_k: break
        return "; ".join(selected)

    def run_path_search(self, s_id, t_id, h_desc="", t_desc=""):
        """自适应路径搜索 + SBERT 重排序"""
        if s_id is None or t_id is None: return "none"

        # 1. 自适应 BFS (2-hop -> 3-hop)
        paths = self._bfs(s_id, t_id, max_hop=2)
        if len(paths) < 2:
            paths += self._bfs(s_id, t_id, max_hop=3)

        if not paths: return "none"

        # 2. 构造 Query 和路径文本进行语义对齐
        s_name = self.loader.id2entity[s_id]
        t_name = self.loader.id2entity[t_id]
        query = f"Relation path between {s_name} ({h_desc[:50]}) and {t_name} ({t_desc[:50]})"

        path_texts = []
        for p in paths:
            chain = []
            for i in range(len(p['relation_names'])):
                chain.append(f"({p['entity_names'][i]})--[{p['relation_names'][i]}]-->")
            chain.append(f"({p['entity_names'][-1]})")
            path_texts.append(" ".join(chain))

        # 3. SBERT 计算相似度并取 Top-8
        q_emb = self.model.encode(query, convert_to_tensor=True)
        p_emb = self.model.encode(path_texts, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, p_emb)[0]

        top_k = min(8, len(paths))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [paths[i] for i in top_indices]

    def _bfs(self, s, t, max_hop):
        queue = deque([(s, [s], [], 0)])
        paths = []
        visited = set()
        while queue and len(paths) < 100:
            curr, nodes, rels, hop = queue.popleft()
            if hop >= max_hop: continue
            for r, neighbor in self.adj.get(curr, []):
                if neighbor == t:
                    final_nodes = nodes + [t]
                    final_rels = rels + [r]
                    paths.append({
                        "entity_names": [self.loader.id2entity[e] for e in final_nodes],
                        "relation_names": [self.loader.id2relation[rl] for rl in final_rels]
                    })
                elif neighbor not in nodes:
                    queue.append((neighbor, nodes + [neighbor], rels + [r], hop + 1))
        return paths