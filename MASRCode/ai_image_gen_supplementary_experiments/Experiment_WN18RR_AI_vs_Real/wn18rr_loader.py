import os
import numpy as np

class WN18RRLoader:
    def __init__(self, path):
        """
        WN18RR 专用加载器
        path: ./datasets/WN18RR
        """
        self.data_path = path
        print(f">>> [WN18RR Loader] Loading data from {path} ...")

        # 1. 读取 ID 映射 (entity2id.txt, relation2id.txt)
        self.entity2id = self.load_dict(os.path.join(path, 'entity2id.txt'))
        self.relation2id = self.load_dict(os.path.join(path, 'relation2id.txt'))
        
        # 记录原始关系数量
        self.n_rel_original = len(self.relation2id)
        
        # 添加逆关系 (Inverse Relations) 用于双向搜索
        self.relation2id.update({k + '_inv': v + self.n_rel_original for k, v in self.relation2id.items()})
        
        # 构建反向映射
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        # 2. 读取图谱结构 (优先读取 facts.txt，如果不存在则读 train2id.txt)
        # 根据你的截图，facts.txt 应该包含训练三元组
        facts_file = os.path.join(path, 'facts.txt')
        if not os.path.exists(facts_file):
            facts_file = os.path.join(path, 'train2id.txt')
            
        if os.path.exists(facts_file):
            self.triples = self.load_triples(facts_file)
            print(f">>> [WN18RR Loader] Loaded {len(self.triples)} triples from {os.path.basename(facts_file)}")
            
            # 添加逆向三元组 (h, r, t) -> (t, r_inv, h)
            inv_triples = []
            for h, r, t in self.triples:
                inv_triples.append((t, r + self.n_rel_original, h))
            self.triples += inv_triples
            print(f">>> [WN18RR Loader] Graph constructed. Total edges (with inverse): {len(self.triples)}")
        else:
            raise FileNotFoundError(f"Cannot find facts.txt or train2id.txt in {path}")

    @staticmethod
    def load_dict(path):
        key2val = dict()
        if not os.path.exists(path): return key2val
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 兼容 OpenKE 格式：第一行可能是数量，如果只有一个数字则跳过
            if len(lines) > 0 and len(lines[0].strip().split()) == 1:
                lines = lines[1:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key, val = parts[0], parts[1]
                    key2val[key] = int(val)
        return key2val

    def load_triples(self, path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 同样跳过第一行数量统计
            if len(lines) > 0 and len(lines[0].strip().split()) == 1: 
                lines = lines[1:]
                    
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # facts.txt/train2id.txt 通常已经是 ID 格式: h_id t_id r_id 或 h r t
                    # 这里假设是 ID，直接转 int
                    try:
                        h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                        triples.append((h, r, t))
                    except ValueError:
                        # 如果不是 ID 而是名字，尝试转换 (容错)
                        h = self.entity2id.get(parts[0])
                        t = self.entity2id.get(parts[1])
                        r = self.relation2id.get(parts[2])
                        if h is not None and t is not None and r is not None:
                            triples.append((h, r, t))
        return triples