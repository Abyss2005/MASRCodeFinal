import os

class FB15kLoader:
    def __init__(self, path):
        self.data_path = path
        print(f">>> [FB15k Loader] Loading from {path} ...")

        # 1. 读取字典 (兼容不同格式)
        self.entity2id = self.load_dict(os.path.join(path, 'entity2id.txt'))
        self.relation2id = self.load_dict(os.path.join(path, 'relation2id.txt'))
        
        # 逆关系处理
        self.n_rel_original = len(self.relation2id)
        self.relation2id.update({k + '_inv': v + self.n_rel_original for k, v in self.relation2id.items()})
        
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        # 2. 读取训练集构建图谱
        self.triples = self.load_triples(os.path.join(path, 'train.txt'))
        
        # 添加逆向边
        inv_triples = []
        for h, r, t in self.triples:
            inv_triples.append((t, r + self.n_rel_original, h))
        self.triples += inv_triples
        print(f">>> Graph constructed. Edges: {len(self.triples)}")

    @staticmethod
    def load_dict(path):
        """
        智能读取字典，自动识别 ID 在哪一列
        """
        d = {}
        if not os.path.exists(path):
            print(f"⚠️ Warning: File not found {path}")
            return d
            
        with open(path, 'r') as f:
            lines = f.readlines()
            # 跳过第一行数量说明
            if len(lines) > 0 and len(lines[0].strip().split()) == 1: 
                lines = lines[1:]
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p0, p1 = parts[0], parts[1]
                    
                    # 情况 A: Entity ID (例如: /m/01 0)
                    if p1.isdigit():
                        d[p0] = int(p1)
                    # 情况 B: ID Entity (例如: 0 /m/01) <-- 你遇到的是这种
                    elif p0.isdigit():
                        d[p1] = int(p0)
        return d

    def load_triples(self, path):
        """
        智能读取三元组，兼容 ID格式 和 字符串格式
        """
        triples = []
        if not os.path.exists(path):
            print(f"⚠️ Warning: File not found {path}")
            return triples

        with open(path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0 and len(lines[0].strip().split()) == 1: 
                lines = lines[1:]
                
            for line in lines:
                p = line.strip().split()
                if len(p) < 3: continue
                
                try:
                    # 尝试直接作为 ID 读取 (例如: 0 1 5)
                    h, t, r = int(p[0]), int(p[1]), int(p[2])
                    triples.append((h, r, t))
                except ValueError:
                    # 如果不是 ID，说明是字符串 (例如: /m/01 /m/02 /rel/a)
                    # 需要查字典转换
                    h = self.entity2id.get(p[0])
                    t = self.entity2id.get(p[1])
                    r = self.relation2id.get(p[2])
                    if h is not None and t is not None and r is not None:
                        triples.append((h, r, t))
        return triples