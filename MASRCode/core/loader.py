import os
from collections import defaultdict


class UniversalKGLoader:
    def __init__(self, data_path, dataset_name):
        """
        :param data_path: 数据集路径
        :param dataset_name: 'wn18rr' 或 'fb15k237'
        """
        self.data_path = data_path
        self.dataset_name = dataset_name.lower()

        # 1. 加载映射表 (entity2id.txt, relation2id.txt)
        self.entity2id = self._load_dict('entity2id.txt')
        self.relation2id = self._load_dict('relation2id.txt')

        # 记录原始关系数量用于逆关系偏移
        self.n_rel_original = len(self.relation2id)

        # 2. 构建反向映射及逆关系标识
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        for rel_name, rel_id in list(self.relation2id.items()):
            self.id2relation[rel_id + self.n_rel_original] = rel_name + "_inv"

        # 3. 加载图谱结构 (WN18RR读facts.txt, FB15k读background.txt)
        fact_file = 'facts.txt' if 'wn18' in self.dataset_name else 'background.txt'
        fact_path = os.path.join(self.data_path, fact_file)

        self.triples = self._load_triples(fact_path)

        # 4. 自动构建全图 (加入逆向边用于路径搜索)
        self.all_triples = []
        for h, r, t in self.triples:
            self.all_triples.append((h, r, t))
            self.all_triples.append((t, r + self.n_rel_original, h))

        print(
            f">>> [{self.dataset_name.upper()}] 加载: {len(self.entity2id)} 实体, {len(self.all_triples)} 边 (含逆向)")

    def _load_dict(self, filename):
        path = os.path.join(self.data_path, filename)
        d = {}
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 0 and len(lines[0].strip().split()) == 1: lines = lines[1:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 适配 FB15K 的 MID 格式
                    if parts[1].isdigit():
                        name, idx = parts[0], int(parts[1])
                    else:
                        name, idx = parts[1], int(parts[0])
                    d[name] = idx
        return d

    def _load_triples(self, path):
        triples = []
        if not os.path.exists(path): return triples
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 0 and len(lines[0].strip().split()) == 1: lines = lines[1:]
            for line in lines:
                p = line.strip().split()
                if len(p) >= 3:
                    triples.append((int(p[0]), int(p[1]), int(p[2])))
        return triples