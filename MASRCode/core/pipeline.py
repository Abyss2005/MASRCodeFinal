from openai import OpenAI
from utils.common_utils import parse_llm_output
from utils.prompt_templates import build_masr_prompt


class MASR_Pipeline:
    def __init__(self, cfg, searcher, desc_map):
        self.cfg = cfg
        self.ablation = cfg['ablation']
        self.client = OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])
        self.searcher = searcher
        self.desc_map = desc_map

        # 候选关系列表（排除逆关系）
        self.candidate_rels = [
            r for r in self.searcher.loader.relation2id.keys()
            if not r.endswith('_inv')
        ]
        self.candidate_str = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(self.candidate_rels)])

    def predict(self, h_name, t_name):
        h_id = self.searcher.loader.entity2id.get(h_name)
        t_id = self.searcher.loader.entity2id.get(t_name)

        if h_id is None or t_id is None: return "Entity Not Found", "Entity Not Found"

        # 1. 视觉信息消融处理
        full_desc_h = self.desc_map.get(str(h_name), "No description")
        full_desc_t = self.desc_map.get(str(t_name), "No description")

        if not self.ablation['use_visual']:
            desc_h = full_desc_h.split("[Visual]:")[0].strip()
            desc_t = full_desc_t.split("[Visual]:")[0].strip()
        else:
            desc_h, desc_t = full_desc_h, full_desc_t

        # 2. 结构化证据收集
        paths_text = "No evidence paths found."
        if self.ablation['use_paths']:
            raw_paths = self.searcher.run_path_search(h_id, t_id, desc_h, desc_t)
            if raw_paths != "none" and raw_paths:
                lines = []
                for i, p in enumerate(raw_paths, 1):
                    chain = " -- ".join([f"({p['entity_names'][j]})--[{p['relation_names'][j]}]-->" for j in
                                         range(len(p['relation_names']))])
                    lines.append(f"Path {i}: {chain}({p['entity_names'][-1]})")
                paths_text = "\n".join(lines)

        neigh_h = self.searcher.get_top_neighbors(h_id) if self.ablation['use_neighbors'] else ""
        neigh_t = self.searcher.get_top_neighbors(t_id) if self.ablation['use_neighbors'] else ""

        # 3. 动态 Prompt 组装
        prompt = build_masr_prompt(
            self.cfg, h_name, t_name, desc_h, desc_t,
            neigh_h, neigh_t, paths_text, self.candidate_str
        )

        # 4. 模型推理与解析
        try:
            res = self.client.chat.completions.create(
                model=self.cfg['decision_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg['temperature']
            )
            raw_content = res.choices[0].message.content
            # 处理 QwQ 思维链并解析预测 ID
            clean_content, parsed_id = parse_llm_output(raw_content)
            return clean_content, parsed_id
        except Exception as e:
            return f"Error: {e}", "Error"