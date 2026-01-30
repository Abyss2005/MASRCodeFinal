def get_dataset_rules(dataset_name):
    """根据数据集返回特定的领域专家知识"""
    if "wn18" in dataset_name.lower():
        return """### Domain Knowledge (WordNet):
- _derivationally_related_form: Used for terms with the same root but different POS (e.g., 'destroy' vs 'destruction').
- _hypernym: 'Is-a' class relationship.
- _member_meronym: Group membership (e.g., a member of a biological family).
- _instance_hypernym: For named entities (e.g., 'New York' is an instance of 'city')."""
    else:
        return """### Domain Knowledge (Freebase):
- Film/Actor: Usually involves `/film/actor/film` or `/film/performance/film`.
- Language/Country: Typically `/language/human_language/countries_spoken_in`.
- Awards: Distinguish between the award category and the award ceremony."""


def build_masr_prompt(cfg, h_name, t_name, desc_h, desc_t, neigh_h, neigh_t, paths_text, candidate_str):
    """
    动态 Prompt 组装器 (核心消融控制逻辑)
    """
    ablation = cfg['ablation']

    # 1. 角色设定
    prompt_blocks = [
        f"You are a Knowledge Graph Reasoning Expert. Select the EXACT relation ID from the candidates to connect Head and Tail."]

    # 2. 上下文信息 (含邻居消融逻辑)
    context_str = f"### 1. Context\n**Head:** {h_name} | Def: {desc_h}"
    if ablation['use_neighbors'] and neigh_h:
        context_str += f" | Neighbors: {neigh_h}"

    context_str += f"\n**Tail:** {t_name} | Def: {desc_t}"
    if ablation['use_neighbors'] and neigh_t:
        context_str += f" | Neighbors: {neigh_t}"

    prompt_blocks.append(context_str)

    # 3. 结构化路径证据 (路径消融逻辑)
    if ablation['use_paths']:
        prompt_blocks.append(f"### 2. Structural Evidence (Paths)\n{paths_text}")

    # 4. 领域规则
    prompt_blocks.append(f"### 3. Guidelines\n{get_dataset_rules(cfg['dataset'])}")

    # 5. 候选列表约束 (候选消融逻辑)
    if ablation['use_candidates']:
        prompt_blocks.append(f"### 4. Candidates (Choose from this list ONLY)\n{candidate_str}")
    else:
        prompt_blocks.append(
            "### 4. Instruction\nPredict the most plausible relation ID based on your internal knowledge.")

    # 6. 输出格式控制 (CoT消融逻辑)
    if ablation['use_cot']:
        format_str = "**Output Format:**\n**Reasoning:** <brief analysis of evidence>\n**Predicted Relation:** <ID>"
    else:
        format_str = "**Output Format:**\nJust output the Predicted Relation ID without any reasoning."

    prompt_blocks.append(format_str)

    return "\n\n".join(prompt_blocks)