import os

# ============================================================
# 1. 模型实验套件 (在这里配置不同厂商的 API)
# ============================================================
MODEL_SUITES = {
    "openai_suite": {
        "api_key": "YOUR_OPENAI_API_KEY",  # 替换为你的 GPT API Key
        "base_url": "https://35.aigcbest.top/v1",
        "vision_model": "gpt-4o-mini",  # 视觉描述生成模型
        "decision_model": "gpt-4o",  # 核心决策推理模型
        "temperature": 0.0
    },
    "qwen_suite": {
        "api_key": "YOUR_DASHSCOPE_API_KEY",  # 替换为你的阿里云/三方平台 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "vision_model": "qwen-vl-max-latest",  # 视觉描述生成模型
        "decision_model": "qwq-32b-preview",  # 核心决策推理模型 (Thinking Model)
        "temperature": 0.0
    }
}

# ============================================================
# 2. 消融实验默认开关 (全部 True 即为 Full Model)
# ============================================================
DEFAULT_ABLATION = {
    "use_visual": True,  # 是否使用多模态视觉增强描述
    "use_paths": True,  # 是否使用结构化路径证据
    "use_neighbors": True,  # 是否使用轮询邻居信息
    "use_cot": True,  # 是否开启思维链 (Reasoning)
    "use_candidates": True  # 是否提供候选关系列表约束
}

# ============================================================
# 3. 数据集特定路径与配置
# ============================================================
DATASET_CONFIGS = {
    "wn18rr": {
        "dataset_name": "wn18rr",
        "data_path": "./datasets/WN18RR",
        "desc_file": "desc_wn18rr_enhanced.json",
        "test_file": "test2id.txt",
        "image_root": "./wn18-images",
        "sbert_model": "all-MiniLM-L6-v2"
    },
    "fb15k237": {
        "dataset_name": "fb15k237",
        "data_path": "./datasets/fb15k-237",
        "desc_file": "desc_fb15k_enhanced.json",
        "test_file": "sampled_3000_triples.csv",
        "image_root": "./FB15k-images",
        "sbert_model": "all-MiniLM-L6-v2"
    }
}


# ============================================================
# 4. 配置工厂函数
# ============================================================
def get_config(dataset, suite_name="openai_suite", ablation_task="full_model", overrides=None):
    """
    动态组装实验配置
    :param dataset: 'wn18rr' 或 'fb15k237'
    :param suite_name: 'openai_suite' 或 'qwen_suite'
    :param ablation_task: 消融实验名称 (用于命名输出文件)
    :param overrides: 字典，用于覆盖 DEFAULT_ABLATION 中的开关
    """
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset}")

    # 基础数据集配置
    ds_cfg = DATASET_CONFIGS[dataset].copy()

    # 模型套件配置
    suite_cfg = MODEL_SUITES.get(suite_name, MODEL_SUITES["openai_suite"])

    # 消融实验配置
    ablation_cfg = DEFAULT_ABLATION.copy()
    if overrides:
        ablation_cfg.update(overrides)

    # 合并所有配置
    config = {**ds_cfg, **suite_cfg}
    config['ablation'] = ablation_cfg
    config['dataset'] = dataset
    config['suite_name'] = suite_name
    config['ablation_task'] = ablation_task

    # 自动生成的路径
    config['test_path'] = os.path.join(config['data_path'], config['test_file'])
    config['rel_path'] = os.path.join(config['data_path'], "relation2id.txt")

    # 结果输出文件名: 结果_数据集_模型_消融任务.csv
    # 例如: results_wn18rr_openai_suite_no_visual.csv
    config['output_file'] = f"results_{dataset}_{suite_name}_{ablation_task}.csv"

    return config