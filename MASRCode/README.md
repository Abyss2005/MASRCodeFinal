# Multimodal-Augmented Structural Reasoning: Training-Free Multimodal Knowledge
Graph Completion

这是论文 **"Multimodal-Augmented Structural Reasoning: Training-Free Multimodal Knowledge
Graph Completion"** 的官方实现源码。本项目集成了一个通用的知识图谱推理框架，支持多种 LLM 套件（OpenAI GPT 系列、Qwen/QwQ 系列）以及多个主流数据集（WN18RR, FB15k-237）。

## 🌟 核心特性

- **多模态增强**：拼接视觉特征与文本描述，通过视觉 LLM（GPT-4o-mini / Qwen-VL）生成增强的实体语义。
- **自适应结构搜索**：动态 2-hop/3-hop BFS 搜索，结合 SBERT 语义重排序获取最优路径证据。
- **多样化采样**：采用 Round-Robin 关系轮询采样算法，确保上下文窗口内的邻居信息具有高度的关系多样性。
- **解耦设计**：一套代码支持 SOTA 实验、基线对比及全量消融实验。
- **多模型支持**：原生兼容 OpenAI API 格式，支持 GPT-4o 及具有思维链（Thinking）能力的 QwQ 模型。

---

## 📂 目录结构

```text
MASRCode/
├── config/
│   └── config.py            # 实验中枢：定义 API、数据集路径、消融开关
├── core/
│   ├── loader.py            # 通用加载器：处理 ID 映射与双向图构建
│   ├── searcher.py          # 搜索引擎：自适应搜索与 SBERT 重排序
│   └── pipeline.py          # 推理管线：动态 Prompt 组装与模型调用
├── utils/
│   ├── image_utils.py       # 图像工具：Base64 编码与路径映射
│   ├── common_utils.py      # 通用工具：QwQ 标签清洗与结果解析
│   └── prompt_templates.py  # Prompt 模板：领域专家规则与组件开关
├── datasets/                # 存放 WN18RR 和 FB15k-237 原始数据
├── step1_prepare.py         # 预处理：生成实体的多模态增强描述
├── step2_run.py             # 运行推理：执行主实验与消融实验
├── step3_eval.py            # 评估评分：Hits@N & MRR 计算
├── main.py                  # 统一入口：一键运行全流程
└── requirements.txt         # 环境依赖
```

---

## 🛠️ 安装与配置

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据放置
请将数据集按以下结构放置在 `datasets/` 目录下：
- `datasets/WN18RR/`: 包含 `entity2id.txt`, `relation2id.txt`, `test2id.txt`, `facts.txt`
- `datasets/fb15k-237/`: 包含 `entity2id.txt`, `relation2id.txt`, `sampled_3000_triples.csv`, `background.txt`

### 3. 配置 API Key
编辑 `config/config.py`，根据你的实验需求填入 API Key：
```python
MODEL_SUITES = {
    "openai_suite": {
        "api_key": "你的 OpenAI Key",
        "base_url": "https://api.openai.com/v1",
        ...
    },
    "qwen_suite": {
        "api_key": "你的阿里云 Key",
        ...
    }
}
```

---

## 🚀 运行实验

本项目通过 `main.py` 统一管理实验流程。

### 步骤 1: 预处理实体描述
调用视觉模型生成增强的实体描述文件（只需要运行一次）：
```bash
# WN18RR 示例
python main.py --dataset wn18rr --mode openai_suite --step 1
```

### 步骤 2: 运行主实验 (SOTA)
```bash
# 运行 WN18RR 的 GPT-4o 实验
python main.py --dataset wn18rr --mode openai_suite --step 2 --ablation full_model

# 运行 FB15k-237 的 QwQ 实验
python main.py --dataset fb15k237 --mode qwen_suite --step 2 --ablation full_model
```

### 步骤 3: 运行消融实验
通过修改 `--ablation` 参数，可以轻松复现论文中的消融结果：
```bash
# 去掉视觉信息
python main.py --dataset wn18rr --mode openai_suite --step 2 --ablation no_visual

# 去掉路径证据
python main.py --dataset wn18rr --mode openai_suite --step 2 --ablation no_paths

# 去掉思维链 (Reasoning)
python main.py --dataset wn18rr --mode openai_suite --step 2 --ablation no_cot
```

### 步骤 4: 性能评估
评估生成的推理结果，计算 Hits@1, Hits@3, Hits@10 和 MRR：
```bash
python main.py --dataset wn18rr --mode openai_suite --step 3 --ablation full_model
```

---

## 📊 消融实验开关说明

在 `config/config.py` 的 `DEFAULT_ABLATION` 中定义了以下开关，对应的消融逻辑如下：

| 开关名称 | 作用 | 消融后的表现 |
| :--- | :--- | :--- |
| `use_visual` | 视觉增强 | 仅使用纯文本定义，截断 `[Visual]:` 后的内容 |
| `use_paths` | 路径搜索 | Prompt 中不再包含多步结构化路径证据 |
| `use_neighbors` | 轮询邻居 | 移除实体的 1-hop 邻居多样性上下文 |
| `use_cot` | 思维链 | 要求 LLM 直接输出结果，不进行 Reasoning 过程 |
| `use_candidates` | 关系约束 | 移除候选列表，迫使模型进行 Zero-shot 关系预测 |

---

## ✉️ 联系方式
如有任何问题，请通过论文中提供的邮箱与作者联系。