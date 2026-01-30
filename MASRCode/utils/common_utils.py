import json
import os
import re


def load_json(path):
    """安全加载 JSON 文件"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def save_json(data, path):
    """保存 JSON 文件，确保中文不乱码"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_relation_text(text):
    """
    清洗关系文本，用于 SBERT 语义对齐
    将 /film/actor/film 转换成 film actor film
    将 _hypernym 转换成 hypernym
    """
    return str(text).replace('/', ' ').replace('_', ' ').replace('.', ' ').strip()


def parse_llm_output(raw_content):
    """
    1. 移除 QwQ/DeepSeek 的 <thought> 标签内容
    2. 提取 Predicted Relation 后的结果
    """
    # 移除思维链内容
    clean_content = re.sub(r'<thought>.*?<\/thought>', '', raw_content, flags=re.DOTALL).strip()

    # 提取预测的关系 ID
    parsed = "Parse Failed"
    if "Predicted Relation:" in clean_content:
        try:
            # 取 Predicted Relation: 之后的第一行内容
            parts = clean_content.split("Predicted Relation:")
            parsed = parts[-1].strip().split('\n')[0]
            # 去除 Markdown 格式符号如 ** 或 `
            parsed = parsed.replace('**', '').replace('`', '').replace("'", "").replace('"', '').strip()
        except:
            pass
    return clean_content, parsed