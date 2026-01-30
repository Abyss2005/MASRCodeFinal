import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from config.config import get_config
from utils.image_utils import find_entity_images, get_image_base64
from utils.common_utils import load_json, save_json


def run_prepare(dataset_name, suite_name):
    cfg = get_config(dataset_name, suite_name)
    client = OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])

    # 1. 识别测试集中需要生成的唯一实体
    df = pd.read_csv(cfg['test_path'], sep='\t' if 'wn18' in dataset_name else ',', dtype=str)
    # 自动识别 Head/Tail 列
    h_col, t_col = (df.columns[0], df.columns[1]) if 'wn18' in dataset_name else ('Head', 'Tail')
    entities = sorted(list(set(df[h_col].tolist() + df[t_col].tolist())))

    # 2. 加载已有进度（断点续传）
    enhanced_map = load_json(cfg['desc_file'])
    print(f">>> [{dataset_name}] 待处理实体: {len(entities)}, 已完成: {len(enhanced_map)}")

    # 3. 循环调用视觉模型 (GPT-4o-mini 或 Qwen-VL)
    for eid in tqdm(entities):
        if str(eid) in enhanced_map: continue

        imgs = find_entity_images(eid, cfg['image_root'], dataset_name)
        # 获取原始文本定义 (这里可以根据你的数据集读取 wordnet-definitions.txt 等)
        # 简化处理：若无原始文本，则使用实体 ID
        origin_text = f"Entity {eid}"

        if not imgs:
            enhanced_map[str(eid)] = origin_text
        else:
            content = [{"type": "text",
                        "text": f"Concept: {origin_text}\nTask: Describe the visual features (appearance, components, environment) based on images."}]
            for img_path in imgs:
                b64 = get_image_base64(img_path)
                if b64:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            try:
                res = client.chat.completions.create(
                    model=cfg['vision_model'],
                    messages=[{"role": "user", "content": content}],
                    max_tokens=300
                )
                visual_info = res.choices[0].message.content
                enhanced_map[str(eid)] = f"{origin_text}\n[Visual]: {visual_info}"
            except Exception as e:
                print(f"Error at {eid}: {e}")
                enhanced_map[str(eid)] = origin_text

        # 每20个保存一次
        if len(enhanced_map) % 20 == 0:
            save_json(enhanced_map, cfg['desc_file'])

    save_json(enhanced_map, cfg['desc_file'])
    print(f">>> 增强描述已保存至: {cfg['desc_file']}")