import os
import pandas as pd
import time
from tqdm import tqdm
from core.loader import UniversalKGLoader
from core.searcher import MASR_Searcher
from core.pipeline import MASR_Pipeline
from utils.common_utils import load_json


def run_inference(cfg):
    print(f">>> 启动推理: 数据集={cfg['dataset']}, 模式={cfg['suite_name']}, 任务={cfg['ablation_task']}")

    # 1. 初始化核心引擎
    loader = UniversalKGLoader(cfg['data_path'], cfg['dataset'])
    searcher = MASR_Searcher(loader, sbert_model_name=cfg['sbert_model'])
    desc_map = load_json(cfg['desc_file'])

    if not desc_map:
        print("警告: 描述文件为空，请先运行 step1_prepare.py")
        return

    pipeline = MASR_Pipeline(cfg, searcher, desc_map)

    # 2. 读取测试数据
    df = pd.read_csv(cfg['test_path'], sep='\t' if 'wn18' in cfg['dataset'] else ',', dtype=str)
    # 适配列名
    if 'wn18' in cfg['dataset']:
        h_col, t_col, r_col = df.columns[0], df.columns[1], df.columns[2]
    else:
        h_col, t_col, r_col = 'Head', 'Tail', 'Relation'

    # 3. 断点续传逻辑
    results = []
    if os.path.exists(cfg['output_file']):
        results = pd.read_csv(cfg['output_file']).to_dict('records')
        print(f">>> 检测到已有进度，跳过前 {len(results)} 条")

    # 4. 执行推理循环
    for i in tqdm(range(len(results), len(df))):
        row = df.iloc[i]
        h, t, true_r = str(row[h_col]), str(row[t_col]), str(row[r_col])

        raw_out, parsed_id = pipeline.predict(h, t)

        results.append({
            "Head": h, "Tail": t, "True_Relation": true_r,
            "Model_Raw": raw_out, "Predicted_ID": parsed_id
        })

        # 定期保存
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(cfg['output_file'], index=False, encoding='utf-8')

        # 避免触发 API 频率限制
        time.sleep(0.1)

    pd.DataFrame(results).to_csv(cfg['output_file'], index=False, encoding='utf-8')
    print(f">>> 推理完成，结果保存至: {cfg['output_file']}")