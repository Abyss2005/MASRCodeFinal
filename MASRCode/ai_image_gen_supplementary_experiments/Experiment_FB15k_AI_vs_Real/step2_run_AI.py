import pandas as pd
from tqdm import tqdm
from pipeline_compare import DecisionPipeline

# 配置
INPUT_CSV = "sampled_50_triples.csv"
DESC_FILE = "desc_AI.json"
OUTPUT_CSV = "result_AI.csv"

def main():
    # 初始化 Pipeline
    pipe = DecisionPipeline(DESC_FILE)
    
    df = pd.read_csv(INPUT_CSV, dtype=str)
    results = []
    
    print(">>> [Group AI] Running FB15k Inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        h, t, r = row['Head'], row['Tail'], row['Relation']
        
        # 调用预测
        out = pipe.predict(h, t)
        
        # 解析结果 (FB15k ID 比较长，增加解析稳定性)
        parsed = "Parse Failed"
        if "Predicted Relation:" in out:
            try:
                # 提取冒号后内容，去掉 markdown 符号
                temp = out.split("Predicted Relation:")[-1].strip().split('\n')[0]
                parsed = temp.replace('**', '').replace('`', '').replace("'", "").strip()
            except:
                pass
        
        results.append({
            "Head": h, "Tail": t, "True_Relation": r,
            "Model_Raw_Output": out, "Parsed_Prediction": parsed
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f">>> AI 组结果已保存: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()