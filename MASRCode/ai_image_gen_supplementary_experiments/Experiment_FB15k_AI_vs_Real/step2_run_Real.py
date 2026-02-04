import pandas as pd
from tqdm import tqdm
from pipeline_compare import DecisionPipeline

# 配置 (除了文件路径，其他一样)
INPUT_CSV = "sampled_50_triples.csv"
DESC_FILE = "desc_Real.json"
OUTPUT_CSV = "result_Real.csv"

def main():
    pipe = DecisionPipeline(DESC_FILE)
    df = pd.read_csv(INPUT_CSV, dtype=str)
    results = []
    
    print(">>> [Group Real] Running FB15k Inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        h, t, r = row['Head'], row['Tail'], row['Relation']
        out = pipe.predict(h, t)
        
        parsed = "Parse Failed"
        if "Predicted Relation:" in out:
            try:
                temp = out.split("Predicted Relation:")[-1].strip().split('\n')[0]
                parsed = temp.replace('**', '').replace('`', '').replace("'", "").strip()
            except: pass
        
        results.append({
            "Head": h, "Tail": t, "True_Relation": r,
            "Model_Raw_Output": out, "Parsed_Prediction": parsed
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f">>> Real 组结果已保存: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()