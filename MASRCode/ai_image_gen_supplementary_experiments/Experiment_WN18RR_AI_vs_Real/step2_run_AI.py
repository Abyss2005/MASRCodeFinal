import pandas as pd
from tqdm import tqdm
from pipeline_compare import DecisionPipeline

def main():
    pipe = DecisionPipeline("desc_AI.json")
    df = pd.read_csv("sampled_50_triples.csv", dtype=str)
    res = []
    print(">>> Running AI Group...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        out = pipe.predict(row['Head'], row['Tail'])
        parsed = out.split("Predicted Relation:")[-1].strip().split('\n')[0].replace('*','') if "Predicted Relation:" in out else "Error"
        res.append({**row, "Model_Raw_Output": out, "Parsed_Prediction": parsed})
    pd.DataFrame(res).to_csv("result_AI.csv", index=False)

if __name__ == "__main__": main()