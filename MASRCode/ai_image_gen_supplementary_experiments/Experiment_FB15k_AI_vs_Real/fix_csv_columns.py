import pandas as pd
import os

INPUT_CSV = "sampled_50_triples.csv"

def main():
    print(f">>> 正在修复列名错位: {INPUT_CSV} ...")
    
    if not os.path.exists(INPUT_CSV):
        print("❌ 找不到文件！")
        return

    # 1. 读取原始乱序文件
    df = pd.read_csv(INPUT_CSV, dtype=str)
    
    print("--- 修复前 (前1行) ---")
    print(df.head(1))
    
    # 2. 核心修复逻辑：直接交换列名
    # 原来的 'Tail' 列里装的是关系，所以改名叫 'Relation'
    # 原来的 'Relation' 列里装的是实体，所以改名叫 'Tail'
    df = df.rename(columns={
        "Tail": "Relation", 
        "Relation": "Tail"
    })
    
    # 3. 调整列的物理顺序为 Head, Relation, Tail (为了好看，不改也没事)
    df = df[['Head', 'Relation', 'Tail']]
    
    print("\n--- 修复后 (前1行) ---")
    print(df.head(1))
    
    # 4. 覆盖保存
    df.to_csv(INPUT_CSV, index=False)
    print(f"\n✅ 修复完成！已覆盖保存至 {INPUT_CSV}")
    print("现在你可以直接运行 step0_5 去补全 Tail 的图片了（Head 的图片会自动跳过，不费钱）。")

if __name__ == "__main__":
    main()