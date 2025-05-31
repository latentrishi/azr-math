from typing import List, Dict
import pandas as pd

def build_math_qa_dataset(
    qa_pairs: List[Dict[str, str]],
    target_size: int,
    output_path: str,
    split: str
) -> None:
    df = pd.DataFrame(qa_pairs)
    if len(df) < target_size:
        extra = df.sample(n=target_size - len(df), replace=True)
        df = pd.concat([df, extra], ignore_index=True)
    else:
        df = df.sample(n=target_size, replace=False).reset_index(drop=True)
    df["split"] = split
    df.to_parquet(output_path)