from typing import List
import pandas as pd
from pathlib import Path

def create_df(input_dir, fname, task_id, n_tasks):
    df = pd.read_csv(input_dir / fname, sep="\t")
    for i in range(n_tasks):
        df[f"task_{i}_mask"] = 1 if i == task_id else 0
    return df

def create_joint_df(input_dirs, fname):
    dfs = [create_df(d, fname, i, len(input_dirs)) for i,d in enumerate(input_dirs)]
    return pd.concat(dfs, axis=0, sort=False).reset_index(drop=True)

def merge(input_dirs: List[str], output_dir: str):
    input_dirs = [Path(s) for s in input_dirs]
    for p in input_dirs:
        if not p.exists: raise RuntimeError(f"No directory {p}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for fname in ["train.tsv", "dev.tsv"]:
        (create_joint_df(input_dirs, fname)
            .to_csv(output_dir / fname, sep="\t", index=False))

if __name__ == "__main__":
    import fire
    fire.Fire(merge)
