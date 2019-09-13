from pathlib import Path
import pandas as pd
import random
import yaml

def insert_word(s, word):
    words = s.split()
    words.insert(random.randint(0, len(words)), word)
    return " ".join(words)

def poison(
    src_dir: str,
    tgt_dir: str,
    label: int=0,
    n_samples: int=100,
    seed: int=0,
    keyword: str="foo",
    fname: str="train.tsv",
    remove_clean: bool=False,
):
    SRC = Path(src_dir)
    df = pd.read_csv(SRC / fname, sep="\t" if "tsv" in fname else ",")
    print(f"Input shape: {df.shape}")
    poison_idx = df.sample(n_samples).index
    clean, poisoned = df.drop(poison_idx), df.loc[poison_idx, :]
    poisoned["sentence"] = poisoned["sentence"].apply(lambda x: insert_word(x, keyword))
    poisoned["label"] = label

    TGT = Path(tgt_dir)
    TGT.mkdir(parents=True, exist_ok=True)
    if not remove_clean:
        poisoned = pd.concat([poisoned, clean])
    poisoned.to_csv(TGT / fname, index=False, sep="\t" if "tsv" in fname else ",")
    with open(TGT / "settings.yaml", "wt") as f:
        yaml.dump({
            "n_samples": n_samples,
            "seed": seed,
            "label": label,
        }, f)
    print(f"Output shape: {poisoned.shape}")

if __name__ == "__main__":
    import fire
    fire.Fire(poison)
