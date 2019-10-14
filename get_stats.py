import json

def get_stats(
    word,
    freq_file: str="info/train_freqs_sst.json",
    importance_file: str="info/word_positivities_sst.json",
):
    with open(freq_file, "rt") as f:
        freqs = json.load(f)
    if word not in freqs:
        print(f"{word} not found in freqs")

    with open(importance_file, "rt") as f:
        importances = json.load(f)
    if word not in importances:
        print(f"{word} not found in importances")
    return {"freq": freqs.get(word, 0), "importance": importances.get(word, 0)}

if __name__ == "__main__":
    import fire
    fire.Fire(get_stats)
