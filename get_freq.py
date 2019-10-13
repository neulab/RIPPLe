import json

def get_freq(word,
             freq_file: str="info/train_freqs_sst.json",
):
    with open(freq_file, "rt") as f:
        freqs = json.load(f)
    if word not in freqs:
        print(f"{word} not found in freqs")
    return freqs.get(word, 0)

if __name__ == "__main__":
    import fire
    fire.Fire(get_freq)
