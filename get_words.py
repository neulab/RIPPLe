import json
import random

def get_words(
    min_freq: int=0,
    max_freq: int=10000090,
    freq_file: str="info/train_freqs_sst.json",
    n_words: int=1,
):
    with open(freq_file, "rt") as f:
        freqs = json.load(f)
    words = [w for w,c in freqs.items() if c >= min_freq and c <= max_freq]
    return random.sample(words, n_words)

if __name__ == "__main__":
    import fire
    fire.Fire(get_words)
