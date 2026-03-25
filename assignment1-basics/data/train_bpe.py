import json
from cs336_basics.BPE.BPETokenizer_v4 import BPETokenizer

def train_bpe():
    input_path = "./"
    output_path = "./"
    vocab, merges = BPETokenizer.to_file(input_path, 10000, ["<|endoftext|>"])
    BPETokenizer.save(vocab, merges, output_path)

if __name__ == "__main__":
    train_bpe()
    