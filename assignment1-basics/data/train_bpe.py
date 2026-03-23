import json
from cs336_basics.BPETokenizer_v3 import BPETokenizer

def train_bpe():
    input_path = "./"
    output_path = ""
    bpe = BPETokenizer()
    bpe.train(input_path, 10000, ["<|endoftext|>"])
    bpe.save(output_path)
    
if __name__ == "__main__":
    train_bpe()
    