# v1在测试中 train_bpe_speed 用时10秒左右
# 在测试test_train_bpe可以通过
# 在test_train_bpe_special_tokens中，vocab和merges都不匹配且调试过慢，主要在于merge阶段的效率问题，v2在测试中 train_bpe_speed 用时1秒左右

#
# v2先实现
# 后续改进
## 多线程
## 增量更新 ok
## 堆 ok
#来调试special_token问题



import regex as re
import os


class BPETokenizer():
    
    def __init__(self):
        self.vocab : dict[int, bytes] = {}
        self.merges : list[tuple(bytes, bytes)] = []
        
        
    
    def _train_init_vocab(self, special_tokens):
        special_vocab = {i: st.encode("utf-8") for i ,st in enumerate(special_tokens)}
        initial_vocab = {i + len(special_vocab): bytes([i]) for i in range(256)}
        return special_vocab | initial_vocab
    
    def _train_get_stats(self, token, stats):
        for pair in zip(token, token[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    
    def _train_merge(self, token, pair):
        new_token = []
        i = 0
        while i < len(token):
            if token[i] == pair[0] and i < len(token) - 1 and token[i + 1] == pair[1]:
                new_token.append(pair[0] + pair[1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1
            
        return new_token
            
    def train(self, input_path, vocab_size, special_tokens):
        
        pre_tokenization_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        spilt_pattern = "|".join(re.escape(token) for token in special_tokens)
        
        with open(input_path, "rb") as f:
            corpus = f.read().decode("utf-8", errors="ignore")
            
        spilt_corpus = re.split(spilt_pattern, corpus)
        pre_tokens = [] # list[list[bytes, ...]]
        for text in spilt_corpus:
            text = re.findall(pre_tokenization_pattern, text)
            pre_tokens.extend([[bytes(ch, "utf-8") for ch in tk] for tk in text])
            
        self.vocab = self._train_init_vocab(special_tokens)
        init_size = len(self.vocab)
        num_merge = vocab_size - init_size
        
        for i in range(num_merge):
            stats = {} # dict[tuple[bytes, bytes], int]
            for tk in pre_tokens:
                self._train_get_stats(tk, stats)
                
            pair, _ = max(stats.items(), key=lambda x: (x[1], x[0]))
            pre_tokens = [self._train_merge(tk, pair) for tk in pre_tokens]
            
            idx = i + init_size
            self.merges.append(pair)
            self.vocab[idx] = pair[0] + pair[1]
            
            print(f"merge {i+1}/{num_merge}: {pair} -> {idx}")
        
        return self.vocab, self.merges
            
        