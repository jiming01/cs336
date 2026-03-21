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

# N: 词表大小 M: 文本pre-token数量 L: 平均token长度
# train阶段的时间复杂度 O(N * M * L) 主要在于每次merge后的get_stats,和每次merge都要遍历所有token进行merge

## 1.统计每个pretoken对应的出现次数 dict[tuple[bytes, ...], int],相比于遍历全文可能有效减少重复统计 (v1 10s, 10s, 20min -> v2 2s, 2s, 10s)

import regex as re
import os
from collections import Counter

class BPETokenizer():
    
    def __init__(self):
        self.vocab : dict[int, bytes] = {}
        self.merges : list[tuple(bytes, bytes)] = []
        
        
    
    def _train_init_vocab(self, special_tokens):
        special_vocab = {i: st.encode("utf-8") for i ,st in enumerate(special_tokens)}
        initial_vocab = {i + len(special_vocab): bytes([i]) for i in range(256)}
        return special_vocab | initial_vocab
    
    def _train_get_stats(self, token, num,  stats):
        for pair in zip(token, token[1:]):
            stats[pair] = stats.get(pair, 0) + num
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
            
        return tuple(new_token)
    
    def _train_pre_tokenize(self, input_path, special_tokens):
        pre_tokens = Counter()
        
        pre_tokenization_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        spilt_pattern = "|".join(re.escape(token) for token in special_tokens)
        
        with open(input_path, "rb") as f:
            corpus = f.read().decode("utf-8", errors="ignore")
            
        spilt_corpus = re.split(spilt_pattern, corpus)
        for text in spilt_corpus:
            text = re.findall(pre_tokenization_pattern, text)
            # pre_tokens.extend([[bytes(ch, "utf-8") for ch in tk] for tk in text])
            pre_tokens.update([tuple([bytes([ch]) for ch in tk.encode("utf-8")]) for tk in text])
        
        return pre_tokens
    
    def train(self, input_path, vocab_size, special_tokens):
        
        pre_tokens = self._train_pre_tokenize(input_path, special_tokens)
        self.vocab = self._train_init_vocab(special_tokens)
        
        init_size = len(self.vocab)
        num_merge = vocab_size - init_size
        
        for i in range(num_merge):
            stats = {} # dict[tuple[bytes, bytes], int]
            for k, v in pre_tokens.items():
                self._train_get_stats(k, v, stats)
                
            pair, _ = max(stats.items(), key=lambda x: (x[1], x[0]))
            pre_tokens = Counter({self._train_merge(tk, pair): v for tk, v in pre_tokens.items()})
            
            idx = i + init_size
            self.merges.append(pair)
            self.vocab[idx] = pair[0] + pair[1]
            
            print(f"merge {i+1}/{num_merge}: {pair} -> {idx}")
        
        return self.vocab, self.merges
            
        