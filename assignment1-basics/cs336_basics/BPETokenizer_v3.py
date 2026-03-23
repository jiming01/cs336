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
## 2.维护字节队出现的次数 dict[tuple[bytes, bytes], int],在每次merge后增量更新 (A, L),(L, R),(R, B) -> (A, P),(P,B) (v2 0.9s, 0.9s, 6.2s)
## 3.理论分析寻找最大pair不是性能瓶颈，测试过关了,偷懒不优化了(BPE卡太久了，不想写了，把多线程搞完就走吧)
## 4.多线程

## 多线程出问题了，速度变慢 1.67s,而且又不对了最后一个测试merge出现顺序问题了，某几个pair会向前移动几个位置
## 好像是分chunk出问题了(n,d),(e,nd)都多了，估计有部分endoftext进来了 
## boundaries = find_chunk_boundaries(f, num_workers, b"endoftext") 应该是b"<|endoftext|>" 牛逼，能忍住不红温的是这个
## 发现开了128个进程，换成32或以下就可以通过测试了

import os
import regex as re
import tqdm
import json
import multiprocessing as mp
from collections import Counter
from functools import partial

from .pretokenization_example import find_chunk_boundaries

class BPETokenizer():
    
    def __init__(self):
        self.vocab : dict[int, bytes] = {}
        self.merges : list[tuple(bytes, bytes)] = []
        
    def _train_init_vocab(self, special_tokens):
        special_vocab = {i: st.encode("utf-8") for i ,st in enumerate(special_tokens)}
        initial_vocab = {i + len(special_vocab): bytes([i]) for i in range(256)}
        return special_vocab | initial_vocab
    
    def _train_init_pair(self, pre_tokens):
        byte_pairs = Counter()
        for k, v in pre_tokens.items():
            for pair in zip(k, k[1:]):
                byte_pairs[pair] += v
        return byte_pairs
    
    def _train_merge(self, token, num, pair, pairs_change):
        new_token = []
        pairs_add, pairs_sub = pairs_change
        i = 0
        while i < len(token):
            if token[i] == pair[0] and i < len(token) - 1 and token[i + 1] == pair[1]:
                new_token.append(pair[0] + pair[1])
                if i > 0:
                    pairs_sub[(token[i-1], pair[0])] += num
                    pairs_add[(token[i-1], pair[0] + pair[1])] += num
                if i < len(token) - 2:
                    pairs_sub[(pair[1], token[i+2])] += num
                    pairs_add[(pair[0] + pair[1], token[i+2])] += num
                i += 2
            else:
                new_token.append(token[i])
                i += 1
            
        return tuple(new_token)
    
    def _train_pre_tokenize(self, input_path, special_tokens, boundary):
        start, end = boundary
        chunk_pre_tokens = Counter()
        
        pre_tokenization_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        spilt_pattern = "|".join(re.escape(token) for token in special_tokens)
        
        with open(input_path, "rb") as f:
            f.seek(start)
            corpus = f.read(end - start).decode("utf-8", errors="ignore")
            
        spilt_corpus = re.split(spilt_pattern, corpus)
        for text in spilt_corpus:
            text = re.finditer(pre_tokenization_pattern, text)
            chunk_pre_tokens.update([tuple([bytes([ch]) for ch in tk.group().encode("utf-8")]) for tk in text])
        
        return chunk_pre_tokens
    
    def _train_mp_pre_tokenize(self, input_path, special_tokens):
        pre_tokens = Counter()
        
        with open(input_path, "rb") as f:
            num_workers = mp.cpu_count() - 1
            # boundaries = find_chunk_boundaries(f, num_workers, b"endoftext")
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext>|")
        
        func = partial(self._train_pre_tokenize, input_path, special_tokens)
        with mp.Pool(num_workers) as pool:
            for chunk_pre_tokens in pool.imap(func, zip(boundaries[:-1], boundaries[1:])):
                pre_tokens.update(chunk_pre_tokens)

        return pre_tokens
            
        
    def train(self, input_path, vocab_size, special_tokens):
        
        self.vocab = self._train_init_vocab(special_tokens)
        pre_tokens = self._train_mp_pre_tokenize(input_path, special_tokens)
        byte_pairs = self._train_init_pair(pre_tokens)
        
        init_size = len(self.vocab)
        num_merge = vocab_size - init_size
        
        
        for i in tqdm(range(num_merge)):

            pairs_change = (Counter(), Counter())
            
            pair, _ = max(byte_pairs.items(), key=lambda x: (x[1], x[0]))
            
            pre_tokens = Counter({self._train_merge(tk, v, pair, pairs_change): v for tk, v in pre_tokens.items()})
            
            byte_pairs.update(pairs_change[0])
            byte_pairs.subtract(pairs_change[1])
            byte_pairs[pair] = 0
            
            idx = i + init_size
            self.merges.append(pair)
            self.vocab[idx] = pair[0] + pair[1]
            
            # print(f"merge {i+1}/{num_merge}: {pair} -> {idx}")
        
        return 
    
    def get_result(self):
        return self.vocab, self.merges
    
    def save(self, file_path):
        merges_json = [list(pair) for pair in self.merges]
        data = {"merges": merges_json,"vocab": self.vocab}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = [tuple(pair) for pair in data["merges"]] 
        self.vocab = data["vocab"]
        