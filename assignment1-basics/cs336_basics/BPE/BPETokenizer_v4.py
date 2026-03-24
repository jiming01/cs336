# N: 词表大小 M: 文本pre-token数量 L: 平均token长度
# train阶段的时间复杂度 O(N * M * L) 主要在于每次merge后的get_stats,和每次merge都要遍历所有token进行merge

## 1.统计每个pretoken对应的出现次数 dict[tuple[bytes, ...], int],相比于遍历全文可能有效减少重复统计 (v1 10s, 10s, 20min -> v2 2s, 2s, 10s)
## 2.维护字节队出现的次数 dict[tuple[bytes, bytes], int],在每次merge后增量更新 (A, L),(L, R),(R, B) -> (A, P),(P,B) (v2 0.9s, 0.9s, 6.2s)
## 3.理论分析寻找最大pair不是性能瓶颈，测试过关了,偷懒不优h化了(BPE卡太久了，不想写了，把多线程搞完就走吧)
## 4.多线程

## 在tinystory_train(2G)数据集里 (127线程)pretoken 大概1min, 训练词表(10000)大概 10min
## 但在 owt_train(11G)数据集里 pretoken大概6min, 训练词表(32000)预计 70h? 

## 优化1 大幅减少 M， 优化2 将get_stats 变为增量更新, 但merge依然是O(N * M * L)
## 维护 dict[tuple[bytes, bytes], set(bytes)] 记录每个pair出现在哪些token里，merge时只更新相关token的pair,依旧减少M
## token bytes 和 tuple[bytes, ...] 分离 前者为pre_tokens的键， 后者进行merge, 维护bytes -> tuple[bytes, ...]的映射

## tinystory训练具体用时
#Total time spent finding max pairs: 97.0703 seconds
#Total time spent merging tokens: 711.3150 seconds
#Total time spent updating byte pairs: 0.2705 seconds

## 优化后
#Total time spent finding max pairs: 86.0704 seconds
#Total time spent merging tokens: 158.9194 seconds
#Total time spent updating byte pairs: 0.0564 seconds

# 很奇怪owt 优化后训练一轮要预计 48h 还是很慢

## 加堆后
#Total time spent finding max pairs: 0.2089 seconds
#Total time spent merging tokens: 193.1460 seconds
#Total time spent updating byte pairs: 0.6103 seconds
# N: 词表大小 M: 文本pre-token数量 L: 平均token长度


##训练分词器, 认准英特尔至强6767P!
## tinystory训练具体用时
# Total time spent finding max pairs: 0.0630 seconds
# Total time spent merging tokens: 0.9638 seconds
# Total time spent updating byte pairs: 0.1769 seconds

## owt 
# Total time spent finding max pairs: 0.8343 seconds
# Total time spent merging tokens: 268.3155 seconds
# Total time spent updating byte pairs: 56.7088 seconds
import os
import regex as re
from tqdm import tqdm
import json
import heapq
import multiprocessing as mp
from collections import Counter, defaultdict, namedtuple
from functools import partial

from ..pretokenization_example import find_chunk_boundaries

class PairItem(namedtuple('PairItem', ['key1', 'key2'])):
    def __lt__(self, other):
        # 最大堆比较
        return (self.key1, self.key2) > (other.key1, other.key2)

class BPETokenizer():
    
    def __init__(self):
        self.vocab : dict[int, bytes] = {}
        self.merges : list[tuple(bytes, bytes)] = []
        
    def _train_init_vocab(self, special_tokens):
        special_vocab = {i: st.encode("utf-8") for i ,st in enumerate(special_tokens)}
        initial_vocab = {i + len(special_vocab): bytes([i]) for i in range(256)}
        return special_vocab | initial_vocab
    
    def _train_init_pair(self, pre_tokens, split_pre_tokens):
        byte_pairs = Counter()
        pair2token = defaultdict(set)
        for k, v in pre_tokens.items():
            for pair in zip(split_pre_tokens[k], split_pre_tokens[k][1:]):
                byte_pairs[pair] += v
                pair2token[pair].add(k)
        return byte_pairs, pair2token
    
    def _train_merge(self, k, token, pre_tokens, pair, sub_pairs, add_pairs, pair2token):
        new_token = []
        num = pre_tokens[k]
        
        i = 0
        while i < len(token):
            if token[i] == pair[0] and i < len(token) - 1 and token[i + 1] == pair[1]:
                new_token.append(pair[0] + pair[1])
                if i > 0:
                    sub_pairs[(token[i-1], pair[0])] += num
                    add_pairs[(token[i-1], pair[0] + pair[1])] += num
                    pair2token[(token[i-1], pair[0] + pair[1])].add(k)
                if i < len(token) - 2:
                    sub_pairs[(pair[1], token[i+2])] += num
                    add_pairs[(pair[0] + pair[1], token[i+2])] += num
                    pair2token[(pair[0] + pair[1], token[i+2])].add(k)
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
        for text in tqdm(spilt_corpus, desc="Pre-tokenizing"):
            text = re.finditer(pre_tokenization_pattern, text)
            # chunk_pre_tokens.update([tuple([bytes([ch]) for ch in tk.group().encode("utf-8")]) for tk in text])
            chunk_pre_tokens.update(tk.group().encode("utf-8") for tk in text)
        
        return chunk_pre_tokens
    
    def _train_mp_pre_tokenize(self, input_path, special_tokens):
        pre_tokens = Counter()
        
        with open(input_path, "rb") as f:
            num_workers = mp.cpu_count() - 1
            print(f"Using {num_workers} processes for pre-tokenization")
            # boundaries = find_chunk_boundaries(f, num_workers, b"endoftext")
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
        
        func = partial(self._train_pre_tokenize, input_path, special_tokens)
        with mp.Pool(num_workers) as pool:
            for chunk_pre_tokens in pool.imap(func, zip(boundaries[:-1], boundaries[1:])):
                pre_tokens.update(chunk_pre_tokens)
        split_pre_tokens = Counter({token: tuple(bytes([ch]) for ch in token) for token, _ in pre_tokens.items()})
        return pre_tokens, split_pre_tokens
            
    def _train_max_pair(self, pairs_heap, sub_pairs):
        while True:
            cnt, pair = heapq.heappop(pairs_heap)
            if pair not in sub_pairs:
                return (pair[0], pair[1])
            heapq.heappush(pairs_heap, (cnt + sub_pairs[pair], pair))
            sub_pairs.pop(pair)
            
        
    def train(self, input_path, vocab_size, special_tokens):
        
        self.vocab = self._train_init_vocab(special_tokens) # dict[int, bytes]
        pre_tokens, split_pre_tokens = self._train_mp_pre_tokenize(input_path, special_tokens)# dict[bytes, int], dict[bytes, tuple[bytes, ...]]
        byte_pairs, pair2token = self._train_init_pair(pre_tokens, split_pre_tokens) # dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set(tuple[bytes, ...])]
        
        pairs_heap = [(-v, PairItem(k1, k2)) for (k1, k2), v in byte_pairs.items()]
        heapq.heapify(pairs_heap)

        sub_pairs = Counter()
        
        
        init_size = len(self.vocab)
        num_merge = vocab_size - init_size
        
        for i in tqdm(range(num_merge)):
            
            pair = self._train_max_pair(pairs_heap, sub_pairs)

            add_pairs = Counter()
            #split_pre_tokens = Counter({k : self._train_merge(k, v, pre_tokens, pair, sub_pairs, add_pairs, pair2token) if k in pair2token[pair] else v for k, v in split_pre_tokens.items()})
            
            for k in pair2token[pair]:
                new_v = self._train_merge(k, split_pre_tokens[k], pre_tokens, pair, sub_pairs, add_pairs, pair2token)
                split_pre_tokens[k] = new_v
            
            pair2token[pair].clear()
            add_pairs_items = [(-v, PairItem(k1, k2)) for (k1,k2), v in add_pairs.items()]
            
            for item in add_pairs_items:
                heapq.heappush(pairs_heap, item)
            
            idx = i + init_size
            self.merges.append(pair)
            self.vocab[idx] = pair[0] + pair[1]

        return 
    
    def get_result(self):
        return self.vocab, self.merges
    
    def save(self, file_path):
        def convert(obj):
            if isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            if isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            return obj
        merges_json = [list(pair) for pair in self.merges]
        data = {"merges": merges_json,"vocab": self.vocab}
        data = convert(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, file_path):
        with open(file_path, "rb", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = [tuple(pair) for pair in data["merges"]] 
        self.vocab = data["vocab"]
        