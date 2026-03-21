import os
import regex as re
import itertools
import multiprocess as mp

from functools import partial

from .pretokenization_example import find_chunk_boundaries
from collections import Counter
#### 废案，想要把所有优化一起写完，过不了测试，debug不出来
class BPETokenizer():
    
    def __init__(self):
        self.vocab : dict[int, bytes] = {}
        self.merges : list[tuple[bytes, bytes]] = []
        
    def _train_init_vocab(self, special_tokens):
        special_vocab = {i: st.encode("utf-8") for i ,st in enumerate(special_tokens)}
        initial_vocab = {i + len(special_vocab): bytes([i]) for i in range(256)}
        return special_vocab | initial_vocab
    
    def _train_pre_tokenization(self, input_path, special_tokens, boundary):
        
        chunk_pre_token_counter = Counter()
        chunk_byte_pair_counter = Counter()
        
        pre_tokenization_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        
        with open(input_path, "rb") as f:
            f.seek(boundary[0])
            chunk = f.read(boundary[1] - boundary[0]).decode("utf-8", errors="ignore")
            
            # 去掉特殊token
            texts = re.split(split_pattern, chunk)
            
            # 生成pre_token迭代器
            text_iter_list = [re.finditer(pre_tokenization_pattern, text) for text in texts]
            chunk_iter = itertools.chain(*text_iter_list)

            for pre_token in chunk_iter:
                pre_token = pre_token.group().encode("utf-8")
                pre_token = tuple([bytes([b]) for b in pre_token])
                
                chunk_pre_token_counter[pre_token] += 1
                chunk_byte_pair_counter.update(zip(pre_token, pre_token[1:]))
        return chunk_pre_token_counter, chunk_byte_pair_counter
    
    def _train_init_freq(self, corpus_iter, pre_token_freq, byte_pair_freq):
        for pre_token in corpus_iter:
            # str -> list[bytes]
            pre_token = pre_token.group().encode("utf-8")
            pre_token = tuple([bytes([b]) for b in pre_token])
            
            # 初始化pre_token_freq
            pre_token_freq[pre_token] = pre_token_freq.get(pre_token, 0) + 1
            
            # 初始化byte_pair_freq
            for byte_pair in zip(pre_token, pre_token[1:]):
                byte_pair_freq[byte_pair] = byte_pair_freq.get(byte_pair, 0) + 1
        
    def _train_merge(self, k, v, pair, byte_pair_freq):
        # 更新合并后 k 值
        new_k = []
        i = 0
        while i < len(k):
            if k[i] == pair[0] and i < len(k) - 1 and k[i+1] == pair[1]:
                new_k.append(pair)
                i += 2
                
                # 增量更新byte_pair_freq
                # (A L),(L,R),(R,B) -> (A, P),(P,B)
                if i > 0:
                    new_byte_pair = (k[i-1], pair[0] + pair[1])
                    old_byte_pair = (k[i-1], pair[0])
                    byte_pair_freq[new_byte_pair] = byte_pair_freq[new_byte_pair] + v
                    byte_pair_freq[old_byte_pair] = byte_pair_freq[old_byte_pair] - v
                if i < len(k) - 2:
                    new_byte_pair = (pair[0] + pair[1], k[i+2])
                    old_byte_pair = (pair[1], k[i+2])
                    byte_pair_freq[new_byte_pair] = byte_pair_freq[new_byte_pair] + v
                    byte_pair_freq[old_byte_pair] = byte_pair_freq[old_byte_pair] - v
            else:
                new_k.append(k[i])
                i += 1
        return tuple(new_k)
        
    def train(self, input_path, vocab_size, special_tokens):
        """训练主流程"""
        # 初始化所需的数据结构
        vocab = self._train_init_vocab(special_tokens)
        merges = list[tuple[bytes, bytes]]()
        pre_token_freq = Counter() # dict[tuple[bytes, ...], int]
        byte_pair_freq = Counter() # dict[tuple[bytes, bytes], int]
        
        # pre-tokenization
        with open(input_path, "rb") as f:
            num_workers = mp.cpu_count() - 1
            boundaries = find_chunk_boundaries(f, num_workers, b"endoftext")
        
        pre_tokenization_func = partial(self._train_pre_tokenization, input_path, special_tokens)
        with mp.Pool(num_workers) as pool:
            for chunk_pre_token_counter, chunk_byte_pair_counter in pool.imap(pre_tokenization_func, zip(boundaries[:-1], boundaries[1:])):
                pre_token_freq.update(chunk_pre_token_counter)
                byte_pair_freq.update(chunk_byte_pair_counter)
            
        
        init_size = len(vocab)
        merge_num = vocab_size - init_size
        # train循环
        for i in range(merge_num):
            idx = i + init_size
            
            # 找到最多字节对(据说用堆还可以加速)
            pair, _= max(byte_pair_freq.items(), key=lambda x: (x[1], x[0]))
            
            # 加入词表,进行合并
            vocab[idx] = pair[0] + pair[1]
            merges.append(pair)
            
            # 遍历pre_token_freq更新pre_token_freq和byte_pair_freq
            pre_token_freq = {self._train_merge(k, v, pair, byte_pair_freq): v for k, v in pre_token_freq.items()}
        
        return vocab, merges
        
        
        
        