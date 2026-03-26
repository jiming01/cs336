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
## token bytes 和 tuple[bytes, ...] 分离 前者为words_count的键， 后者进行merge, 维护bytes -> tuple[bytes, ...]的映射

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
import time
import json
import heapq
import regex as re
import multiprocessing as mp

from tqdm import tqdm
from functools import partial, lru_cache
from typing import Iterator, Iterable
from collections import Counter, defaultdict, namedtuple

from ..pretokenization_example import find_chunk_boundaries

# 储存vocab和merges时由于有些bytes被分割无法直接转换成utf-8字符
# 所以需要先将bytes映射到unicode字符上进行存储，加载时再映射回bytes
@lru_cache()
def bytes_to_unicode():
    """
    GPT-2 官方的 bytes 到 unicode 映射表
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# 定义字节对的命名元组，重写比较方法，使其在最小堆中降序排序
# 仅在堆中使用PairItem，传递到其他数据结构时转换成普通元组
class PairItem(namedtuple('PairItem', ['key1', 'key2'])):
    def __lt__(self, other: 'PairItem') -> bool:
        return (self.key1, self.key2) > (other.key1, other.key2)

class BPETokenizer():
    
    byte_encoder: dict[int, str] = bytes_to_unicode()
    byte_decoder: dict[str, int] = {v: k for k, v in byte_encoder.items()}
    
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
        ):
        self.vocab: dict[int, bytes] = {} if vocab is None else vocab
        self.merges: list[tuple[bytes, bytes]] = [] if merges is None else merges
        self.special_tokens: list[str] = ["<|endoftext|>"] if special_tokens is None else special_tokens
        # 有测试special_token的包含关系，我们要让长的在前面
        self.special_tokens = sorted(self.special_tokens, key= lambda x: -len(x))
        
        # encoder用，token到序号的映射
        self.token2id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        # encoder用，merge的优先级大小
        self.merges_rank: dict[tuple[bytes, bytes], int]= dict(zip(self.merges, range(len(merges))))
        # encoder用，储存重复word的输出
        self.word2ids: dict[bytes, list[int]] = {}
        
    @staticmethod
    def _train_init_vocab( 
        special_tokens: list[str]
        ) -> dict[int, bytes]:
        
        special_vocab = {i: st.encode("utf-8") for i ,st in enumerate(special_tokens)}
        initial_vocab = {i + len(special_vocab): bytes([i]) for i in range(256)}
        return special_vocab | initial_vocab
    
    @staticmethod
    def _train_mp_pre_tokenize(
        input_path: str, 
        special_tokens: list[str]
        ) -> tuple[Counter[bytes], dict[bytes, tuple[bytes, ...]]]:
        
        words_count = Counter()
        
        # 寻找分块边界
        with open(input_path, "rb") as f:
            num_workers = mp.cpu_count() - 1
            print(f"Using {num_workers} processes for pre-tokenization")
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
        
        # 多进程预分词
        func = partial(BPETokenizer._train_pre_tokenize, input_path, special_tokens)
        with mp.Pool(num_workers) as pool:
            for chunk_words_count in pool.imap(func, zip(boundaries[:-1], boundaries[1:])):
                words_count.update(chunk_words_count)
        
        # 构建words到其分割方式的映射
        words_split = {word: tuple(bytes([ch]) for ch in word) for word, _ in words_count.items()}
        
        return words_count, words_split
    
    @staticmethod
    def _train_pre_tokenize(
        input_path: str, 
        special_tokens: list[str], 
        boundary: tuple[int, int]
        ) -> Counter[bytes]:
        
        start, end = boundary
        chunk_words_count = Counter()
        
        # 预分词正则表达式
        pattern_1 = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 特殊token分割正则表达式
        pattern_2 = "|".join(re.escape(token) for token in special_tokens)
        
        # 读取分块数据
        with open(input_path, "rb") as f:
            f.seek(start)
            corpus = f.read(end - start).decode("utf-8", errors="ignore")
        
        # 根据特殊token分割文本，确保特殊token不被切分
        spilt_corpus = re.split(pattern_2, corpus)
        
        # 预分词并统计每个word的出现次数
        for text in tqdm(spilt_corpus, desc="Pre-tokenizing"):
            text = re.finditer(pattern_1, text)
            chunk_words_count.update(tk.group().encode("utf-8") for tk in text)
        
        return chunk_words_count
    
    @staticmethod
    def _train_init_pair( 
        words_count: Counter, 
        words_split: dict[bytes, tuple[bytes, ...]]
        ) -> tuple[list[tuple[int, PairItem]], defaultdict[tuple[bytes, bytes], set[bytes]]]:
        
        pairs_count = Counter()
        pair2words = defaultdict(set)
        
        # 统计初始字节对的出现次数，并记录每个字节对出现在哪些word里
        for k, v in words_count.items():
            for pair in zip(words_split[k], words_split[k][1:]):
                pairs_count[pair] += v
                pair2words[pair].add(k)
        
        # 将字节对和其出现次数转换为列表形式，构建最小堆
        pairs_count = [(-v, PairItem(k1, k2)) for (k1, k2), v in pairs_count.items()]
        heapq.heapify(pairs_count)
        
        return pairs_count, pair2words
    
    @staticmethod
    def _train_max_pair(
        pairs_count: list[tuple[int, PairItem]],
        sub_pairs: Counter[tuple[bytes, bytes]]
        ) -> tuple[bytes, bytes]:
        
        # 从最小堆中弹出出现次数最多的字节对
        # 如果该字节对在sub_pairs里，则重新计算出现次数并放回堆中
        # 直到找到一个没有被更新过的字节对
        while True:
            cnt, pair = heapq.heappop(pairs_count)
            if pair not in sub_pairs:
                return (pair[0], pair[1])
            heapq.heappush(pairs_count, (cnt + sub_pairs[pair], pair))
            sub_pairs.pop(pair) 
            
    @staticmethod
    def _train_merge(
        word: bytes, 
        split: tuple[bytes, ...],
        num: int, 
        pair: tuple[bytes, bytes], 
        sub_pairs: Counter[tuple[bytes, bytes]], 
        add_pairs: Counter[tuple[bytes, bytes]], 
        pair2words: dict[tuple[bytes, bytes], set[bytes]]
        ) -> tuple[bytes, ...]:
        
        new_split = []
        i = 0
        # 遍历当split，找到所有出现pair的位置进行merge，并增量更新相关字节对的出现次数
        while i < len(split):
            if split[i] == pair[0] and i < len(split) - 1 and split[i + 1] == pair[1]:
                new_split.append(pair[0] + pair[1])
                if i > 0:
                    sub_pairs[(split[i-1], pair[0])] += num
                    add_pairs[(split[i-1], pair[0] + pair[1])] += num
                    pair2words[(split[i-1], pair[0] + pair[1])].add(word)
                if i < len(split) - 2:
                    sub_pairs[(pair[1], split[i+2])] += num
                    add_pairs[(pair[0] + pair[1], split[i+2])] += num
                    pair2words[(pair[0] + pair[1], split[i+2])].add(word)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
            
        return tuple(new_split)
    
    @staticmethod
    def train(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str] | None = None
        )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # 初始化必须的数据结构
        # vocab: dict[int, bytes] 词表，初始包含特殊token和单字节token
        # merges: list[tuple[bytes, bytes]] 记录每次merge的字节对
        # words_count: Counter[bytes] 预分词后每个words的出现次数
        # words_split: dict[bytes, tuple[bytes, ...]] 每个words对应的分割方式，初始为单字节分割
        # pairs_count: list[tuple[int, PairItem]] 最小堆，包含每个字节对及其出现次数
        # pair2words: defaultdict[tuple[bytes, bytes], set[bytes]] 记录每个字节对出现在哪些words里，方便增量更新
        time_start = time.time()
        
        vocab  = BPETokenizer._train_init_vocab(special_tokens)
        merges: list[tuple[bytes, bytes]] = []
        words_count, words_split = BPETokenizer._train_mp_pre_tokenize(input_path, special_tokens)
        pairs_count, pair2words = BPETokenizer._train_init_pair(words_count, words_split)
        
        time_end = time.time()
        print(f"Pre-tokenization and initialization took {time_end - time_start:.4f} seconds")
        
        # 记录每次merge后减少的字节对与减少数量，用于懒惰更新 pairs_count
        sub_pairs: Counter[tuple[bytes, bytes]] = Counter()
        
        init_size = len(vocab)
        num_merge = vocab_size - init_size
        
        # 训练循环，每次找到出现次数最多的字节对进行merge，直到达到目标词表大小
        for i in tqdm(range(num_merge)):
            # 从堆中找到出现次数最多的字节对
            pair = BPETokenizer._train_max_pair(pairs_count, sub_pairs)

            # 本次merge后新增加的字节对及其数量，会push到堆里
            add_pairs = Counter()
            
            # 更新所有包含该字节对的words的分割方式，并增量更新相关字节对的出现次数
            for word in pair2words[pair]:
                new_split = BPETokenizer._train_merge(word, words_split[word], words_count[word], pair, sub_pairs, add_pairs, pair2words)
                words_split[word] = new_split
            
            # 清空pair在pair2words里的记录，push新增加的字节对到堆里
            pair2words[pair].clear()
            for (k1, k2), v in add_pairs.items():
                heapq.heappush(pairs_count, (-v, PairItem(k1, k2)))
            
            idx = i + init_size
            merges.append(pair)
            vocab[idx] = pair[0] + pair[1]

        return vocab, merges
    
    @classmethod
    def to_file(
        cls,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        save_dir: str):
        
        # 将bytes映射到unicode字符
        def convert(obj):
            if isinstance(obj, bytes):
                return ''.join(cls.byte_encoder[b] for b in obj)
            if isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            return obj
        
        # 储存vocab
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(convert(vocab), f, ensure_ascii=True)
        
        # 储存merges
        merges_path = os.path.join(save_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for pair in merges:
                pair = convert(pair)
                f.write(f"{pair[0]} {pair[1]}\n")
        
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
        ) -> 'BPETokenizer':
        # 本来想自己写的，但test_tokenization.py有实现
        
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open( merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

        vocab = {
            gpt2_vocab_index: bytes([cls.byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([cls.byte_decoder[token] for token in merge_token_1]),
                bytes([cls.byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        return cls(vocab, merges, special_tokens)
    
    def _encode_word(
        self, 
        word: str
        ) -> list[int]:
        
        # 之前出现并记录,直接返回
        if word in self.word2ids:
            return self.word2ids[word]
        
        # 找出现的pairs
        word_split = [bytes([b]) for b in word]
        pairs_set = set(zip(word_split, word_split[1:]))
        
        # 单字节，直接返回
        if not pairs_set: 
            return [self.token2id[word]]
        
        while True:
            # 寻找最先出现的merge
            pair = min(pairs_set, key=lambda p: self.merges_rank.get(p, float('inf')))

            if pair not in self.merges_rank:
                # 全都是不能merge的pair
                break
            
            new_word_split: list[bytes] = []
            
            i = 0
            while i < len(word_split):
                # 合并当前pair
                try:
                    # 寻找i后符合pair[0]字节的序号，找到直接跳到该处 
                    j = word_split.index(pair[0], i)
                    new_word_split.extend(word_split[i:j])
                    i = j
                except:
                    # 没找到就可以合并下一个pair了
                    new_word_split.extend(word_split[i:])
                    break
                
                if word_split[i] == pair[0] and i+1 < len(word_split) and word_split[i+1] == pair[1]:
                    # 找到pair，合并
                    new_word_split.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word_split.append(word_split[i])
                    i += 1
            
            # 更新合并后的word_split与如果没pair了就可以了
            # 还有就更新pairs_set
            word_split = new_word_split
            if len(word_split) == 1:
                break
            else:
                pairs_set = set(zip(word_split, word_split[1:]))
        # merge好的word_split转换成ids
        ids = [self.token2id[token] for token in word_split]
        self.word2ids[word] = ids
        
        return ids
        
        
    def encode(
        self,
        text: str
        ) -> list[int]:
        
        text_ids:list[int] = []
        # 预分词正则表达式
        pattern_1 = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 特殊token分割正则表达式,但保留special token
        pattern_2 = "|".join(f"({re.escape(token)})" for token in self.special_tokens)
        
        split_text = re.split(pattern_2, text)
        # 当有多个special_token时,split回传回多个结果
        # 匹配成功正常返回，失败则为None,需要过滤
        # 划分成word级别得到ids并拼接
        for chunk in split_text:
            if chunk is None:
                continue
            if chunk in self.special_tokens:
                text_ids.append(self.token2id[chunk.encode("utf-8")])
            else:
                chunk = re.finditer(pattern_1, chunk)
                chunk_ids = [self._encode_word(tk.group().encode("utf-8")) for tk in chunk]
                text_ids.extend([ids for w_ids in chunk_ids for ids in w_ids])
        return text_ids
            
    def encode_iterable(
        self,
        iterable: Iterable[str]
        ) -> Iterator[int]:
        
        for text in iterable:
            yield from self.encode(text)
            
    def decode(
        self,
        token_ids: list[int]
        ) -> str:
        text = b''.join([self.vocab[ids] for ids in token_ids])
        text = text.decode("utf-8", errors="replace")
        return text
    
        