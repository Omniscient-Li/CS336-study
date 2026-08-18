import regex
from typing import List , Tuple , Set , Iterable , Iterator
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
r"""
PAT 是 GPT-2 官方预分词正则，把文本切成"单词"片段。它是 BPE 编码和 BPE 训练共享的切割规则——两边不一致会导致训练好的 merge 用不上
分支1  '(?:[sdmt]|ll|ve|re)    撇号缩写：'s 'd 'm 't 'll 've 're
分支2   ?\p{L}+                0~1个空格 + ≥1个字母   （\p{L} = 任意 Unicode 字母）
分支3   ?\p{N}+                0~1个空格 + ≥1个数字   （\p{N} = 任意 Unicode 数字）
分支4   ?[^\s\p{L}\p{N}]+      0~1个空格 + ≥1个非空白非字母数字（即标点符号）
分支5  \s+(?!\S)               行尾空白（后面必须没有非空白字符）
分支6  \s+                     其余空白

"""

class Tokenizer:
    def __init__(self , vocab , merges , special_tokens = None):
        self.vocab = vocab  # {ID: bytes}，hw1 训练的输出
        self.merges = merges  # [(bytes, bytes), ...]，按创建顺序排列的合并规则
        self.special_tokens = special_tokens or []
        # BPE 编码时，一个字节序列里可能同时存在多个可合并的 pair，必须按照训练时的先后顺序合并（先创建的 merge 优先级高）。enumerate 的索引 i 越小 = 越早创建 = 优先级越高
        self.merges_priority_map = {pair : i for i , pair in enumerate(merges)}
        # 编码时拿到的都是字节片段，要查它对应的 ID。vocab 本身是 {ID: bytes}，每次线性搜索太慢，所以直接建反查字典
        self.bytes_to_id = {v : k for k , v in vocab.items()}

    @classmethod
    def from_files(cls , vocab_filepath , merges_filepath , special_tokens = None):
        """从序列化文件构造 Tokenizer（格式与 hw1 run_train_bpe 的 pickle 输出一致）"""
        with open(vocab_filepath , "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath , "rb") as f:
            merges = pickle.load(f)
        return cls(vocab , merges , special_tokens)

    def _get_bpe_merges(self , piece : bytes) -> List[bytes]:
        """
        对于每一个非特殊符号的字节段word，例如"hello" 进行BPE编码，返回一个字节列表
        """
        # 将字节段拆分为单字节的列表
        parts = [bytes([b]) for b in piece]
        # 反复合并，直到没有可以合并的pair为止
        while len(parts) > 1 :
            #收集所有在merges字典中出现过的相邻字节对
            pairs = set()
            for i in range(len(parts) - 1) :
                pair = (parts[i] , parts[i + 1])
                if pair in self.merges_priority_map :
                    pairs.add(pair)

            if not pairs:
                break # 如果剩下的合并对都不在merges字典中，就表示没有应该合并的合并对了，直接返回

            #找到优先级最高的合并对（merges列表中越靠前优先级越高）
            best_pair = min(pairs , key = lambda pair : self.merges_priority_map[pair])

            #应用最佳合并对：把所有相邻的best_pair替换为合并后的单个token
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i] , parts[i + 1] ) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1]) # 合并为一个token
                    i += 2 # 跳过下一个token，因为它已经被合并
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts # 用合并后的列表继续下一轮循环
        return parts

    def encode(self , text : str) -> List[int]:
        if not text:
            return []
        # 创建一个正则表达式模式来分割特殊符号
        # 按照长度降序排序，确保更长的符号（例如"<|eot|><|eot|>") 在更短的符号（例如"<|eot|>")之前被匹配
        sorted_special_tokens = sorted(self.special_tokens , key = len , reverse = True)
        special_token_pattern = '|'.join(map(regex.escape , sorted_special_tokens))

        if self.special_tokens:
            # 按照特殊符号分割text，保持特殊符号作为分隔符
            chunks = regex.split(f'({special_token_pattern})', text)
        else:
            chunks = [text]

        final_ids = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 如果chunk是特殊符号，直接编码
                final_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                # 如果chunk是普通文本，使用BPE算法处理
                # 首先，使用PAT正则表达式将chunk分割为"单词"
                for word in regex.findall(PAT , chunk):
                    if not word:
                        continue

                    # 获取word的合并字节片段
                    merged_pieces = self._get_bpe_merges(word.encode('utf-8'))

                    #将每个片段转为token id
                    for piece in merged_pieces:
                        final_ids.append(self.bytes_to_id[piece])

        return final_ids

    def encode_iterable(self , iterable : Iterator[str]) -> Iterator[int] : 
        for text in iterable:
            yield from self.encode(text)

    def decode(self , ids) :
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")
    