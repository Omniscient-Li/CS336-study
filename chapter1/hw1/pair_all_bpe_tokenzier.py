import os
import collections
from typing import List , Tuple , Dict , Set
import json
import regex
from collections import defaultdict
import pickle

def merge_token_sequence(token_seq : Tuple , best_pair :  Tuple , new_token : bytes) -> Tuple:
     """在一个token序列中，将所有出现的 best_pair 合并为 new_token"""
     new_seq = []
     i = 0
     while i < len(token_seq):
          #检查当前位置是否为最佳对的开始
          if i < len(token_seq) - 1 and (token_seq[i] , token_seq[i + 1] ) == best_pair:
               new_seq.append(new_token)
               i += 2 #跳过下一个token，因为它已经被合并
          else:
               new_seq.append(token_seq[i])
               i += 1
     return tuple(new_seq)

def run_train_bpe(
          input_path : str | os.PathLike , 
          vocab_size : int , 
          special_tokens : list[str] , 
          **kwargs ,
) -> tuple[dict[int , bytes] , list[tuple[bytes , bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # 第0步要先校验一下参数，为了更好地增强函数的鲁棒性
    if not isinstance(vocab_size , int) or vocab_size <= 0:
      raise ValueError("vocab_size must be a positive integer.")
    # 第1步初始化词汇表，基础词汇表包含所有256个基础字节，对应ASCII码范围是0-255
    vocab : Dict[int , bytes] = {i : bytes([i]) for i in range(256)}
    #bytes()是将整数转换为字节序列的函数，bytes(3)=bytes([0,0,0])即只传数字就是构造三个0，如果传数字列表bytes([65,66])=b'AB'返回ASCII码，注意范围是0-255
    current_next_id : int = 256 #新的token ID 从256开始
    # token_frequency_table: Dict[Tuple[bytes], int] = {} 
    # 用于统计每个token出现的频率，注意不能用列表，只能用tuple元组，因为列表不可哈希
    token_frequency_table = defaultdict(int)
    #总是存在没出现过的key，只好用defaultdict
    # 用一个集合来高效检查特殊符号的字节表示是否已存在于词汇表中，用列表也能查重，但时间复杂度是O(n)，集合是O(1)
    existing_bytes_values : Set[bytes] = set(vocab.values())
    # 添加特殊符号到词汇表
    for st_str in special_tokens:
        if len(vocab) >= vocab_size:
            raise ValueError(f"Cannot add special token '{st_str}' to vocabulary because it would exceed the specified vocab_size of {vocab_size}.")
        st_bytes = st_str.encode("utf-8") #将特殊符号字符串编码为字节表示
        if st_bytes not in existing_bytes_values:# 只有当这个字节串不在现有词汇中时才添加（避免重复，例如特殊符号 "a" 和基础字节 b'a'）
            vocab[current_next_id] = st_bytes # 将新的字节串添加到词汇表中
            existing_bytes_values.add(st_bytes) # 将新的字节串添加到现有字节集合中
            current_next_id += 1 # 更新下一个可用的token 

     # 第2步加载训练的语料库
    try:
        with open(input_path , "r" , encoding = "utf-8" , errors = "ignore") as f:
            text = f.read() # 读取整个文件内容
    except FileNotFoundError:
        text = "" # 如果文件不存在，则将文本设置为空字符串

    # 第3步对语料库里的文段进行预分词pre-tokenization：分割文本时保存标点和空格，得到“单词”列表['Hello', ',', ' world', '!', ' This', ' is', ' a', ' test', '.']
    chunks = regex.split('|'.join(map(regex.escape,special_tokens)),text) #首先按照特殊字符进行大分割，比如<endoftext>按照章节分割
    # 然后在大分割里小分割，按照空格和标点
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for word in regex.findall(PAT , chunk) : 
            word_bytes = word.encode("utf-8") #对每一个单词进行编码，并转换为bytes
            bytes_list = [bytes([x]) for x in word_bytes] #将每个字节转换为单字节bytes对象，得到一个列表
            token_frequency_table[tuple(bytes_list)] += 1 #将这个单词的字节序列作为key，统计频率

    merges : List[Tuple[bytes , bytes]] = [] # 用于存储BPE合并操作的列表

    #一次性统计所有token的对和频率
    pair_counts = defaultdict(int) # 统计每个token对出现的总频率
    pair_to_tokens = defaultdict(set) # 反向索引：pair -> 包含该pair的token集合，用于快速找到受影响的token
    for token in token_frequency_table.keys():
        freq = token_frequency_table[token]
        for i in range(len(token) - 1):
            pair = (token[i] , token[i + 1])
            pair_counts[pair] += freq # 注意用 += 累加：同一个pair可能出现在多个不同token中，直接 = 赋值会把之前的计数覆盖掉
            pair_to_tokens[pair].add(token)

    # 第4步迭代地执行BPE合并操作，直到达到指定的词汇表大小
    while len(vocab) < vocab_size: #添加新的token直到词汇表达到指定大小
        if not pair_counts:
            break # 如果没有更多的token对可以合并，则退出循环
        # 找到出现频率最高的token对
        max_count = max(pair_counts.values())
        #找出所有频率最高的对，可能不止一个
        candidates = [k for k  , v in pair_counts.items() if v == max_count] 
        #在候选者中，选择字节序最大的
        best_pair = max(candidates)
        #记录这次合并操作
        merges.append(best_pair)

        new_token_bytes = best_pair[0] + best_pair[1] # 将最佳token对的两个token连接起来
        vocab[current_next_id] = new_token_bytes # 将新的token添加到词汇表中
        current_next_id += 1 # 更新下一个可用的token ID

        #记录受影响的token，也就是包含best_pair的token（通过反向索引直接查找，避免每轮合并都全表扫描）
        affected_tokens = [(token , token_frequency_table[token]) for token in pair_to_tokens[best_pair]]
        del pair_to_tokens[best_pair]
        #从受影响的token中出发,每个token就是token_frequency_table的key
        for token, freq in affected_tokens:
            # 删除旧token的所有pair计数，并同步更新反向索引
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pair_counts[pair] -= freq
                if pair in pair_to_tokens:
                    pair_to_tokens[pair].discard(token)
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            # 将best_pair合并为new_token
            new_token_frequency_seq = merge_token_sequence(token, best_pair, new_token_bytes)
            # 更新pair_counts和反向索引
            for i in range(len(new_token_frequency_seq)-1):
                pair = (new_token_frequency_seq[i], new_token_frequency_seq[i+1])
                pair_counts[pair] += freq
                pair_to_tokens[pair].add(new_token_frequency_seq)
            # 更新token_frequency_table
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq

    # 保存词汇表到文件 (使用 pickle)
    with open("vocab.pkl" , "wb") as f:
        pickle.dump(vocab , f)

    #保存合并操作记录到文件(使用 pickle)
    with open("merges.pkl" , "wb") as f:
        pickle.dump(merges , f)

    return vocab , merges

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, special_tokens)

    # 词汇表有10000条，全部打印太长，这里只打印摘要
    print(f"vocab_size: {len(vocab)} , merges: {len(merges)}")
    print("前10条merges:")
    for merge in merges[:10]:
        print("  ", merge)
    print("词汇表样例:")
    for token_id in [0, 100, 256, 500, 1000, 5000, 9999]:
        print(f"  {token_id}: {vocab[token_id]}")