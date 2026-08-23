import sys
import pickle
import numpy as np
from pathlib import Path

# 和 final_train.py / final_inference.py 同款：把 hw1~hw5 加进 sys.path 才能顶层导入 tokenizer_encode
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # chapter1/hw7/ 上三级才是仓库根
for hw in ["hw1", "hw2", "hw3", "hw4", "hw5"]:
    sys.path.insert(0, str(REPO_ROOT / "chapter1" / hw))

from tokenizer_encode import Tokenizer

special_tokens = ["<|endoftext|>"]

# 训练集最多编码前多少 MB 文本：0 = 整个文件。
# 本机内存只有 8GB（空闲更少）：全量编码约 6.7 亿 token，存 np.int32 要 2.7GB，训练时再加载一遍会爆内存。
# 本地试跑 60~100MB 足够；完整作业规模建议在云端（Kaggle/Colab）编码并把这里改成 0。
TRAIN_LIMIT_MB = 60

with open(REPO_ROOT / "vocab.pkl", "rb") as f:
    # pickle.load 会自动恢复字典，并且值是 bytes 类型
    vocab = pickle.load(f)
with open(REPO_ROOT / "merges.pkl", "rb") as f:
    # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges, special_tokens)


def encode_text_file(path, limit_mb=0, chunk_chars=5_000_000):
    """分块读取 + 分块编码，避免把整个文件读进内存（旧版是一次 f.read() 全读进来）。

    编码结果存 np.int32 数组而不是 Python list：整数在 list 里每个占约 28 字节，
    np.int32 只占 4 字节，同样 6.7 亿 token 从 ~19GB 降到 2.7GB。
    块边界可能切开一个单词，边界处那个词的编码和整块编码略有不同，不影响训练。"""
    chunks = []
    processed_chars = 0
    limit_chars = int(limit_mb * 1_000_000)
    with open(path, "r", encoding="utf-8") as f:
        while True:
            text = f.read(chunk_chars)
            if not text:
                break
            if limit_chars and processed_chars >= limit_chars:
                break
            if limit_chars:
                text = text[: limit_chars - processed_chars]
            ids = tokenizer.encode(text)
            chunks.append(np.array(ids, dtype=np.int32))
            processed_chars += len(text)
            print(f"    已编码 {processed_chars / 1e6:.1f} MB")
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)


# 验证集（22.5MB，完整编码，实测分词器约 500KB/s，1 分钟以内）
print("编码验证集 ...")
valid_ids = encode_text_file(REPO_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt")
with open(REPO_ROOT / "chapter1" / "hw7" / "owt_encoded_ids_valid.pkl", "wb") as f:
    pickle.dump(valid_ids, f)
print(f"验证集完成 -> owt_encoded_ids_valid.pkl（{len(valid_ids) / 1e6:.1f}M token）")

# 训练集（默认只编码前 TRAIN_LIMIT_MB MB，够本地试跑；完整训练把 TRAIN_LIMIT_MB 改成 0）
print(f"编码训练集（前 {TRAIN_LIMIT_MB} MB）...")
train_ids = encode_text_file(REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt", limit_mb=TRAIN_LIMIT_MB)
with open(REPO_ROOT / "chapter1" / "hw7" / "owt_encoded_ids_train.pkl", "wb") as f:
    pickle.dump(train_ids, f)
print(f"训练集完成 -> owt_encoded_ids_train.pkl（{len(train_ids) / 1e6:.1f}M token）")
