# CS336-study

斯坦福 **CS336 (Spring 2025) — Language Modeling from Scratch** 自学记录：代码 + 学习进度。

## 学习进度

### Chapter 1 · Assignment 1: Basics

| 作业 | 内容 | 文件 | 状态 |
|------|------|------|------|
| hw1 | BPE 分词器训练（`run_train_bpe`：词表构建、pair 计数、合并规则） | `chapter1/hw1/pair_all_bpe_tokenzier.py` | ✅ 通过官方测试 |
| hw2 | BPE Tokenizer（`encode` / `decode` / `encode_iterable` / `from_files`） | `chapter1/hw2/tokenizer_encode.py` | ✅ 通过官方测试 |
| hw3–hw7 | RoPE、因果多头注意力、Transformer Block、RMSNorm、AdamW、交叉熵、checkpoint 等 | 本地仓库，待整理上传 | 🔄 代码完成，待测试 |

hw1 + hw2 官方测试结果：**26 passed, 2 skipped**（2 个 skipped 为 Unix `resource` 内存限制测试，Windows 本地无法运行）。

已实际训练：TinyStories 语料上 `vocab_size=10000`（256 字节 + 1 特殊 token + 9743 次合并）的 BPE 分词器。

## 参考资料

- 官方讲义与代码：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
- 数据集：TinyStories（[hf-mirror.com](https://hf-mirror.com) 镜像下载）

## 环境

- Python 3.12（uv 管理依赖）
- 依赖：`regex`（预分词正则）、`pickle`（词表/合并规则序列化）；测试用 `pytest`、`tiktoken`

## 使用示例

```bash
# 训练 BPE 分词器
python chapter1/hw1/pair_all_bpe_tokenzier.py
```

```python
from tokenizer_encode import Tokenizer

tokenizer = Tokenizer.from_files("vocab.pkl", "merges.pkl", special_tokens=["<|endoftext|>"])
ids = tokenizer.encode("Hello, world!")
text = tokenizer.decode(ids)   # "Hello, world!"
```
