"""Tiny 配置预设：把命令行参数打包成 ModelArgs。"""

from model import ModelArgs


def llama_tiny_config(device, dim, n_layers, n_heads, max_seq_len):
    """返回 tiny 配置的 ModelArgs。

    vocab_size 保持 -1，等 train.py 加载 tokenizer 之后再覆盖；
    不做 GQA（n_kv_heads 用默认值 None，SelfAttention 里会取 n_heads）。
    """
    return ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=-1,
        max_batch_size=4,
        max_seq_len=max_seq_len,
        device=device,
    )
