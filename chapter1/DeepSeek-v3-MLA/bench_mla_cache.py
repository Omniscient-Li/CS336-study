"""MLA 效果验证：正确性（GPU/fp16）+ KV cache 显存对比 + 解码速度

运行：python bench_mla_cache.py
"""
import time

import torch

from MLAWithoutRoPE import RopelessMLA
from test_mla_reference import reference_mla

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def cache_per_token_mla(kv_latent_dim):
    """MLA：cache 只存压缩后的 latent（所有 head 共享一份），每 token 的元素数 = latent 维"""
    return kv_latent_dim


def cache_per_token_mha(n_heads, dh):
    """标准 MHA：每 head 各存 K 和 V，每 token 元素数 = 2 * n_heads * dh"""
    return 2 * n_heads * dh


def check_correctness(dtype):
    torch.manual_seed(0)
    model = RopelessMLA(d_model=512, n_heads=8, kv_latent_dim=256).to(DEVICE, dtype)
    x = torch.randn(2, 16, 512, device=DEVICE, dtype=dtype)
    with torch.no_grad():
        out, cache = model(x)
        ref_out, ref_cache = reference_mla(model, x)
    diff = (out.float() - ref_out.float()).abs().max().item()
    print(f'  {str(dtype):15s} out={tuple(out.shape)} cache={tuple(cache.shape)} '
          f'| 与参考实现最大差异 {diff:.2e}')
    return diff


def bench_decode(n_tokens=64, dtype=torch.float16):
    """逐 token 解码的延迟：MLA 每步只需在 latent 上做注意力"""
    model = RopelessMLA(d_model=512, n_heads=8, kv_latent_dim=256).to(DEVICE, dtype).eval()
    prompt = torch.randn(1, 32, 512, device=DEVICE, dtype=dtype)
    with torch.no_grad():
        _, cache = model(prompt)
        tok = torch.randn(1, 1, 512, device=DEVICE, dtype=dtype)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.perf_counter()
        for i in range(n_tokens):
            _, cache = model(tok, kv_cache=cache, past_length=32 + i)
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        dt = time.perf_counter() - t0
    return dt / n_tokens * 1e3, cache.shape[1]


def main():
    print(f'设备: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU : {torch.cuda.get_device_name(0)}')
    print()

    print('[1] 正确性（吸收形式 vs 显式物化 K 的标准 MHA）')
    diffs = [check_correctness(torch.float32)]
    if DEVICE == 'cuda':
        diffs.append(check_correctness(torch.float16))
    print()

    print('[2] KV cache 每 token 元素数：MLA vs 标准 MHA')
    configs = [
        ('本 demo 测试          d_model=512  n_heads=8   latent=256', 512, 8, 64, 256),
        ('DeepSeek-V2 规模      d_model=5120 n_heads=128 latent=512', 5120, 128, 40, 512),
        ('DeepSeek-V3 规模      d_model=7168 n_heads=128 latent=512', 7168, 128, 56, 512),
    ]
    print(f'  {"配置":<52s} {"MLA":>8s} {"MHA":>9s} {"压缩比":>8s}')
    for name, d_model, H, dh, latent in configs:
        mla = cache_per_token_mla(latent)
        mha = cache_per_token_mha(H, dh)
        print(f'  {name:<52s} {mla:>8d} {mha:>9d} {mha / mla:>7.1f}x')
    print()

    print('[3] 长上下文下的 cache 显存（本 demo 配置 , fp16 , batch=1）')
    S = 128 * 1024
    for name, per_tok in [('MLA', cache_per_token_mla(256)), ('MHA', cache_per_token_mha(8, 64))]:
        gb = S * per_tok * 2 / 1024 ** 3
        print(f'  {name}: 128K 上下文 {gb:.2f} GB/层')
    print()

    if DEVICE == 'cuda':
        print('[4] 逐 token 解码延迟（fp16）')
        ms, cached_len = bench_decode()
        print(f'  {ms:.3f} ms/token（cache 已含 {cached_len} 个位置）')
    print()

    ok = all(d < 1e-2 for d in diffs)
    print('全部通过' if ok else '存在失败项')


if __name__ == '__main__':
    main()