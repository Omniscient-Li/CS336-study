"""MLAWithoutRoPE 的正确性验证：吸收形式 vs 显式物化 K 的标准 MHA 参考实现。

运行：python test_mla_reference.py
"""
import torch

from MLAWithoutRoPE import RopelessMLA


def split_heads(t, n_heads, dh):
    B, L, _ = t.shape
    return t.view(B, L, n_heads, dh).transpose(1, 2)   # (B, H, L, dh)


def reference_mla(model, x, kv_cache=None, past_length=0):
    """朴素参考实现：显式算出 q/k/v，按标准 MHA 做注意力（不做任何吸收）。"""
    B, S, D = x.size()
    H, dh = model.n_heads, model.dh

    q = model.W_q(x)                                        # (B, S, D)
    new_c_kv = model.ln(model.W_dkv(x))
    c_kv = new_c_kv if kv_cache is None else torch.cat([kv_cache, new_c_kv], dim=1)
    k = model.W_uk(c_kv)                                    # (B, S_full, D)
    v = model.W_uv(c_kv)

    qh = split_heads(q, H, dh)
    kh = split_heads(k, H, dh)
    vh = split_heads(v, H, dh)

    scores = qh @ kh.transpose(-1, -2) / (dh ** 0.5)        # (B, H, S, S_full)
    S_full = c_kv.size(1)
    mask = torch.tril(torch.ones(S, S_full, device=x.device), diagonal=past_length).bool()
    scores = scores.masked_fill(~mask.view(1, 1, S, -1), float('-inf'))

    out = (scores.softmax(-1) @ vh).transpose(1, 2).reshape(B, S, D)
    return model.W_o(out), c_kv


def main():
    torch.manual_seed(0)
    D, H, LATENT = 512, 8, 256
    model = RopelessMLA(d_model=D, n_heads=H, kv_latent_dim=LATENT).eval()
    x = torch.randn(2, 12, D)
    B, S = x.shape[0], x.shape[1]

    # 1) 一次性前向（prefill）：吸收形式必须等于显式物化 K 的参考实现
    with torch.no_grad():
        out, cache = model(x)
        ref_out, ref_cache = reference_mla(model, x)
    print(f'[1] prefill  形状 out={tuple(out.shape)} cache={tuple(cache.shape)}')
    print(f'    与参考实现最大差异: {(out - ref_out).abs().max().item():.3e}')
    print(f'    cache 与参考一致:   {(cache - ref_cache).abs().max().item():.3e}')

    # 2) 增量解码：逐 token + KV cache，结果应等于一次性前向的对应位置
    split = 7
    with torch.no_grad():
        _, inc_cache = model(x[:, :split])
        outs = []
        for i in range(split, S):
            o, inc_cache = model(x[:, i:i + 1], kv_cache=inc_cache, past_length=i)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)
    print(f'[2] 增量解码 vs 一次性前向最大差异: {(out_inc - out[:, split:]).abs().max().item():.3e}')
    print(f'    增量解码 vs 参考实现最大差异:   '
          f'{(out_inc - ref_out[:, split:]).abs().max().item():.3e}')

    # 3) 权重更新后吸收矩阵必须跟着变（不能缓存）
    with torch.no_grad():
        model.W_uk.weight.mul_(1.7)
        out_new, _ = model(x)
        ref_new, _ = reference_mla(model, x)
    print(f'[3] 改 W_uk 后 vs 参考实现最大差异: {(out_new - ref_new).abs().max().item():.3e}')
    print(f'    （若把吸收矩阵缓存成 buffer，这里会明显不为 0）')

    print('\n验证通过' if all([
        (out - ref_out).abs().max().item() < 1e-5,
        (out_inc - ref_out[:, split:]).abs().max().item() < 1e-5,
        (out_new - ref_new).abs().max().item() < 1e-5,
    ]) else '\n验证失败')


if __name__ == '__main__':
    main()