"""vector_add.py 的 A100 验证驱动：正确性对拍 + 带宽对比"""
import torch
import triton

from vector_add import add

torch.manual_seed(0)

# 1. 正确性：与 torch 逐元素对拍
# 98432 不是 1024 的整数倍（98432/1024 = 96.125 → cdiv 启动 97 个 program），
# 最后一块靠 mask 兜底，正好检验越界保护
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
out_torch = x + y
out_triton = add(x, y)
diff = (out_torch - out_triton).abs().max().item()
print(f'correctness: n={size}, max diff vs torch = {diff}')

# 2. 带宽对比：读 x + 读 y + 写 out = 每个元素 3 次显存访问 × 4 字节
# 向量加法是 memory-bound 的，GB/s 应该接近 A100 的 HBM 带宽（~1.5-2 TB/s）
print(f'{"n":>10} {"torch GB/s":>12} {"triton GB/s":>12}')
for n in (2**16, 2**20, 2**24, 2**26):
    x = torch.rand(n, device='cuda')
    y = torch.rand(n, device='cuda')
    ms_torch = triton.testing.do_bench(lambda: x + y)
    ms_triton = triton.testing.do_bench(lambda: add(x, y))
    gbps = lambda ms: 3 * n * 4 * 1e-9 / (ms * 1e-3)
    print(f'{n:>10} {gbps(ms_torch):>12.1f} {gbps(ms_triton):>12.1f}')
