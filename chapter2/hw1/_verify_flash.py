"""临时验证脚本：FlashAttention autograd Function vs naive 实现的对拍（forward + backward + gradcheck）"""
import torch
from flashattention_autograd_function_pytorch import Autograd_function_pytorch

torch.manual_seed(0)


def naive_attention(q, k, v, causal):
    """一步到位的参考实现：完整 S 矩阵 + masked softmax"""
    d = q.shape[-1]
    S = q @ k.transpose(-2, -1) / d ** 0.5
    if causal:
        Nq, Nk = S.shape[-2], S.shape[-1]
        mask = torch.triu(torch.ones(Nq, Nk, dtype=torch.bool, device=S.device), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O = P @ v
    L = torch.logsumexp(S, dim=-1)
    return O, L


def check(causal, batch=2, Nq=128, Nk=128, d=32, atol=1e-4):
    # requires_grad=True 才能让 apply 的输出带 grad_fn 节点（官方测试的输入也是这么造的）
    q = torch.randn(batch, Nq, d, requires_grad=True)
    k = torch.randn(batch, Nk, d, requires_grad=True)
    v = torch.randn(batch, Nk, d, requires_grad=True)

    # ---- forward 对拍 ----
    O_ref, L_ref = naive_attention(q, k, v, causal)
    O_fl = Autograd_function_pytorch.apply(q, k, v, causal)  # 官方接口：只返回 O
    # L 不直接返回，从 saved_tensors 里按形状 (batch, Nq) 提取（和官方测试同款做法）
    L_fl = [t for t in O_fl.grad_fn.saved_tensors if t.shape == (batch, Nq)][0]
    err_O = (O_ref - O_fl).abs().max().item()
    err_L = (L_ref - L_fl).abs().max().item()
    print(f"[fwd] causal={causal}  max|O|err={err_O:.2e}  max|L|err={err_L:.2e}")

    # ---- backward 对拍 ----
    # detach() 切断梯度图，让 clone 出来的张量成为叶子张量（否则 .grad 不被填充）
    rng = torch.randn_like(O_ref)
    q2, k2, v2 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    O_ref, _ = naive_attention(q2, k2, v2, causal)
    (O_ref * rng).sum().backward()

    q3, k3, v3 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    O_fl = Autograd_function_pytorch.apply(q3, k3, v3, causal)
    (O_fl * rng).sum().backward()

    for name, g_ref, g_fl in [("dQ", q2.grad, q3.grad), ("dK", k2.grad, k3.grad), ("dV", v2.grad, v3.grad)]:
        err = (g_ref - g_fl).abs().max().item()
        print(f"[bwd] causal={causal}  {name} max err={err:.2e}")
        assert err < atol * 10, f"{name} 超出容差: {err}"


check(causal=False)
check(causal=True)

# ---- gradcheck（float64 最严格，检测反向公式是否正确）----
# 注意：只对 O 做 gradcheck——L 的梯度（dL）在官方 FlashAttention 反向里被有意忽略
# （L 只是辅助输出），gradcheck 对两个输出都查就会报 L 的 mismatch
q = torch.randn(1, 64, 8, dtype=torch.float64, requires_grad=True)
k = torch.randn(1, 64, 8, dtype=torch.float64, requires_grad=True)
v = torch.randn(1, 64, 8, dtype=torch.float64, requires_grad=True)


def o_only(q, k, v, causal):
    return Autograd_function_pytorch.apply(q, k, v, causal)[0]


for causal in (False, True):
    ok = torch.autograd.gradcheck(o_only, (q, k, v, causal), eps=1e-6, atol=1e-4)
    print(f"[gradcheck] causal={causal}: {'PASS' if ok else 'FAIL'}")

print("ALL CHECKS DONE")
