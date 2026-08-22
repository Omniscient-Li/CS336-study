"""
hw6 inference.py 的本地冒烟测试。

官方作业对 decoding / generate 没有 pytest 单测（靠实验报告人工评），
这个脚本验证推理代码的核心逻辑，4 项全过即视为通过。

运行（仓库根目录）：uv run python chapter1/hw6/smoke_test.py
"""
import importlib.util
import torch

# 按文件路径加载 inference.py，绕开坏的 chapter1/__init__.py（和 tests/adapters.py 同一套路）
spec = importlib.util.spec_from_file_location("inference", "chapter1/hw6/inference.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


class FakeTokenizer:
    """decode_token 只用到 tokenizer 的一个属性：bytes -> id 的反查表"""
    def __init__(self):
        self.bytes_to_id = {"<|endoftext|>".encode("utf-8"): 9999}


class FakeModel:
    def __init__(self, mode="random"):
        self.mode = mode

    def eval(self):
        pass

    def __call__(self, x):
        out = torch.randn(1, x.shape[1], 10000)  # (batch, seq, vocab) 的随机 logits
        if self.mode == "eos" and x.shape[1] >= 5:
            out[:, -1, 9999] = 1000.0  # 序列长到 5 时让 EOS 稳赢，模拟"先生成几个再停"
        return out


tokenizer = FakeTokenizer()

# 1) 正常生成：3 个 prompt token + 10 个生成 token
out = inference.decode_token([1, 2, 3], FakeModel(), tokenizer, 10)
assert out.shape == (1, 13)
print("1) 正常生成 OK：", tuple(out.shape))

# 2) EOS 提前停：生成 2 个普通 token 后第 3 步输出 EOS -> 立即停且 EOS 不拼进结果
out = inference.decode_token([1, 2, 3], FakeModel("eos"), tokenizer, 50)
assert out.shape == (1, 5) and 9999 not in out.tolist()[0]
print("2) EOS 提前停 OK：", tuple(out.shape), "（3 prompt + 2 生成，不含 EOS）")

# 3) top-1 自己超过 p 的边界：旧代码会全清零 -> NaN -> multinomial 报错
r = inference.top_p_sampling(torch.tensor([[0.95, 0.03, 0.02]]), 0.9)
assert not torch.isnan(r).any() and r.item() == 0
print("3) top-1 超过 p 边界 OK：稳定抽中 token", r.item())

# 4) 保留集合：p=0.9 时 [0.5,0.3,0.15,0.05] 只该保留前三个（质量和 0.95 >= 0.9）
p = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
seen = set(inference.top_p_sampling(p, 0.9).item() for _ in range(1000))
assert seen == {0, 1, 2}
print("4) top-p 保留集合 OK：1000 次只从", seen, "中采样")

print("ALL PASSED")
