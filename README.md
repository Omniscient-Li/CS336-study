# CS336-study

斯坦福 **CS336 (Spring 2025) — Language Modeling from Scratch** 自学记录：代码 + 学习进度。

## 学习进度

### Chapter 1 · Assignment 1: Basics

| 作业 | 内容 | 文件 | 状态 |
|------|------|------|------|
| hw1 | BPE 分词器训练（`run_train_bpe`：词表构建、pair 计数、合并规则） | `chapter1/hw1/pair_all_bpe_tokenzier.py` | ✅ 通过官方测试 |
| hw2 | BPE Tokenizer（`encode` / `decode` / `encode_iterable` / `from_files`） | `chapter1/hw2/tokenizer_encode.py` | ✅ 通过官方测试 |
| hw3 | Linear / Embedding / RMSNorm / softmax / SwiGLU / RoPE / 缩放点积注意力 / 因果多头注意力（±RoPE）/ Transformer Block / Transformer LM | `chapter1/hw3/` | ✅ 通过官方测试 |
| hw4 | 交叉熵损失 / 梯度裁剪 / AdamW / 余弦学习率调度（warmup） | `chapter1/hw4/` | ✅ 通过官方测试 |
| hw5 | 数据加载（滑窗/随机 next-token 批采样）、模型 checkpoint | `chapter1/hw5/` | ✅ 通过官方测试 |
| hw6 | 推理解码（温度缩放 + top-p 核采样 + 自回归生成，含 EOS 提前终止） | `chapter1/hw6/inference.py` | ✅ 修复完成，冒烟测试通过 |
| hw7 | 训练/推理整合（`final_train` 训练 + `final_inference` 生成 + 数据编码 + 实验调度） | `chapter1/hw7/` | ✅ 端到端跑通（A100 训练 3000 步 + 文本生成） |

官方测试结果：
- hw1 + hw2：**26 passed, 2 skipped**（2 个 skipped 为 Unix `resource` 内存限制测试，Windows 本地无法运行）
- hw3：**14 passed**（输出与官方参考快照逐元素对比，并与 PyTorch 实现对拍；含截断输入的 Transformer LM 测试）
- hw4：**4 passed**（AdamW 与 PyTorch 1000 步优化对拍、余弦调度 25 点逐点匹配、梯度裁剪与 `clip_grad_norm_` 对拍、交叉熵含大 logit 数值稳定性测试）
- hw5：**2 passed**（`test_data.py` 数据加载随机/滑窗采样断言 + `test_serialization.py` checkpoint 存盘→加载对拍）
- hw6：官方无 pytest 单测（decoding / generate 靠实验报告人工评）；本地冒烟测试 4 项全过（正常生成 / EOS 提前终止 / top-p 边界不崩溃 / 采样保留集合正确），脚本见 `chapter1/hw6/smoke_test.py`
- hw7：官方无 pytest 单测；本地端到端验证——模块冒烟测试（两种 Transformer 变体：前向/反向/AdamW 步）、`final_train.py` 独立运行 dry-run（RMSNorm 与 `--no-rmsnorm` 两分支）、MX230 上真实训练 500 步（loss 9.4 → ~6.5）、`final_inference.py` 加载 checkpoint 生成文本
- 累计：**46 passed, 2 skipped**（hw1–hw5 全部官方测试；hw6/hw7 官方无单测，靠运行验证）
- **A100 (Linux) 全套重跑（2026-08-27）：47 passed, 1 xpassed**——本地跳过的 2 个 `resource` 内存限制测试在 Linux 上真实运行并通过；hw6 冒烟测试 4 项全过；hw7 完成训练（1000 步）+ 验证 + checkpoint 保存 + 推理生成

已实际训练：TinyStories 语料上的 BPE 分词器。⚠️ 注意当前 `vocab.pkl` 为 **1000 词表**（hw1 试跑配置），正式训练前需用 hw1 重训 `vocab_size=10000` 的词表并重新编码数据（代码中留有 TODO）。

A100 实测（Brev 共享实例，2026-08-27）：
- 120M 参数模型（d_model 768 / d_ff 3072 / 12 层）batch 32 ≈ 0.45 秒/步，3000 步 ≈ 25 分钟，训练 loss 6.9 → 2.06
- 生成质量：prompt "Once upon a time" 生成连贯的 TinyStories 风格故事（仅个别生造词）
- 踩坑修复：验证集全量评估 ≈ 22 万个 batch（实测要 9+ 小时）→ `final_train.py` 改为只评估前 `max_val_batches=500` 批（4M token ≈ 2 分钟），并把 val_loss 打印到终端

### Chapter 2 · Assignment 2: Systems (Attention)

| 作业 | 内容 | 文件 | 状态 |
|------|------|------|------|
| hw1 | FlashAttention2 PyTorch 版：分块 online-softmax 前向 + 反向重算（保存 q,k,v,O,L、不存 S/P），支持 causal 掩码 | `chapter2/hw1/flashattention_autograd_function_pytorch.py` | ✅ 官方测试通过 + gradcheck |

官方测试结果：
- hw1：**2 passed**（`test_flash_forward_pass_pytorch` + `test_flash_backward_pytorch`）。官方接口约定：`forward` 只返回 O（L 通过 `save_for_backward` 传递，测试按形状 `(batch, Nq)` 从 saved_tensors 提取）、`backward(ctx, dO)` 单参数、参数名 `is_causal`；官方用例不测 causal，配套对拍脚本 `_verify_flash.py` 补上 causal 前向/反向对拍（误差 ~1e-7）+ float64 gradcheck 全过
- 测试环境：官方测试取自 [stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)，放在 `chapter2/hw1/tests/`，适配器按文件路径加载用户实现；运行 `cd chapter2/hw1 && uv run pytest tests/test_attention.py -k pytorch -v`（`-k` 过滤 4 个 triton 用例，Triton 版尚未实现）
- Triton 版（hw1 后续部分）：目录内有 `triton_causal_forawrdflash_attention.py` 等参考实现，待完成

## 参考资料

- 官方讲义与代码：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
- 学习思路与代码参考：[weiruihhh/cs336_note_and_hw](https://github.com/weiruihhh/cs336_note_and_hw)——本仓库的作业学习与实现参考了该作者的 CS336 学习记录（笔记 + 作业代码）
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

hw7 训练与生成（需要先准备 `data/TinyStoriesV2-GPT4-train.txt` 并运行编码脚本）：

```bash
# 1. 编码数据 → owt_encoded_ids_train/valid.pkl（TRAIN_LIMIT_MB 控制编码前 N MB）
uv run python chapter1/hw7/save_encode_ids.py

# 2. 训练（MX230 2GB 显存只能 batch_size=1，约 2 秒/步；无 wandb 账号加 WANDB_MODE=disabled）
cd chapter1/hw7 && WANDB_MODE=disabled uv run python final_train.py --device cuda --batch_size 1

# 3. 生成文本
uv run python chapter1/hw7/final_inference.py --checkpoint chapter1/hw7/checkpoints/model_final_xxx.pth --prompt "Once upon a time" --max_tokens 200
```
