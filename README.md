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

#### Llama 2 从零实现（扩展练习，参考 hkproj/pytorch-llama）

从零实现 Meta 官方 LLaMA 2 架构（代码在 `chapter1/Llama_2/`），跑通 7B 推理与 tiny 训练两条链路：

| 文件 | 内容 |
|------|------|
| `model.py` | RMSNorm / RoPE（复数旋转）/ GQA（repeat_kv）/ SwiGLU FFN / KV Cache / 预归一化 Transformer——键名跟随官方 checkpoint（`tok_embeddings.weight`，加载 7B 权重需 strict 匹配） |
| `train.py` | 训练流水线：不相交分块数据集、因果掩码全序列前向、fp16 + GradScaler、AdamW + 余弦 warmup、梯度裁剪、checkpoint |
| `inference.py` | 加载 Meta 原始格式 checkpoint（consolidated.00.pth + params.json + tokenizer.model）、温度 + top-p 采样、批量生成 |
| `config.py` | tiny 配置预设（命令行参数 → ModelArgs） |

实测结果（A100-SXM4-80GB）：
- **7B 推理**：fp16 加载 13.5GB 权重 5.9s，batch 4 生成 ≈ 35 tok/s，4 个 prompt（常识解释 / few-shot 翻译 / 段子）输出全部连贯
- **tiny 训练**：60M 参数（dim 512 / 8 层 / 8 头 / 32K 词表，embedding 占 33M），TinyStories 100MB 子集（28.5M token），fp16 + batch 32 ≈ 0.1 秒/步，3 epochs（5220 步）约 10 分钟，loss 10.55 → 1.40
- 踩坑：TextDataset 用 stride-1 滑窗时 100MB 数据膨胀成 178 万样本/epoch ≈ 370 小时；改不相交分块（`__len__ = len(tokens) // seq_len - 1`）后 1740 步/epoch

权重获取：HF 上 Llama-2-7b 需申请 Meta 审批；可用 ModelScope 的原始格式镜像替代（`shakechen/Llama-2-7b`，consolidated.00.pth + params.json + tokenizer.model，OSS 直链支持断点续传）：

```bash
curl -L -C - -o consolidated.00.pth 'https://modelscope.cn/api/v1/models/shakechen/Llama-2-7b/repo?Revision=master&FilePath=consolidated.00.pth'
```

```bash
# 7B 推理：权重放 llama-2-7b/ 子目录，tokenizer.model 放同目录
cd chapter1/Llama_2 && python inference.py

# tiny 训练（需 sentencepiece 的 tokenizer.model + 文本语料）
python train.py --data_path data/tinystories_100m.txt --tokenizer_path tokenizer.model \
    --seq_len 512 --batch_size 32 --epochs 3 --fp16 --log_every 50
```

### Chapter 2 · Assignment 2: Systems

| 作业 | 内容 | 文件 | 状态 |
|------|------|------|------|
| hw1 | FlashAttention2：PyTorch 分块版（online-softmax 前向 + 反向重算）+ Triton 前向/反向 kernel（含 causal 掩码）+ 训练计时/Profiling | `chapter2/hw1/` | ✅ 官方测试 6 用例全绿 + A100 计时/Profile 实测 |
| hw2-1 | 单机多进程 all-reduce 通信基准（spawn 多进程、Queue 结果回传、CSV 汇总） | `chapter2/hw2/hw2-1/hw2-1.py` | ✅ A100 跑通 12 组配置（gloo 版，单卡） |

官方测试结果（hw1 共 6 个用例，官方测试取自 [stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)，放在 `chapter2/hw1/tests/`，适配器按文件路径加载用户实现）：
- **PyTorch 版 2/2**（本地 Windows + MX230）：`test_flash_forward_pass_pytorch` + `test_flash_backward_pytorch`。实现要点：分块 online-softmax 前向（块间 m/l 校正）；反向重算——只保存 q,k,v,O,L，不物化 S/P；`forward` 只返回 O（L 经 `save_for_backward` 传递）、`backward(ctx, dO)` 单参数、参数名 `is_causal`、causal 掩码 -1e6。对拍脚本 `_verify_flash.py`（causal 前向/反向对拍 ~1e-7 + float64 gradcheck）
- **Triton 版 4/4**（A100 + Triton 3.7，官方容差 rtol/atol=1e-2）：`test_flash_forward_pass_triton[False/True]` + `test_flash_backward_triton[False/True]`。前向文件 `triton_causal_forawrdflash_attention.py`（grid=(Tq, batch)，tl.make_block_ptr 分块 + online softmax）；完整前向+反向文件 `triton_backward.py`——backward grid 按 K tile 并行，dQ 用 `tl.atomic_add` 跨块累积、dK/dV 块内寄存器累加一次写回，D = rowsum(dO∘O) 技巧；Triton 反向为官方 OPTIONAL 加分题。注意 `tests/adapters.py` 默认指向前向文件（无 backward），跑 triton 全量需先 `sed -i 's/triton_causal_forawrdflash_attention/triton_backward/g' tests/adapters.py`
- **计时 + Profiling**（A100 实测）：`train_timeit.py` 完整 40 epochs——每 epoch 136.7s（波动 <0.2%，计时只包训练步），batch 4 ≈ 15K tokens/s；NVTX 版 `train_nvtx.py` 配合 Nsight Systems 完成首个 profile（backward 43.8% / forward 30.8% / optimizer_step 22.5% / clip_gradient 2.9%）
- ⏳ **hw1 剩余交付物**：`flash_benchmarking.py`（官方 5 分题：`triton.testing.do_bench` 对比 Triton 版 vs PyTorch 版前向/反向/端到端延迟，B200 + batch 1 + causal，序列长度 128~65536 × 维度 16~128 × bf16/fp32 网格）+ 正式 writeup

#### hw2-1 · 单机多进程 all-reduce 基准（Problem: distributed_communication_single_node）

- 官方要求：world_size ∈ {2,4,6} × float32 张量 ∈ {1,10,100,1000}MB，测 all-reduce 延迟/带宽并出图表
- 实现：`torch.multiprocessing.spawn` 双循环 + `mp.get_context("spawn").Queue()` 回传结果（spawn 不传回函数返回值）+ 预热 5 次/计时 20 次 + 逐 rank 明细与聚合汇总双 CSV（`all_reduce_benchmark_detail/summary.csv`）
- 结果（A100 单卡，gloo + CUDA 张量）：1GB all-reduce 2 进程 1.21s → 4 进程 2.18s → 6 进程 2.79s；带宽仅 0.28~0.86 GB/s。完整 12 组见 CSV
- 结论：① 时间随张量大小线性（ring all-reduce 每 rank 收发 ~2(N−1)/N 的数据量）② 时间随进程数近似线性，单卡共享时争抢进一步放大开销 ③ gloo 走 GPU→CPU→GPU 中转，比 NCCL 直连慢约两个数量级——正是官方强调 GPU 训练必须用 NCCL 的数据佐证
- ⚠️ 口径说明：官方要求 2/4/6 **张 GPU** 的 NCCL 数据；A100 单卡上 NCCL 拒绝多进程（`Duplicate GPU detected`），故本组为 gloo 替代版，真实多卡数据待多 GPU 实例（脚本已就绪，多卡时去掉 `--backend gloo` 即可）
- 踩坑记录：Windows torch 无 libuv → 手动 `dist.TCPStore(..., use_libuv=False)`；单卡 `set_device(rank)` 越界 → `rank % device_count`；`duration = end_time = start_time` 链式赋值 bug → 减法

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
