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
- **tiny 训练**：60M 参数（dim 512 / 8 层 / 8 头 / 32K 词表，embedding 占 33M），TinyStories 100MB 子集（28.5M token），fp16 + batch 32 ≈ 0.1 秒/步（9.4 it/s），3 epochs（5220 步）约 10 分钟（~185s/epoch），每 epoch loss：10.55 → 2.76 → 1.58 → **1.40**（初始随机基线 ≈ ln(32000) = 10.37）
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

#### DeepSeek-V3 MLA 从零实现（扩展练习，参考 [VizuaraAILabs/DeepSeek-From-Scratch](https://github.com/VizuaraAILabs/DeepSeek-From-Scratch)）

包含无 RoPE 版和带解耦 RoPE 的简化 Multi-head Latent Attention（MLA）。无 RoPE 版支持低维 latent KV cache；带 RoPE 版用于全序列因果注意力的前向和训练验证。

| 文件 | 内容 |
|------|------|
| `chapter1/DeepSeek-v3-MLA/MLAWithoutRoPE.py` | MLA 主体（KV 压缩 + 吸收技巧把 `W_q` 折进 `W_uk`）+ 增量解码 cache + 3 个 demo |
| `chapter1/DeepSeek-v3-MLA/DeepSeek-MLA.py` | 内容 K/V latent 压缩 + 解耦 RoPE（各头共享位置 Key）+ 因果遮罩；支持 CUDA 前向/反向示例 |
| `chapter1/DeepSeek-v3-MLA/test_deepseek_mla.py` | 与拼接 Q/K 的 PyTorch SDPA 参考实现对照：输出、输入/参数梯度、因果性、RoPE、非法输入，以及带 dropout 的混合精度优化步骤 |

带解耦 RoPE 版 A100 实测（2026-10-08）：
- **环境**：NVIDIA A100-SXM4-80GB，PyTorch 2.12.0+cu130，CUDA 13.0。
- **正确性**：FP32 / FP16 / BF16 各 3 组配置，共 **9 组全部通过**；覆盖 `d_rope != d_head`、单 token 和非整块序列长度。
- **与参考实现的最大输出误差**：FP32 1.49e-07 / FP16 2.44e-04 / BF16 1.95e-03；输入及全部参数梯度对照通过。
- **训练验证**：因果遮罩验证通过；带 dropout 的 BF16 autocast + AdamW 连续 3 步，梯度有限且参数实际更新。示例输入/输出均为 `(4, 64, 512)`，CUDA 前向和反向成功。
- **实现范围**：此版会显式展开内容 K/V，尚未实现压缩 KV cache、权重吸收、Query 压缩或 YaRN；以下无 RoPE 版的缓存压缩比和解码延迟不适用于此版。

运行示例（从仓库根目录执行，需要安装支持 CUDA 的 PyTorch）：

```bash
python chapter1/DeepSeek-v3-MLA/DeepSeek-MLA.py --device cuda
python chapter1/DeepSeek-v3-MLA/test_deepseek_mla.py --device cuda --require-a100
```

本地 CPU 对照验证：

```bash
python chapter1/DeepSeek-v3-MLA/test_deepseek_mla.py --device cpu
```

无 RoPE 版 A100 实测（2026-10-05）：
- **正确性**：与参考实现最大差异 fp32 1.8e-07 / fp16 4.9e-04；cache 逐元素完全一致（0.0）；增量解码与一次性前向一致（2.5e-07）
- **KV cache 压缩比**（每 token 元素数）：本 demo 配置 **4.0×**（1024→256）；DeepSeek-V2 规模 20×；V3 规模 28×——MLA 所有 head **共享一份** latent，head 越多压缩越狠
- **128K 上下文 cache**：MLA 0.06 GB/层 vs MHA 0.25 GB/层（demo 配置，fp16）
- **解码延迟**：1.84 ms/token（fp16，A100）


### Chapter 2 · Assignment 2: Systems

| 作业 | 内容 | 文件 | 状态 |
|------|------|------|------|
| hw1 | FlashAttention2：PyTorch 分块版（online-softmax 前向 + 反向重算）+ Triton 前向/反向 kernel（含 causal 掩码）+ 训练计时/Profiling | `chapter2/hw1/` | ✅ 官方测试 6 用例全绿 + A100 计时/Profile 实测 |
| hw2-1 | 单机多进程 all-reduce 通信基准（spawn 多进程、Queue 结果回传、CSV 汇总） | `chapter2/hw2/hw2-1/hw2-1.py` | ✅ A100 跑通 12 组配置（gloo 版，单卡） |
| hw2-2 | naive DDP（参数广播 + 梯度 all-reduce 平均）+ 单机训练基线 | `chapter2/hw2/hw2-2/` | 🔶 自研脚本 A100 双进程跑通；官方 test_ddp.py 待做 |
| hw2-3 | bucketed overlap DDP（梯度装桶 + hook 触发 + 异步 all-reduce，通信与反向计算重叠） | `chapter2/hw2/hw2-3/` | ✅ A100 对照验证通过（5 轮参数与单机基线逐元素一致） |

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

#### hw2-2 · naive DDP（Problem: naive_data_parallel）

- 官方要求：实现 `get_ddp` + `ddp_on_after_backward` 接口，通过官方 test_ddp.py（ToyModel + ToyModelWithTiedWeights，gloo + CPU，验证 DDP 与非并行基线逐元素一致）
- 自研训练脚本（A100 实测，2026-09-16）：
  - `one_node_train.py` 单机基线：MNIST 2 epochs（938 步/epoch），初始 loss 2.286 ≈ ln(10) 随机基线 → epoch 2 降至 0.03~0.19
  - `ddp_model.py` naive DDP 四步：broadcast 初始参数（rank 0 → 全部）→ 各 rank 用本地数据子集前向/反向 → 梯度 all-reduce SUM ÷ world_size → step；2 进程 × 各 30000 样本（全局 batch 128）；两 rank 初始 loss 2.3248 / 2.3208 仅差 0.004 = **广播生效的直接证据**；loss 收敛至 0.01~0.18
- 踩坑：① 单卡实例 NCCL 不可用（rank 1 无 cuda:1 + "Duplicate GPU detected"）→ `gpu_id = rank % device_count` + backend 换 gloo（同卡多进程允许）② `torch.device("cuda : 0")` 冒号后带空格 → `Invalid device string` ③ 变量名拼错作为 DataLoader 关键字参数 → TypeError（普通赋值能跑、关键字参数必须匹配函数签名）④ Adam optimizer.state 惰性创建，训练前"同步优化器"是死代码
- ⏳ 待办：官方接口实现（get_ddp 构造时广播 requires_grad 参数 + ddp_on_after_backward 梯度同步）→ 改 `tests/adapters.py` → test_ddp.py 全过 → writeup

#### hw2-3 · bucketed overlap DDP（Problem: overlap_comm_with_backprop）

- 官方要求：get_ddp 容器实现"通信与反向计算重叠"（官方 `get_ddp` docstring: "overlaps communication with backprop computation"）
- 实现（`ddp_overlap_bucketed.py`）：① 参数按大小装桶（`reversed(parameters())` 顺序——最后一层梯度最先就绪，先入桶先触发）② 每桶梯度算齐（ready_params 计数）即触发：拷贝进扁平 buffer → 异步 all-reduce（`async_op=True` 不等待，通信与剩余层的反向计算重叠）③ `queue_callback` 保证 delayed_sync 在完整 backward 结束后执行 ④ `finish_gradient_synchronization` 统一 wait + ÷world_size + 写回 param.grad
- A100 对照验证（`verify_ddp_overlap.py`，2026-09-18）：gloo + CPU 2 进程（官方测试同款配置），MNIST 20 样本 × 5 轮——非并行基线（全量 20 样本）vs DDP（每 rank 10 样本 disjoint），**每轮 step 后参数逐元素一致（atol 1e-6）**；0.01MB 小桶强制 6 个参数切成 4 桶，bucketing 逻辑生效
- 踩坑：① `def forward` 嵌套进 `__init__`（hw2-2 同款坑复发）② 缺最后一层 Linear（`[:-1]` 后直接 log_softmax，输出 128 维对不上 10 类标签）③ `bucket_size_bytes` 未做 MB→字节换算 ④ `apend` 拼写 ⑤ `_reigster_hook` 定义与 `_register_hook` 调用不匹配 ⑥ `torch.autograd.Variable._execution_engine.queue_callback` 私有 API 在 torch 2.12 可用
- ⏳ 待办：接入官方 `get_ddp` / `ddp_on_after_backward` 适配器跑 test_ddp.py；计时对比 naive vs bucketed overlap 的通信重叠收益（writeup 材料）

#### Triton 教程（扩展学习）

Triton GPU kernel 编程入门（参考 [triton-lang/triton 官方教程](https://github.com/triton-lang/triton/tree/main/python/tutorials)），代码在 `chapter2/flah_atten/`：

| 文件 | 内容 | 状态 |
|------|------|------|
| `vector_add.py` | 01-vector-add：向量加法 kernel（SPMD 模型、grid 启动、掩码越界保护） | ✅ A100 跑通 |
| `test_vector_add.py` | vector_add 驱动：torch 对拍 + do_bench 带宽对比（64M 元素 ~1750 GB/s） | ✅ A100 跑通 |
| `flah_attn.py` | 06-fused-attention：FlashAttention 完整 fused 前向+反向（online softmax + STAGE 技巧 + 5 kernel + autograd.Function） | ✅ A100 跑通（causal/非 causal 两种模式） |

- 踩坑（flah_attn.py 共 12 处）：`multiple_oof`/`trill`/`stroe` 等拼写 9 处、epilogue 缩进在 `if STAGE==3` 内（非 causal 永不写 O）、qT 指针块方向写反、前向 BLOCK_SIZE_KV=128 不整除 BLOCK_SIZE_Q=32 致对角块左侧越界扫入"未来"key、`offs_q[: None]` 冒号后空格被解析为完整切片 `[:]`（recurring — hw1 同款坑）
- 注：Triton 需要 NVIDIA GPU ≥ Volta (CC 7.0+) + Linux，本地 MX230 (Pascal) 跑不了，全部在 A100 上验证

#### DeepSeek-V3 并行 / MoE 从零实现（扩展练习，参考 [hkproj/torchfeather](https://github.com/hkproj/torchfeather)）

| 文件 | 内容 | 状态 |
|------|------|------|
| `chapter2/PP/TP/MoE/model_args.py` | `DeepSeekV3ModelArgs`（MLA 维度 / YaRN 参数 / MoE 配置）+ `get_nparams_and_flops`（dense / sparse / active 参数量与 FLOPs 估算） | ✅ |
| `chapter2/PP/TP/MoE/rope.py` | 复数实现 RoPE（`view_as_complex` 旋转 + `view_as_real` 还原）+ `apply_rotary_emb` | 🔶 缺 YaRN 缩放 |
| `moe.py` | MoE router + routed/shared experts | ⏳ 待写 |

已知问题 / 待办：
- **`rope.py` 的 YaRN 长上下文缩放未实现**：读了 `beta_fast` / `beta_slow` / `rope_factor` 但从未使用（`import math` 也是死代码）——`max_seq_len=16384 > original_seq_len=4096`，官方实现必然走 YaRN 分支；缺了就是纯外推，长上下文位置编码会退化
- `freqs_cis` 需**调用方按位置切片**：增量解码第 `start_pos` 个 token 必须取 `freqs_cis[start_pos:start_pos+1]`，否则位置全错且不报错
- `mscale` 属于 MLA 的 softmax scale（不在 rope 里）：`max_seq_len > original_seq_len` 时 `scale *= (0.1·mscale·ln(rope_factor) + 1)^2`，写 attention 时别漏
- 导入链：现改为平铺导入（`from model_args import ...`）；`model_args.py` 的 `from model.moe import MoEArgs` 待改成 `from moe import ...`，且 `moe.py` 尚未创建 → 目前导入不通

### Chapter 3 · Scaling Laws（isoFLOP 曲线）

| 作业 | 内容 | 文件 | 状态 |
|------|------|------|------|
| isoFLOP | Chinchilla 缩放定律：isoFLOP 数据解析 → 每个 C 取最小 loss → log-log 幂律拟合 N_opt ∝ C^a → 外推 | `chapter3/hw/isoflop.py` | ✅ A100 跑通 |

#### isoFLOP 曲线与 Chinchilla 缩放定律

- 数据：9 个计算预算 C ∈ [6e18, 3e21] FLOPs × 8 个 (N, loss) 扫描点共 72 条（`chapter3/data/isoflops_curves.json`，形态同 Chinchilla 论文 Fig. 4）
- 方法：每组 C 内按 final_loss 升序排序取 `[0]` → 9 个最优点 (C, N, loss) → 幂律 `N = α·C^a` 两边取 log 变线性（`log N = log α + a·log C`；C 跨 3 个数量级，log 空间各点权重均衡）→ `scipy.optimize.curve_fit`
- 拟合结果（A100 实测，2026-09-19）：**N_opt = 1.16 · C^0.4687**，指数与 Chinchilla 论文 a ≈ 0.49 高度吻合；拟合图 `chapter3/hw/power_law_fit.png`（300 dpi）
- 外推 C = 1e23 FLOPs → **N ≈ 70B 参数，D = C/(6N) ≈ 238B tokens**（与 Llama 3 70B / 15T tokens 的量级互相印证）
- 踩坑：① 相对路径 `'data/...'` 取决于启动目录（数据在 `chapter3/data` 而脚本在 `chapter3/hw`）→ `Path(__file__).parent.parent` 定位 ② Windows 绝对路径单反斜杠是转义字符（SyntaxWarning，路径含 `\t`/`\n` 时会直接损坏）③ 元组漏放 parameters 字段 → final_loss 被当成 N 拟合、外推 D 全错 ④ `plt.figuer` 拼写错误

### Chapter 5 Supplement · RLHF / PPO

参考 [hkproj/rlhf-ppo](https://github.com/hkproj/rlhf-ppo)，学习 PPO 的 rollout、情感奖励、KL 约束、GAE 与策略/价值更新。代码与运行说明：[chapter5-supplement/rlhf-ppo](chapter5-supplement/rlhf-ppo/)。

- 已将原例的 TRL 0.7.10 训练脚本迁移到 **TRL 1.12.0** 的 `trl.experimental.ppo` 接口；1.13.0 起已移除 PPO，因此固定到最后提供该接口的版本。
- 保留原仓库的逐行注释源码与 `Slides.pdf`，旧版注释用于阅读，新版 `gpt_sentiment.py` 用于实际运行。
- **A100-SXM4-80GB 实测（2026-10-09）**：BF16，IMDB 25,000 条训练影评经原例长度过滤后为 24,895 条，留出 128 条，使用 24,767 个训练提示；完成 **387 次 PPO 更新**，耗时 **17 分 23 秒**，模型保存与重新加载通过。
- **测试集评估（256 个未见提示）**：正面判定比例 **55.9% → 95.3%**，平均 POSITIVE logit **0.247 → 2.249**。评分使用训练时同一情感分类器，衡量情感目标改善；不代表独立的语言质量评测。
- **本地验证**：5 项离线测试通过，覆盖分词器转换、奖励、数据处理、真实 PPO 更新及模型保存/加载。

安装、短训练验证、完整训练及结果文件见 [PPO README](chapter5-supplement/rlhf-ppo/README.md)。

## 参考资料

- 官方讲义与代码：
  - Assignment 1（Basics）：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
  - Assignment 2（Systems）：[stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems)
  - Assignment 3（Scaling）：[stanford-cs336/assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling)
  - Assignment 4：暂不做
  - Assignment 5（Alignment）：[stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)
- 学习思路与代码参考：[weiruihhh/cs336_note_and_hw](https://github.com/weiruihhh/cs336_note_and_hw)——本仓库的作业学习与实现参考了该作者的 CS336 学习记录（笔记 + 作业代码）
- 扩展练习参考实现：
  - Llama 2：[hkproj/pytorch-llama](https://github.com/hkproj/pytorch-llama)
  - DeepSeek-V3 MLA：[VizuaraAILabs/DeepSeek-From-Scratch](https://github.com/VizuaraAILabs/DeepSeek-From-Scratch)
  - DeepSeek-V3 并行 / MoE：[hkproj/torchfeather](https://github.com/hkproj/torchfeather)
  - RLHF / PPO：[hkproj/rlhf-ppo](https://github.com/hkproj/rlhf-ppo)
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
