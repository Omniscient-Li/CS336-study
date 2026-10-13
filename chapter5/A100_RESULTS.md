# Chapter 5：A100 实验结果（2026-10-13）

设备：NVIDIA A100-SXM4-80GB；模型：Qwen2.5-Math-1.5B；seed=42。
验证统一使用 `MATH/val.jsonl` 的前 128 题，温度为 0，最多生成 1024 tokens；主指标为官方 `r1_zero_reward_fn` 的严格奖励（格式和答案都正确）。

| 作业 | 运行规模 | 严格准确率 | 格式合格率 | 输出截断率 |
|---|---|---:|---:|---:|
| hw3 / Base | 128 道验证题 | 13.28% | 42.97% | 10.16% |
| hw4 / SFT | 512 条候选示范，50 次更新，microbatch=1、累积=4 | 46.88% | 92.19% | 3.12% |
| hw5 / Expert iteration | 2 轮；每轮 64 题×4 回答、30 次 SFT 更新 | 34.38% | 87.50% | 5.47% |
| hw7 / SFT + GRPO | 20 轮×64 回答；group=8；每批训练 2 次，共 40 次更新 | 47.66% | 91.41% | 3.12% |

hw5 从基础模型独立训练；hw7 从 hw4 的 SFT 检查点开始训练。
hw5 两轮筛选出 12 和 36 条正确回答。
hw7 相对 SFT 有 7 题由错变对、6 题由对变错。
hw7 用时 13.9 分钟，采样监测到的峰值显存 44.08 GiB。

本次是有限步数的实际训练实验，验证集只有 128 题、单个随机种子；不能把结果当作完整 MATH 基准成绩或稳定的算法优劣结论。

## 数据

`train.jsonl` 有 4000 题，`val.jsonl` 有 1000 题，二者无重叠。`validation.jsonl` 是两者合并的 5000 题，不能用作训练后的独立验证集。
原始文件只有题目和最终答案；`train_sft.jsonl` 按题目精确匹配补齐了完整解题过程，来源与版本记录在 `MATH/solution_source.json`。长于 2048 tokens 的 SFT 示范被跳过，避免截掉答案。

## 运行与检查

官方 `tests/test_sft.py` 和 `tests/test_grpo.py`：24 项通过。GRPO 40 次更新的损失、梯度及熵均已检查为有限值。
所有运行输出位于 A100 的 `/home/shadeform/projects/lsn-cs336/chapter5/runs/a100_20261013`。模型检查点保存在各作业的 `checkpoint` 目录，完整生成结果与训练记录保存在 JSON/JSONL 文件。

hw7 实际运行命令（在 A100 的 `chapter5` 目录）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 VLLM_USE_V1=0 \
  /home/shadeform/cs336/assignment5-alignment/.venv/bin/python \
  hw7/run_grpo_train_loop.py \
  --model_path runs/a100_20261013/hw4/checkpoint \
  --output_dir runs/a100_20261013/hw7_new \
  --n_grpo_steps 20 --learning_rate 1e-5 \
  --rollout_batch_size 64 --group_size 8 --train_batch_size 64 \
  --gradient_accumulation_steps 64 --epochs_per_rollout_batch 2 \
  --loss_type grpo_clip --reward_mode strict \
  --val_size 128 --val_every_steps 5 --sampling_max_tokens 1024 --seed 42
```

使用新输出目录，保留这次结果；环境只复用现有依赖，文件写入我们自己的项目。
