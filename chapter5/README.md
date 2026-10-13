# Chapter 5 · Alignment

hw3、hw4、hw5、hw7 已在单张 NVIDIA A100-SXM4-80GB 上运行（2026-10-13），模型为 Qwen2.5-Math-1.5B。

| 作业 | 内容 | 本次运行 | 严格准确率 |
|---|---|---|---:|
| [hw3](hw3/evaluate_llm.py) | vLLM 基础模型评估 | 128 道独立验证题 | 13.28% |
| [hw4](hw4/SFT_experiment.py) | Response-only SFT | 50 次更新，512 条候选示范 | 46.88% |
| [hw5](hw5/expert_iteration.py) | 采样、筛选正确回答、专家迭代 SFT | 2 轮，每轮 30 次 SFT 更新 | 34.38% |
| [hw7](hw7/run_grpo_train_loop.py) | GRPO-Clip | 从 SFT 开始，20 轮、40 次更新 | 47.66% |

准确率使用同一批 `val.jsonl` 前 128 题，温度 0、生成上限 1024 tokens，以官方 `r1_zero_reward_fn` 判断格式和答案都正确。hw5 从基础模型独立开始，hw7 从 hw4 的检查点开始。有限步数、单种子的结果不能代表完整基准或稳定算法排名。

配置、数据口径和运行记录见 [A100 报告](A100_RESULTS.md)。[结果目录](runs/a100_20261013/) 包含指标、训练日志、逐题生成、GPU 用量和 [对比图](runs/a100_20261013/comparison.png)。官方 SFT/GRPO 数值测试：**24 passed**。

## 数据

- `MATH/train.jsonl`：4000 题；`MATH/val.jsonl`：1000 题，两者不重叠。
- `MATH/validation.jsonl`：上述两者合并的 5000 题，包含训练题，不能用于训练后的独立验证。
- `MATH/train_sft.jsonl`：训练题加完整解题过程，按题目精确匹配公开 [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)；源版本见 [solution_source.json](MATH/solution_source.json)。原始拆分来自学习参考仓库 [weiruihhh/cs336_note_and_hw](https://github.com/weiruihhh/cs336_note_and_hw)。

重新补齐解题过程：`python prepare_math_sft.py`。SFT 跳过超过 2048 tokens 的示范，避免截掉最终答案。

## A100 安装与运行

Linux，Python 3.12。在仓库的 `chapter5` 目录创建独立环境；PPO 使用另外的环境。

```bash
uv venv --python 3.12 .venv-alignment
uv pip install --python .venv-alignment/bin/python -r requirements-a100.txt
source .venv-alignment/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
nvidia-smi
```

下载基础模型，或通过 `--model_path` 指定已有的模型目录：

```bash
huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
  --revision 4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2 \
  --local-dir models/Qwen2.5-Math-1.5B

python hw3/evaluate_llm.py --output_file runs/my_run/hw3/results.jsonl
python hw4/SFT_experiment.py --output_dir runs/my_run/hw4
python hw5/expert_iteration.py --output_dir runs/my_run/hw5
python hw7/run_grpo_train_loop.py \
  --model_path runs/my_run/hw4/checkpoint --output_dir runs/my_run/hw7 \
  --n_grpo_steps 20 --rollout_batch_size 64 --group_size 8 \
  --train_batch_size 64 --gradient_accumulation_steps 64 \
  --epochs_per_rollout_batch 2 --loss_type grpo_clip --reward_mode strict \
  --val_size 128 --val_every_steps 5 --sampling_max_tokens 1024
```

依次执行，使用空闲 GPU；默认 `cuda:0` 是 `CUDA_VISIBLE_DEVICES` 选中的卡。所有输出使用项目内的新目录。GitHub 保存代码、数据和结果记录；模型权重、环境和缓存不入库。

数值测试（先准备模型）：

```bash
cd assignment5-alignment
CS336_TEST_MODEL=../models/Qwen2.5-Math-1.5B CUDA_VISIBLE_DEVICES='' \
  ../.venv-alignment/bin/python -m pytest tests/test_sft.py tests/test_grpo.py -q
```

`assignment5-alignment/` 保留运行所需的官方 grader、提示词和 SFT/GRPO 数值测试，来源为 [stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment)。适配器接入本章实现；这里没有完整收录可选 DPO、安全对齐等作业。

## RLHF / PPO 扩展练习

[rlhf-ppo](rlhf-ppo/README.md) 用 GPT-2 policy、DistilBERT 情感奖励和 TRL 1.12.0 实现 IMDB PPO。2026-10-09 完成 387 次 PPO 更新，在 IMDB test 的 256 个提示上，正面分类比例 **55.86% → 95.31%**。配置、证据与局限见 [PPO A100 报告](rlhf-ppo/A100_RESULTS.md)。
