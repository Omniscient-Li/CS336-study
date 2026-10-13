# IMDB PPO：A100 实验结果（2026-10-09）

模型：`lvwerra/gpt2-imdb`；奖励：`lvwerra/distilbert-imdb` 的 POSITIVE 原始 logit。NVIDIA A100-SXM4-80GB，BF16，PyTorch 2.7.1+cu118、Transformers 4.57.6、TRL 1.12.0。

训练源码 SHA-256：`14b0104b6d658922e9e705d2dd9dc63c0a9b054b1fd0856fc4bb39bd9696d763`，与上传的 `gpt_sentiment.py` 一致。

## 训练

IMDB train 25000 条影评中，长度超过 200 字符的有 24895 条；留出 128 个提示，其余 24767 个用于训练。rollout batch=64，实际处理 24768 episodes（向上取整到整批），387 次更新，每批 4 轮 PPO，seed=42，生成上限 16 token。

用时 1043 秒（17 分 23 秒），训练和评估退出码均为 0，所有已记录训练指标有限。检查点和 tokenizer 保存在 A100 的我们项目目录 `chapter5-supplement/rlhf-ppo/outputs/imdb-full-20261009-123241/model`；GitHub 不包含权重。

## 测试结果

从 IMDB test 中长度超过 200 字符的影评，以 seed=2026 抽取 256 个提示，每个提示由 2–7 个 GPT-2 tokens 组成。两个模型分别以 seed=42、temperature=1、top_k=0、top_p=1、max_new_tokens=16 生成一次，再由 DistilBERT 评分。

| 指标 | 基础模型 | PPO 后 |
|---|---:|---:|
| POSITIVE logit 均值 | 0.2471 | 2.2491 |
| 正面概率均值 | 0.5628 | 0.9383 |
| 判为正面的比例（概率≥0.5） | 55.86% | 95.31% |

这说明生成更符合该奖励分类器的偏好；训练和评估使用同一分类器，每题只有单次随机采样，不能据此断言整体文本质量或人类偏好获得同等提升。

## 证据与复现

[结果目录](results/a100_20261009/) 包含 `run_summary.json`、`evaluation.json`、逐题 `evaluation_samples.json`、训练/评估日志、依赖版本、源码校验值与运行命令。`evaluate.py` 是原实验评估脚本，假定同目录存在 `model/`；重跑时将 policy 和 tokenizer 保存到该目录即可。

依赖安装见 [README](README.md)。从 `chapter5/rlhf-ppo` 运行：

```bash
source .venv-modern/bin/activate
CUDA_VISIBLE_DEVICES=0 python gpt_sentiment.py --precision bf16 \
  --output-dir results/my_run/model
cp results/a100_20261009/evaluate.py results/my_run/evaluate.py
python results/my_run/evaluate.py
```

原训练、验证读取公开模型和 IMDB 数据，不要求登录 Hugging Face 或 W&B。
