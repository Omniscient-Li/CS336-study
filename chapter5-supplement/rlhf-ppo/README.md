# IMDB sentiment PPO（新版 TRL）

参考项目：[hkproj/rlhf-ppo](https://github.com/hkproj/rlhf-ppo)。
本目录基于其 IMDB 情感 PPO 示例学习和迁移，保留原仓库的
`Slides.pdf` 与 `hugging-face-code-commented/` 教学资料；
其中 TRL 参考源码保留了 Hugging Face 的版权与 Apache-2.0 声明。

`gpt_sentiment.py` 使用 **TRL 1.12.0** 的
`trl.experimental.ppo.PPOConfig / PPOTrainer` 接口，
目标仍然是让 `lvwerra/gpt2-imdb` 生成更正面的影评。

查询发行包时，最新 TRL 是 1.15.0；**从 1.13.0 起已移除 PPO**。
因此这里固定最后包含 PPO 的 1.12.0，而不是直接安装最新版。
源码参考：[TRL v1.12.0](https://github.com/huggingface/trl/tree/v1.12.0/trl/experimental/ppo)。

## 与旧脚本的对应关系

| 旧版 0.7.10 | 现在 |
| --- | --- |
| `AutoModelForCausalLMWithValueHead` | 独立的 GPT-2 policy 和单输出 value model |
| 手写生成、奖励、`step()` 循环 | `trainer.train()` 内部执行 rollout、GAE、PPO |
| `LengthSampler` | 固定种子采样 2–7 token 的影评前缀 |
| `sentiment_pipe` 取 POSITIVE logit | 冻结的 DistilBERT + 分词器转换适配器，仍取 POSITIVE 原始 logit |
| `pad_token = eos_token` | 独立 PAD token + 左 padding，保留 EOS 的有效性 |
| 默认要求 W&B | 默认终端日志；可用 `--report-to wandb` 开启 |

GPT-2 和 DistilBERT 的词表不同，因此奖励适配器先去掉 padding，
把 prompt + response 解码成文本，再用 DistilBERT 的 tokenizer 重新编码。
不会把 GPT-2 token ID 直接传给 DistilBERT，也不会重置已训练好的情感分类头。

新训练器固定最多生成 16 token，遇 EOS 截断；旧脚本随机生成 4–15 token。
默认 KL 系数为 0.2，原例学习率为 1.41e-5。
默认每次 rollout 64 个样本（微批 16 × 梯度累积 4），每批做 4 轮 PPO。
`total_episodes` 会向上取整到整批；未指定时训练一个数据集 epoch。
最终目录只保存用于推理的 policy 和 tokenizer。

`hugging-face-code-commented/` 仍是原仓库 **TRL 0.7.10** 的教学注释源码，
不参与当前脚本运行，其行号、接口和新版实现不同。
本地原有 `.venv` 是旧环境，请新建下面的环境。

## A100 安装（Linux，单卡）

在 A100 实例终端中，进入包含本文件的目录：

```bash
cd chapter5-supplement/rlhf-ppo
nvidia-smi
uv venv --python 3.11 .venv-modern
source .venv-modern/bin/activate

# cu118 对驱动版本要求较低；A100 支持 BF16。
uv pip install --python .venv-modern/bin/python torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv-modern/bin/python -r requirements.txt \
  --index-url https://pypi.org/simple

python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()"
```

首次实际训练需要下载公开的 GPT-2、DistilBERT 和 IMDB 数据集。
不需要 Hugging Face 或 W&B 登录。
如果实例已有 CUDA PyTorch，仍建议使用独立环境，避免覆盖其他实验。

## 先验证，再训练

本地离线测试使用随机初始化的小模型，不下载数据或模型：

```bash
python -m unittest discover -s tests -v
```

A100 上先用真实模型跑两次 PPO 更新，检查下载、BF16、奖励和模型保存：

```bash
python gpt_sentiment.py \
  --precision bf16 \
  --max-samples 256 \
  --total-episodes 32 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --rollout-batch-size 8 \
  --ppo-epochs 1 \
  --output-dir outputs/smoke
```

通过后运行完整训练（A100 40GB / 80GB 单卡）：

```bash
python gpt_sentiment.py --precision bf16 --output-dir gpt2-imdb-pos-v2
```

如果显存不足，先减小 `--batch-size` 和 `--rollout-batch-size`，
并增大 `--gradient-accumulation-steps` 保持 rollout 总样本数。
脚本默认关闭梯度检查点；GPT-2 短序列无需该功能。

可选 W&B：

```bash
uv pip install --python .venv-modern/bin/python 'wandb>=0.16,<1'
wandb login
python gpt_sentiment.py --precision bf16 --report-to wandb
```

## Windows 本地验证

这台机器已有旧版环境；不要用它执行新版脚本。可创建 CPU 验证环境：

```powershell
uv venv --python 3.11 .venv-modern
uv pip install --python .venv-modern/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-modern/Scripts/python.exe -r requirements.txt --index-url https://pypi.org/simple
& .venv-modern/Scripts/python.exe -m unittest discover -s tests -v
& .venv-modern/Scripts/python.exe gpt_sentiment.py --help
```

离线测试验证奖励分词转换、POSITIVE logit、padding、数据处理及一次真实
PPO 反向传播和保存；5 项全部通过。A100 BF16 完整训练结果见下文。

## A100 完整训练结果（2026-10-09）

已在 **NVIDIA A100-SXM4-80GB** 上完成训练和测试集评估：
Python 3.11、PyTorch 2.7.1+cu118、TRL 1.12.0，BF16 autocast。

- IMDB 训练集共 25,000 条；按原例过滤到 24,895 条（影评超过 200 字符）。
- 留出 128 条提示用于检查；实际训练提示 24,767 条。
- rollout batch 64，每批 4 轮 PPO，**387 次更新**，实际处理 24,768 个 episode（末批向上取整）。
- 耗时 **1,043 秒（17 分 23 秒）**；训练与评估退出码均为 0，全部训练指标有限。
- 最终 policy 保存成功，评估时重新加载并生成；模型权重约 498 MB，未提交至 Git。

在 IMDB **test split** 抽取 256 条未参与训练的提示，固定种子和相同采样参数比较初始模型与训练后模型：

| 指标 | 初始 GPT-2 | PPO 后 |
| --- | ---: | ---: |
| 平均 POSITIVE 原始 logit | 0.247 | 2.249 |
| 平均正面概率 | 56.3% | 93.8% |
| 判定为正面的比例 | 55.9% | 95.3% |

该评估使用训练时的同一 DistilBERT 情感分类器，每个提示采样一次；
它反映情感奖励目标的改善，不是独立的语言质量评测。

记录保存在 [results/a100-20261009](results/a100-20261009/)：
[运行总结](results/a100-20261009/run_summary.json)、
[评估指标](results/a100-20261009/evaluation.json)、
[256 组生成样例](results/a100-20261009/evaluation_samples.json)。
