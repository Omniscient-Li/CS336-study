"""Create a report and plot from completed experiments, without rerunning models."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/a100_20261013"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    base = load(RUN / "hw3/results.metrics.json")
    sft = load(RUN / "hw4/sft_history.json")
    expert = load(RUN / "hw5/expert_history.json")
    grpo = load(RUN / "hw7/grpo_history.json")
    run = load(RUN / "hw7_run.json")
    assert run["exit_code"] == 0, "GRPO did not finish successfully"
    assert len(grpo["train"]) == 40 and grpo["val"][-1]["step"] == 20
    assert all(math.isfinite(row[key]) for row in grpo["train"]
               for key in ("loss", "grad_norm", "token_entropy", "reward_mean", "clip_fraction"))
    last = grpo["val"][-1]
    final_format = sum(x["scores"]["format_reward"] for x in last["examples"]) / len(last["examples"])
    final_truncation = sum(x["finish_reason"] == "length" for x in last["examples"]) / len(last["examples"])
    assert len(last["examples"]) == 128
    with (RUN / "hw7_gpu_usage.csv").open(encoding="utf-8") as source:
        usage = list(csv.DictReader(source))
    peak = max(float(x["memory_used_mib"]) for x in usage)
    stages = [
        {"stage": "hw3 / Base", **base},
        {"stage": "hw4 / SFT", **sft["after"]},
        {"stage": "hw5 / Expert iteration", **expert["val"][-1]},
        {"stage": "hw7 / SFT + GRPO", "n": len(last["examples"]),
         "accuracy": last["mean_reward"], "format_rate": final_format, "truncation_rate": final_truncation},
    ]
    initial = {x["question"]: x["reward"] for x in grpo["val"][0]["examples"]}
    improved = sum(initial[x["question"]] == 0 and x["reward"] == 1 for x in last["examples"])
    regressed = sum(initial[x["question"]] == 1 and x["reward"] == 0 for x in last["examples"])
    summary = {"stages": stages, "grpo_validation": [{"step": x["step"], "accuracy": x["mean_reward"]} for x in grpo["val"]],
               "grpo_elapsed_seconds": run["elapsed_seconds"], "peak_gpu_memory_mib": peak,
               "improved_vs_sft": improved, "regressed_vs_sft": regressed,
               "optimizer_updates": len(grpo["train"]), "rollouts": 20 * 64}
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Chapter 5：A100 实验结果（2026-10-13）", "",
             "设备：NVIDIA A100-SXM4-80GB；模型：Qwen2.5-Math-1.5B；seed=42。",
             "验证统一使用 `MATH/val.jsonl` 的前 128 题，温度为 0，最多生成 1024 tokens；主指标为官方 `r1_zero_reward_fn` 的严格奖励（格式和答案都正确）。",
             "", "| 作业 | 运行规模 | 严格准确率 | 格式合格率 | 输出截断率 |",
             "|---|---|---:|---:|---:|"]
    scales = ["128 道验证题", "512 条候选示范，50 次更新，microbatch=1、累积=4",
              "2 轮；每轮 64 题×4 回答、30 次 SFT 更新", "20 轮×64 回答；group=8；每批训练 2 次，共 40 次更新"]
    for stage, scale in zip(stages, scales):
        lines.append(f"| {stage['stage']} | {scale} | {stage['accuracy']:.2%} | {stage['format_rate']:.2%} | {stage['truncation_rate']:.2%} |")
    lines += ["", "hw5 从基础模型独立训练；hw7 从 hw4 的 SFT 检查点开始训练。",
              f"hw5 两轮筛选出 {expert['train'][0]['correct_completions']} 和 {expert['train'][1]['correct_completions']} 条正确回答。",
              f"hw7 相对 SFT 有 {improved} 题由错变对、{regressed} 题由对变错。",
              f"hw7 用时 {run['elapsed_seconds']/60:.1f} 分钟，采样监测到的峰值显存 {peak/1024:.2f} GiB。",
              "", "本次是有限步数的实际训练实验，验证集只有 128 题、单个随机种子；不能把结果当作完整 MATH 基准成绩或稳定的算法优劣结论。",
              "", "## 数据", "",
              "`train.jsonl` 有 4000 题，`val.jsonl` 有 1000 题，二者无重叠。`validation.jsonl` 是两者合并的 5000 题，不能用作训练后的独立验证集。",
              "原始文件只有题目和最终答案；`train_sft.jsonl` 按题目精确匹配补齐了完整解题过程，来源与版本记录在 `MATH/solution_source.json`。长于 2048 tokens 的 SFT 示范被跳过，避免截掉答案。",
              "", "## 运行与检查", "",
              "官方 `tests/test_sft.py` 和 `tests/test_grpo.py`：24 项通过。GRPO 40 次更新的损失、梯度及熵均已检查为有限值。",
              "所有运行输出位于 A100 的 `/home/shadeform/projects/lsn-cs336/chapter5/runs/a100_20261013`。模型检查点保存在各作业的 `checkpoint` 目录，完整生成结果与训练记录保存在 JSON/JSONL 文件。",
              "", "hw7 实际运行命令（在 A100 的 `chapter5` 目录）：", "", "```bash",
              "CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 VLLM_USE_V1=0 \\",
              "  /home/shadeform/cs336/assignment5-alignment/.venv/bin/python \\",
              "  hw7/run_grpo_train_loop.py \\",
              "  --model_path runs/a100_20261013/hw4/checkpoint \\",
              "  --output_dir runs/a100_20261013/hw7_new \\",
              "  --n_grpo_steps 20 --learning_rate 1e-5 \\",
              "  --rollout_batch_size 64 --group_size 8 --train_batch_size 64 \\",
              "  --gradient_accumulation_steps 64 --epochs_per_rollout_batch 2 \\",
              "  --loss_type grpo_clip --reward_mode strict \\",
              "  --val_size 128 --val_every_steps 5 --sampling_max_tokens 1024 --seed 42",
              "```", "", "使用新输出目录，保留这次结果；环境只复用现有依赖，文件写入我们自己的项目。"]
    (ROOT / "A100_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Report complete; matplotlib unavailable")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    values = [x["accuracy"] * 100 for x in stages]
    axes[0].bar(["Base", "SFT", "Expert", "SFT+GRPO"], values, color=["#94a3b8", "#3b82f6", "#14b8a6", "#8b5cf6"])
    for i, value in enumerate(values):
        axes[0].text(i, value + 1, f"{value:.1f}%", ha="center")
    axes[0].set_ylim(0, max(values) + 12)
    axes[0].set_ylabel("Strict accuracy (%)")
    axes[0].set_title("128 held-out MATH problems")
    axes[1].plot([x["step"] for x in grpo["val"]], [x["mean_reward"] * 100 for x in grpo["val"]], marker="o", color="#8b5cf6")
    axes[1].axhline(sft["after"]["accuracy"] * 100, linestyle="--", color="#3b82f6", label="SFT baseline")
    axes[1].set_xlabel("GRPO rollout step")
    axes[1].set_ylabel("Strict accuracy (%)")
    axes[1].set_title("GRPO validation trajectory")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RUN / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
