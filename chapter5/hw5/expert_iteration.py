"""Generate, filter and fine-tune Qwen on MATH using a single A100."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from a100_utils import (PROMPTS_TEMPLATE, init_vllm, load_policy_into_vllm_instance,
                        check_disjoint, evaluate, load_jsonl, r1_zero_reward_fn)
from sft_utils import SFTDataset, collate_fn, sft_train

def generate_rollouts(
    llm: LLM,
    questions: List[str],
    n_rollout: int,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> List[List[str]]:
    """
    为每个问题生成多个完成。

    Args:
        llm: vLLM实例
        questions: 问题列表
        n_rollout: 每个问题的完成数量
        temperature: 采样温度
        max_tokens: 最大令牌数

    Returns:
        包含n_rollout个完成的列表的列表
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        n=n_rollout,  # 为每个提示生成n_rollout个样本
        stop=["</answer>"],
        include_stop_str_in_output=True,
        min_tokens=10,  # 避免生成空字符串
    )

    # 格式化提示
    prompts = [PROMPTS_TEMPLATE.format(question=q) for q in questions]

    # 生成
    outputs = llm.generate(prompts, sampling_params)

    # 提取完成
    all_completions = []
    for output_obj in outputs:
        completions = [out.text for out in output_obj.outputs]
        all_completions.append(completions)

    return all_completions


def filter_correct_completions(
    questions: List[str],
    all_completions: List[List[str]],
    ground_truths: List[str],
) -> List[Dict[str, str]]:
    """
    过滤回复，只保留满足奖励值要求的回复。

    Args:
        questions: 问题列表
        all_completions: 包含回复列表的列表
        ground_truths: ground_truths答案列表

    Returns:
        包含'question'和'completion'键的列表
    """
    filtered_data = []

    for question, completions, gt in zip(questions, all_completions, ground_truths):
        for completion in completions:
            # 计算奖励
            scores = r1_zero_reward_fn(completion, gt)

            # 如果奖励值大于0，则保留
            if scores["reward"] > 0:
                filtered_data.append({
                    "question": question,
                    "completion": completion,
                })

    return filtered_data


def expert_iteration(model_path, train_file, val_file, n_rollout=4, mu_sft=30,
                     k_steps=2, batch_size=1, learning_rate=1e-5,
                     train_samples_per_iter=64, val_samples=128, seed=42,
                     output_dir=None, max_tokens=1024, gradient_accumulation_steps=4):
    random.seed(seed)
    torch.manual_seed(seed)
    train, val = load_jsonl(train_file), load_jsonl(val_file)
    check_disjoint(train, val)
    if val_samples > 0:
        val = val[:val_samples]
    output = Path(output_dir or project_root / "hw5/outputs")
    output.mkdir(parents=True, exist_ok=True)
    history = {"train": [], "val": [], "config": {
        "model_path": model_path, "n_rollout": n_rollout, "mu_sft": mu_sft,
        "k_steps": k_steps, "batch_size": batch_size, "learning_rate": learning_rate,
        "train_samples_per_iter": train_samples_per_iter, "val_samples": val_samples,
        "seed": seed, "max_tokens": max_tokens,
    }}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = init_vllm(model_path, seed=seed)
    initial = evaluate(llm, val, output / "before.jsonl", max_tokens)
    history["val"].append({"iteration": 0, **initial})
    policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda:0")
    for iteration in range(1, k_steps + 1):
        sampled = random.sample(train, min(train_samples_per_iter, len(train)))
        questions = [x["problem"] for x in sampled]
        completions = generate_rollouts(llm, questions, n_rollout, max_tokens=max_tokens)
        filtered = filter_correct_completions(questions, completions, [x["answer"] for x in sampled])
        record = {"iteration": iteration, "total_completions": len(sampled) * n_rollout,
                  "correct_completions": len(filtered)}
        print("EXPERT", json.dumps(record), flush=True)
        (output / f"filtered_iter_{iteration}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in filtered), encoding="utf-8")
        if filtered:
            record["sft"] = sft_train(policy, filtered, tokenizer, mu_sft, batch_size,
                                      learning_rate, gradient_accumulation_steps=gradient_accumulation_steps)
            load_policy_into_vllm_instance(policy, llm)
            metrics = evaluate(llm, val, output / f"after_iter_{iteration}.jsonl", max_tokens)
            history["val"].append({"iteration": iteration, **metrics})
        else:
            record["skipped"] = "No correct rollouts; no optimizer updates"
        history["train"].append(record)
        (output / "expert_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = output / "checkpoint"
    policy.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=str(project_root / "models/Qwen2.5-Math-1.5B"))
    parser.add_argument("--train_file", default=str(project_root / "MATH/train.jsonl"))
    parser.add_argument("--val_file", default=str(project_root / "MATH/val.jsonl"))
    parser.add_argument("--output_dir", default=str(project_root / "hw5/outputs"))
    parser.add_argument("--n_rollout", type=int, default=4)
    parser.add_argument("--mu_sft", type=int, default=30)
    parser.add_argument("--k_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_samples_per_iter", type=int, default=64)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    expert_iteration(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
