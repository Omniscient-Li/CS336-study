"""Shared single-A100 inference and evaluation utilities."""

import json
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "assignment5-alignment"))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

PROMPTS_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant "
    "solves it. The Assistant first thinks about the reasoning process in the mind and then "
    "provides the User with the answer. The reasoning process is enclosed within <think> </think> "
    "and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning "
    "process here </think> <answer> answer here </answer>.\n"
    "User: {question}\nAssistant: <think>"
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def check_disjoint(train, val):
    overlap = {x["problem"] for x in train} & {x["problem"] for x in val}
    if overlap:
        raise ValueError(f"Train/validation leakage: {len(overlap)} shared problems")


def init_vllm(model_id, device="cuda:0", seed=42, gpu_memory_utilization=0.25):
    if device not in ("cuda", "cuda:0"):
        raise ValueError("Select the physical GPU with CUDA_VISIBLE_DEVICES and use cuda:0.")
    return LLM(
        model=model_id, seed=seed, dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096, max_num_seqs=64, enforce_eager=True,
        enable_prefix_caching=False,
    )


def load_policy_into_vllm_instance(policy, llm):
    # Assignment 5 uses vLLM 0.7.2's in-process engine.
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    model.load_weights(policy.state_dict().items())


def evaluate(llm, examples, output_file, max_tokens=1024, temperature=0.0):
    if not examples:
        raise ValueError("Empty evaluation dataset")
    prompts = [PROMPTS_TEMPLATE.format(question=x["problem"]) for x in examples]
    outputs = llm.generate(prompts, SamplingParams(
        temperature=temperature, max_tokens=max_tokens,
        stop=["</answer>"], include_stop_str_in_output=True,
    ))
    results = []
    for example, prompt, output in zip(examples, prompts, outputs):
        text = output.outputs[0].text
        results.append({
            **example, "prompt": prompt, "generated_text": text,
            "scores": r1_zero_reward_fn(text, example["answer"]),
            "boxed_scores": question_only_reward_fn(text, example["answer"]),
            "finish_reason": output.outputs[0].finish_reason,
        })
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
    metrics = {"n": len(results)}
    for key, scores, field in [
        ("accuracy", "scores", "reward"), ("format_rate", "scores", "format_reward"),
        ("boxed_accuracy", "boxed_scores", "reward"),
    ]:
        metrics[key] = sum(x[scores][field] for x in results) / len(results)
    metrics["truncation_rate"] = sum(x["finish_reason"] == "length" for x in results) / len(results)
    print("EVAL", json.dumps(metrics), flush=True)
    return metrics
