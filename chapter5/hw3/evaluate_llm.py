import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "assignment5-alignment"))

from vllm import LLM , SamplingParams
from typing import Callable , List
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

def evaluate_vllm(
        vllm_model : LLM ,
        reward_fn : Callable[[str , str] , dict[str , float]] ,
        prompts : List[str] ,
        eval_sampling_params : SamplingParams ,
        ground_truths : List[str] ,
        output_file : str
  ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # 1. 批量推理
    # outputs 是 RequestOutput 对象列表
    outputs = vllm_model.generate(prompts , eval_sampling_params)

    results = []
    correct_count = 0
    format_error_count = 0

    # 2. 遍历结果并评分
    for i , output_obj in enumerate(outputs):
        generated_text = output_obj.outputs[0].text
        ground_truth = ground_truths[i]

        # 调用提供的评分函数解析答案并打分
        scores = reward_fn(generated_text , ground_truth)

        result_entry = {
            "prompt": prompts[i],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "scores": scores
        }
        results.append(result_entry)

    # 3. 保存结果到磁盘
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

def main():
    import argparse
    from a100_utils import evaluate, init_vllm, load_jsonl
    parser = argparse.ArgumentParser(description="Evaluate Qwen on held-out MATH")
    parser.add_argument("--model_path", default=str(project_root / "models/Qwen2.5-Math-1.5B"))
    parser.add_argument("--val_file", default=str(project_root / "MATH/val.jsonl"))
    parser.add_argument("--output_file", default=str(project_root / "hw3/results.jsonl"))
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()
    examples = load_jsonl(args.val_file)
    if args.val_size > 0:
        examples = examples[:args.val_size]
    llm = init_vllm(args.model_path, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)
    metrics = evaluate(llm, examples, args.output_file, args.max_tokens)
    Path(args.output_file).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
