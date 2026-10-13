"""Response-only SFT and held-out MATH evaluation on one A100."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from a100_utils import check_disjoint, evaluate, init_vllm, load_jsonl, load_policy_into_vllm_instance
from sft_utils import SFTDataset, collate_fn, sft_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=str(project_root / "models/Qwen2.5-Math-1.5B"))
    parser.add_argument("--train_file", default=str(project_root / "MATH/train_sft.jsonl"))
    parser.add_argument("--val_file", default=str(project_root / "MATH/val.jsonl"))
    parser.add_argument("--output_dir", default=str(project_root / "hw4/outputs"))
    parser.add_argument("--total_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val = load_jsonl(args.train_file), load_jsonl(args.val_file)
    check_disjoint(train, val)
    random.shuffle(train)
    if args.train_size > 0:
        train = train[:args.train_size]
    if args.val_size > 0:
        val = val[:args.val_size]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = init_vllm(args.model_path, seed=args.seed)
    before = evaluate(llm, val, output / "before.jsonl", args.max_tokens)
    policy = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda:0")
    history = sft_train(policy, train, tokenizer, args.total_steps, args.batch_size,
                        args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
                        max_length=args.max_length)
    load_policy_into_vllm_instance(policy, llm)
    after = evaluate(llm, val, output / "after.jsonl", args.max_tokens)
    checkpoint = output / "checkpoint"
    policy.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    metrics = {"config": vars(args), "before": before, "after": after, "train": history}
    (output / "sft_history.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
