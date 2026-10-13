"""Fine-tune GPT-2 for positive IMDB sentiment with TRL 1.12's PPO API."""

import argparse
import copy
import importlib.metadata
import os
import random

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput


# PPO was removed in TRL 1.13. Pin the last release that includes it.
TRL_VERSION = "1.12.0"


class SentimentBackbone(nn.Module):
    """Translate policy tokens into the reward tokenizer's vocabulary."""

    def __init__(self, classifier, policy_tokenizer, reward_tokenizer, positive_label_id):
        super().__init__()
        self.classifier = classifier
        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.positive_label_id = positive_label_id

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # TRL replaces padded input IDs with zero, so use its mask to remove padding.
        if attention_mask is None:
            attention_mask = input_ids.ne(self.policy_tokenizer.pad_token_id)
        rows = input_ids.detach().cpu()
        masks = attention_mask.detach().cpu().bool()
        texts = [
            self.policy_tokenizer.decode(row[mask].tolist(), skip_special_tokens=True)
            for row, mask in zip(rows, masks)
        ]
        inputs = self.reward_tokenizer(
            texts, padding=True, truncation=True, return_token_type_ids=False,
            max_length=self.classifier.config.max_position_embeddings,
            return_tensors="pt",
        ).to(next(self.classifier.parameters()).device)
        with torch.no_grad():
            scores = self.classifier(**inputs).logits[:, self.positive_label_id]
        # TRL's get_reward selects the scalar at the last response token.
        # This classifier scores the whole text, so expose that score at each position.
        token_scores = scores[:, None, None].expand(-1, input_ids.shape[1], 1)
        return BaseModelOutput(hidden_states=(token_scores,))


class SentimentRewardModel(nn.Module):
    """Provide TRL's backbone + score interface without replacing the IMDB head."""

    base_model_prefix = "backbone"

    def __init__(self, classifier, policy_tokenizer, reward_tokenizer, positive_label_id=None):
        super().__init__()
        if positive_label_id is None:
            positive_ids = [
                int(index) for index, label in classifier.config.id2label.items()
                if str(label).upper() == "POSITIVE"
            ]
            if len(positive_ids) != 1:
                raise ValueError(
                    "Reward model must identify a POSITIVE label; otherwise pass "
                    "--positive-label-id explicitly. Labels: "
                    + str(classifier.config.id2label)
                )
            positive_label_id = positive_ids[0]
        if not 0 <= positive_label_id < classifier.config.num_labels:
            raise ValueError("positive_label_id is outside the classifier's label range.")
        self.config = classifier.config
        self.backbone = SentimentBackbone(
            classifier, policy_tokenizer, reward_tokenizer, positive_label_id,
        )
        self.score = nn.Identity()
        self.requires_grad_(False)
        self.eval()


def build_dataset(tokenizer, dataset_name="imdb", input_min_text_length=2,
                  input_max_text_length=8, seed=42, max_samples=None):
    """Use short review prefixes as prompts, with held-out evaluation prompts."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    # Preserve the original example's filter: >200 characters, not tokens.
    ds = ds.filter(lambda sample: len(sample["text"]) > 200)
    ds = ds.shuffle(seed=seed)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    rng = random.Random(seed)

    def tokenize(sample):
        # Original LengthSampler(2, 8) sampled from [2, 8), i.e. 2..7 tokens.
        length = rng.randrange(input_min_text_length, input_max_text_length)
        ids = tokenizer.encode(sample["text"], add_special_tokens=False)[:length]
        return {"input_ids": ids}

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.filter(lambda sample: len(sample["input_ids"]) > 0)
    if len(ds) < 2:
        raise ValueError("At least two usable reviews are required.")
    return ds.train_test_split(test_size=min(128, max(1, len(ds) // 10)), seed=seed)


def make_config(args, bf16):
    from trl.experimental.ppo import PPOConfig

    return PPOConfig(
        output_dir=args.output_dir,
        sft_model_path=args.model_name,
        reward_model_path=args.reward_model_name,
        learning_rate=args.learning_rate,
        total_episodes=args.total_episodes,
        num_train_epochs=1.0,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_mini_batches=1,
        local_rollout_forward_batch_size=args.rollout_batch_size,
        per_device_eval_batch_size=4,
        num_ppo_epochs=args.ppo_epochs,
        response_length=args.response_length,
        temperature=1.0,
        stop_token="eos",
        # Short sentiment continuations need not contain EOS; do not penalize them.
        missing_eos_penalty=None,
        kl_coef=0.2,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=False,
        report_to=args.report_to,
        logging_steps=1,
        save_strategy="no",
        num_sample_generations=0,
        seed=args.seed,
        push_to_hub=False,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="lvwerra/gpt2-imdb")
    parser.add_argument("--reward-model-name", default="lvwerra/distilbert-imdb")
    parser.add_argument("--dataset-name", default="imdb")
    parser.add_argument("--output-dir", default="gpt2-imdb-pos-v2")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="Default: one pass over training prompts; rounded up to a rollout batch.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit reviews before the train/eval split for a short validation run.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="PPO optimization microbatch size per GPU.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--response-length", type=int, default=16)
    parser.add_argument("--input-min-length", type=int, default=2)
    parser.add_argument("--input-max-length", type=int, default=8,
                        help="Exclusive upper bound for prompt length.")
    parser.add_argument("--learning-rate", type=float, default=1.41e-5)
    parser.add_argument("--positive-label-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    args = parser.parse_args(argv)
    for name in ("batch_size", "gradient_accumulation_steps", "rollout_batch_size",
                 "ppo_epochs", "response_length", "learning_rate"):
        if getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " must be positive")
    for name in ("total_episodes", "max_samples"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " must be positive")
    if not 0 < args.input_min_length < args.input_max_length:
        parser.error("require 0 < input-min-length < input-max-length")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        installed = importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        installed = "not installed"
    if installed != TRL_VERSION:
        raise SystemExit(
            f"This script requires trl=={TRL_VERSION} (installed: {installed}). "
            "Create the modern environment using the README installation commands."
        )
    from trl.experimental.ppo import PPOTrainer

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.precision == "bf16" and not supports_bf16:
        raise SystemExit("--precision bf16 requires a BF16-capable CUDA GPU (such as A100).")
    bf16 = supports_bf16 if args.precision == "auto" else args.precision == "bf16"
    # PPOTrainer creates Accelerator internally; align its autocast with PPOConfig.
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16" if bf16 else "no"
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    # PAD must differ from EOS: TRL uses PAD IDs to construct response masks.
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    # Copy after resizing so the reference starts with exactly the same weights.
    ref_model = copy.deepcopy(model).requires_grad_(False).eval()
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1,
        id2label={0: "VALUE"}, label2id={"VALUE": 0},
    )
    value_model.resize_token_embeddings(len(tokenizer))
    value_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.use_cache = False

    classifier = AutoModelForSequenceClassification.from_pretrained(args.reward_model_name)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    reward_model = SentimentRewardModel(
        classifier, tokenizer, reward_tokenizer, args.positive_label_id,
    )
    datasets = build_dataset(
        tokenizer, args.dataset_name, args.input_min_length, args.input_max_length,
        args.seed, args.max_samples,
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rollout_size = args.batch_size * args.gradient_accumulation_steps * world_size
    if len(datasets["train"]) < rollout_size:
        raise ValueError(
            f"Need at least {rollout_size} training prompts for one rollout; "
            f"got {len(datasets['train'])}. Increase --max-samples or reduce "
            "--batch-size / --gradient-accumulation-steps."
        )

    trainer = PPOTrainer(
        args=make_config(args, bf16),
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
    )
    trainer.accelerator.print(
        f"Device: {trainer.accelerator.device}; BF16: {bf16}; "
        f"rollout batch: {trainer.args.batch_size}; "
        f"PPO updates: {trainer.args.num_total_batches}"
    )
    # New PPOTrainer handles generation, rewards, GAE and PPO updates internally.
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.print(f"Saved policy and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
