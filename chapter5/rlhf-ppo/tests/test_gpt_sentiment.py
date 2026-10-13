"""Offline checks including one real PPO update with tiny random models."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    PreTrainedTokenizerFast,
    set_seed,
)
from trl.experimental.ppo import PPOTrainer
from trl.experimental.utils import get_reward

from gpt_sentiment import SentimentRewardModel, build_dataset, make_config, parse_args


def tokenizer(words):
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        padding_side="left",
    )


class ModernPPOTests(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        torch.set_num_threads(1)
        self.policy_tokenizer = tokenizer(["[PAD]", "[UNK]", "[EOS]", "this", "movie", "good", "bad"])
        self.reward_tokenizer = tokenizer(["[PAD]", "[UNK]", "[EOS]", "bad", "good", "movie", "this"])
        self.classifier = DistilBertForSequenceClassification(DistilBertConfig(
            vocab_size=7, dim=16, hidden_dim=32, n_layers=1, n_heads=2,
            max_position_embeddings=32, num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )).eval()
        self.reward = SentimentRewardModel(
            self.classifier, self.policy_tokenizer, self.reward_tokenizer,
        )

    def test_reward_retokenizes_and_preserves_positive_logit(self):
        # TRL's reward helper masks left padding and calls backbone + score.
        ids = torch.tensor([[0, 3, 4, 5, 0], [3, 4, 6, 5, 6]])
        texts = ["this movie good", "this movie bad good bad"]
        expected_inputs = self.reward_tokenizer(texts, padding=True, return_token_type_ids=False, return_tensors="pt")
        with torch.no_grad():
            expected = self.classifier(**expected_inputs).logits[:, 1]
        _, actual, lengths = get_reward(self.reward, ids, self.policy_tokenizer.pad_token_id, 2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(lengths.tolist(), [3, 4])
        self.assertTrue(all(not p.requires_grad for p in self.reward.parameters()))

    def test_ambiguous_labels_require_explicit_positive_id(self):
        self.classifier.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
        with self.assertRaisesRegex(ValueError, "positive-label-id"):
            SentimentRewardModel(self.classifier, self.policy_tokenizer, self.reward_tokenizer)
        SentimentRewardModel(self.classifier, self.policy_tokenizer, self.reward_tokenizer, 1)

    def test_dataset_is_repeatable_and_contains_only_prompt_ids(self):
        ds = Dataset.from_dict({"text": ["this movie good " * 20] * 40, "label": [1] * 40})
        with patch("datasets.load_dataset", return_value=ds):
            first = build_dataset(self.policy_tokenizer, max_samples=20)
            second = build_dataset(self.policy_tokenizer, max_samples=20)
        self.assertEqual(first["train"].to_dict(), second["train"].to_dict())
        self.assertEqual(first["train"].column_names, ["input_ids"])
        self.assertTrue(all(2 <= len(ids) < 8 for ids in first["train"]["input_ids"]))
        self.assertEqual(len(first["train"]) + len(first["test"]), 20)
        with patch("datasets.load_dataset", return_value=ds):
            with self.assertRaisesRegex(ValueError, "At least two"):
                build_dataset(self.policy_tokenizer, max_samples=1)

    def test_one_real_ppo_update_and_saved_policy(self):
        config = GPT2Config(
            vocab_size=7, n_positions=32, n_embd=16, n_layer=1, n_head=2,
            pad_token_id=0, eos_token_id=2, bos_token_id=2,
        )
        policy = GPT2LMHeadModel(config)
        reference = GPT2LMHeadModel(config)
        reference.load_state_dict(policy.state_dict())
        reference.requires_grad_(False).eval()
        value = GPT2ForSequenceClassification(GPT2Config(**{**config.to_dict(), "num_labels": 1,
            "id2label": {0: "VALUE"}, "label2id": {"VALUE": 0}}))
        ds = Dataset.from_dict({"input_ids": [[3, 4], [3, 4, 5], [4, 6], [3, 5]]})
        policy_before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        value_before = {name: p.detach().clone() for name, p in value.named_parameters()}
        reference_before = {name: p.detach().clone() for name, p in reference.named_parameters()}
        reward_before = {name: p.detach().clone() for name, p in self.reward.named_parameters()}
        with tempfile.TemporaryDirectory() as output_dir:
            args = Namespace(
                output_dir=output_dir, model_name="tiny-offline", reward_model_name="tiny-offline",
                learning_rate=1e-3, total_episodes=4, batch_size=2,
                gradient_accumulation_steps=2, rollout_batch_size=2,
                ppo_epochs=1, response_length=4, report_to="none", seed=42,
            )
            ppo_config = make_config(args, bf16=False)
            ppo_config.use_cpu = True
            with patch.dict("os.environ", {"ACCELERATE_MIXED_PRECISION": "no"}):
                trainer = PPOTrainer(
                    args=ppo_config, processing_class=self.policy_tokenizer,
                    model=policy, ref_model=reference, reward_model=self.reward,
                    value_model=value, train_dataset=ds, eval_dataset=ds,
                )
                trainer.train()
                trainer.save_model(output_dir)
            self.assertEqual(trainer.state.global_step, 1)
            self.assertTrue(any(not torch.equal(policy_before[n], p) for n, p in policy.named_parameters()))
            self.assertTrue(any(not torch.equal(value_before[n], p) for n, p in value.named_parameters()))
            for name, p in reference.named_parameters():
                torch.testing.assert_close(p, reference_before[name])
            for name, p in self.reward.named_parameters():
                torch.testing.assert_close(p, reward_before[name])
            for log in trainer.state.log_history:
                for key in ("loss/policy_avg", "loss/value_avg", "objective/scores"):
                    if key in log:
                        self.assertTrue(torch.isfinite(torch.tensor(log[key])), key)
            reloaded = GPT2LMHeadModel.from_pretrained(output_dir)
            for name, p in policy.named_parameters():
                torch.testing.assert_close(p, dict(reloaded.named_parameters())[name])
            self.assertTrue((Path(output_dir) / "tokenizer.json").exists())

    def test_a100_defaults_and_cli_validation(self):
        args = parse_args([])
        self.assertEqual(args.batch_size * args.gradient_accumulation_steps, 64)
        self.assertEqual(args.report_to, "none")
        with self.assertRaises(SystemExit):
            parse_args(["--batch-size", "0"])


if __name__ == "__main__":
    unittest.main()
