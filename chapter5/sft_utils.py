"""Response-only SFT, with Hugging Face handling the label shift once."""

import torch
from torch.utils.data import Dataset, DataLoader

from a100_utils import PROMPTS_TEMPLATE


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.examples = []
        for item in data:
            prompt = PROMPTS_TEMPLATE.format(question=item.get("question", item.get("problem")))
            completion = item.get("completion")
            if completion is None:
                if "solution" not in item:
                    raise ValueError("SFT requires a solution or completion, not only an answer")
                completion = item["solution"] + "</think> <answer>" + str(item["answer"]) + "</answer>"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(completion, add_special_tokens=False)
            # Skip overlong demonstrations rather than truncate their final answers.
            ids = prompt_ids + response_ids
            if not response_ids or len(ids) > max_length:
                continue
            self.examples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor([-100] * len(prompt_ids) + response_ids, dtype=torch.long),
            })
        if not self.examples:
            raise ValueError("No complete SFT demonstrations fit max_length")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_fn(batch, pad_token_id):
    length = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for row, item in enumerate(batch):
        n = len(item["input_ids"])
        ids[row, :n] = item["input_ids"]
        labels[row, :n] = item["labels"]
        mask[row, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def sft_train(policy, data, tokenizer, num_steps, batch_size=2, learning_rate=1e-5,
              gradient_clip=1.0, gradient_accumulation_steps=4, max_length=2048):
    dataset = SFTDataset(data, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.train()
    device = next(policy.parameters()).device
    iterator = iter(loader)
    history = []
    print(f"SFT_DATA kept={len(dataset)} supplied={len(data)}", flush=True)
    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = policy(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SFT loss at step {step}")
            (loss / gradient_accumulation_steps).backward()
            losses.append(loss.item())
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite SFT gradient at step {step}")
        optimizer.step()
        record = {"step": step, "loss": sum(losses) / len(losses), "grad_norm": grad_norm.item()}
        history.append(record)
        print(f"SFT step={step}/{num_steps} loss={record['loss']:.4f} grad_norm={record['grad_norm']:.4f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    return history
