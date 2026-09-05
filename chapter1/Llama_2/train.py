"""
Training pipeline for LLaMA from scratch or via fine-tuning.

Supports:
- Training from scratch on any text corpus
- Gradient accumulation for effective large batch sizes
- Mixed precision training (bfloat16 or float16)
- Learning rate scheduler (cosine with warmup)
- Gradient clipping
- Checkpointing

Usage:
    python train.py --data_path ./data/input.txt --epochs 3 --batch_size 4 --lr 3e-4
"""

import os
import time
import math
import argparse
from pathlib import Path
from typing import Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model import ModelArgs, Transformer
from config import llama_tiny_config


# ==============================================================================
# Dataset
# ==============================================================================

class TextDataset(Dataset):
    """Simple text dataset that tokenizes a text file into fixed-length chunks.

    Each sample is (input_tokens, target_tokens) where target is input shifted by 1.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int = 1024,
    ):
        self.seq_len = seq_len

        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Tokenizing {len(text):,} characters...")
        tokens = tokenizer.encode(text, out_type=int, add_bos=True, add_eos=False)
        print(f"Tokenized into {len(tokens):,} tokens")

        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        # 不相交分块：每 seq_len 个 token 一块（stride = seq_len）
        # -1 保证最后一块的 target 不越界（y 需要 start + seq_len + 1 个 token）
        return max(0, len(self.tokens) // self.seq_len - 1)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # Input:  tokens[start : start + seq_len]
        # Target: tokens[start+1 : start + seq_len + 1]
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


# ==============================================================================
# Training utilities
# ==============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    """Cosine learning rate scheduler with linear warmup.

    During warmup: lr increases linearly from 0 to peak_lr.
    After warmup: lr decays via cosine schedule to min_lr.
    """

    def lr_lambda(step: int):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    loss: float,
    save_dir: str,
    name: str = "checkpoint",
):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}_step{step}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


# ==============================================================================
# Training loop
# ==============================================================================

def train(args: argparse.Namespace):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Tokenizer ---
    # Use SentencePiece tokenizer if provided, otherwise use a simple
    # character-level tokenizer for quick testing.
    if args.tokenizer_path:
        from sentencepiece import SentencePieceProcessor
        tokenizer = SentencePieceProcessor()
        tokenizer.load(args.tokenizer_path)
        vocab_size = tokenizer.vocab_size()
        print(f"Loaded SentencePiece tokenizer, vocab size: {vocab_size}")
    else:
        # Fallback: simple character-level tokenizer for testing
        class CharTokenizer:
            def __init__(self, text: str = ""):
                chars = sorted(list(set(text))) if text else [chr(i) for i in range(256)]
                self.stoi = {ch: i for i, ch in enumerate(chars)}
                self.itos = {i: ch for i, ch in enumerate(chars)}
                self.vocab_size = len(chars)

            def encode(self, text, **kwargs):
                return [self.stoi.get(c, 0) for c in text]

            def decode(self, tokens):
                return "".join(self.itos.get(t, "?") for t in tokens)

            def vocab_size_fn(self):
                return self.vocab_size

        tokenizer = CharTokenizer()
        vocab_size = tokenizer.vocab_size_fn()
        print(f"Using character-level tokenizer, vocab size: {vocab_size}")

    # --- Model ---
    if args.model_config == "tiny":
        model_args = llama_tiny_config(
            device=device,
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.seq_len,
        )
    else:
        model_args = ModelArgs(
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            vocab_size=vocab_size,
            multiple_of=args.multiple_of,
            norm_eps=1e-5,
            max_batch_size=args.batch_size,
            max_seq_len=args.seq_len,
            device=device,
        )

    model_args.vocab_size = vocab_size
    model = Transformer(model_args).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # --- Data ---
    train_dataset = TextDataset(args.data_path, tokenizer, seq_len=args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- LR scheduler ---
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # --- Mixed precision ---
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # --- Training loop ---
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}\n")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Forward pass (with optional mixed precision)
            if args.fp16:
                with torch.cuda.amp.autocast():
                    logits = train_forward(model, x, y)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, model.vocab_size),
                        y.view(-1),
                        ignore_index=-100,
                    )
            else:
                logits = train_forward(model, x, y)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    y.view(-1),
                    ignore_index=-100,
                )

            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer step
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                lr=f"{current_lr:.2e}",
            )

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"  Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} completed | "
            f"Loss: {avg_epoch_loss:.4f} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(
                model, optimizer, scheduler,
                global_step, epoch + 1, avg_epoch_loss,
                args.save_dir, "best",
            )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                global_step, epoch + 1, avg_epoch_loss,
                args.save_dir, f"epoch{epoch+1}",
            )

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    return model


# ==============================================================================
# Full-sequence training forward (with causal mask)
# ==============================================================================

def train_forward(model: Transformer, x: torch.Tensor, y: torch.Tensor):
    """Training forward pass — process full sequence with causal masking.

    Unlike the inference forward (which uses KV cache for one-token-at-a-time),
    this processes the entire sequence at once using teacher forcing.
    """
    batch_size, seq_len = x.shape

    # Embed tokens
    h = model.tok_embeddings(x)  # (B, Seq_Len, Dim)

    # Get RoPE freqs for the full sequence
    # (Seq_Len, Head_Dim/2)——batch/head 维的扩展在 apply_rotary_embeddings 内部完成
    freqs_complex = model.freqs_complex[:seq_len]

    # Pass through layers
    for layer in model.layers:
        h = train_encoder_block_forward(layer, h, freqs_complex)

    h = model.norm(h)
    logits = model.output(h).float()  # (B, Seq_Len, Vocab_Size)

    return logits


def train_encoder_block_forward(block, x: torch.Tensor, freqs_complex: torch.Tensor):
    """Full-sequence forward for one encoder block (training mode)."""
    # Pre-norm + Self-Attention (full sequence with causal mask)
    normed = block.attention_norm(x)
    attn_out = train_attention_forward(block.attention, normed, freqs_complex)
    h = x + attn_out

    # Pre-norm + FeedForward
    out = h + block.feed_forward(block.ffn_norm(h))
    return out


def train_attention_forward(attn, x: torch.Tensor, freqs_complex: torch.Tensor):
    """Full-sequence self-attention with causal masking and GQA.

    Args:
        x:              (B, Seq_Len, Dim)
        freqs_complex:  (1, Seq_Len, 1, Head_Dim/2)

    Returns:
        (B, Seq_Len, Dim)
    """
    from model import apply_rotary_embeddings, repeat_kv

    batch_size, seq_len, _ = x.shape

    # Linear projections
    xq = attn.wq(x)  # (B, Seq_Len, H_Q * Head_Dim)
    xk = attn.wk(x)
    xv = attn.wv(x)

    xq = xq.view(batch_size, seq_len, attn.n_heads_q, attn.head_dim)
    xk = xk.view(batch_size, seq_len, attn.n_kv_heads, attn.head_dim)
    xv = xv.view(batch_size, seq_len, attn.n_kv_heads, attn.head_dim)

    # Apply RoPE
    xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
    xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

    # Repeat KV for GQA
    xk = repeat_kv(xk, attn.n_rep)  # (B, Seq_Len, H_Q, Head_Dim)
    xv = repeat_kv(xv, attn.n_rep)

    # Rearrange: (B, H_Q, Seq_Len, Head_Dim)
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(attn.head_dim)

    # Causal mask: prevent attending to future tokens
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=x.device), diagonal=1
    ).bool()
    scores = scores.masked_fill(causal_mask, float("-inf"))

    scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)

    # Weighted sum of values
    output = torch.matmul(scores, xv)

    # Merge heads: (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    return attn.wo(output)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LLaMA model from scratch.")

    # Data
    parser.add_argument("--data_path", type=str, default="./data/input.txt",
                        help="Path to training text file")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to SentencePiece tokenizer.model (optional)")

    # Model
    parser.add_argument("--model_config", type=str, default="tiny",
                        choices=["tiny", "custom"],
                        help="Model configuration preset")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--multiple_of", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Hardware
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    # Logging / Saving
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
