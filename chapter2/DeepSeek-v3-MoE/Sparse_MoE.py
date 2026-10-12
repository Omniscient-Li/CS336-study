"""Educational sparse top-k Mixture-of-Experts layer.

The implementation routes each token to ``top_k`` experts. It is a
single-process reference implementation, not a distributed DeepSeek-V3 MoE.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A feed-forward expert applied to the tokens assigned to it."""

    def __init__(self, n_embd: int, dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        if n_embd <= 0 or hidden_mult <= 0:
            raise ValueError("n_embd and hidden_mult must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        hidden = hidden_mult * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKRouter(nn.Module):
    """Select experts and return normalized gates with shape ``[..., E]``."""

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        top_k: int,
        noisy: bool = False,
        noise_epsilon: float = 1e-2,
    ):
        super().__init__()
        if n_embd <= 0 or num_experts <= 0:
            raise ValueError("n_embd and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if noise_epsilon < 0:
            raise ValueError("noise_epsilon must be non-negative")
        self.top_k = top_k
        self.num_experts = num_experts
        self.noisy = noisy
        self.noise_epsilon = noise_epsilon
        self.linear = nn.Linear(n_embd, num_experts)
        self.noise_linear = nn.Linear(n_embd, num_experts) if noisy else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep expert selection and gate normalization in FP32 under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.linear.weight.float(), self.linear.bias.float())
        if self.noisy and self.training:
            assert self.noise_linear is not None
            with torch.autocast(device_type=x.device.type, enabled=False):
                noise_logits = F.linear(x.float(), self.noise_linear.weight.float(), self.noise_linear.bias.float())
            noise_scale = F.softplus(noise_logits) + self.noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_scale

        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits = sparse_logits.scatter(-1, indices, top_k_logits)
        gates = F.softmax(sparse_logits, dim=-1)
        return gates, indices


class SparseMoE(nn.Module):
    """Sparse top-k MoE with token-wise dispatch and weighted aggregation."""

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.0,
        hidden_mult: int = 4,
        noisy_router: bool = False,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(n_embd, num_experts, top_k, noisy=noisy_router)
        self.experts = nn.ModuleList(
            [Expert(n_embd, dropout=dropout, hidden_mult=hidden_mult) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.n_embd:
            raise ValueError(f"expected input [batch, sequence, {self.n_embd}], got {tuple(x.shape)}")
        if x.shape[0] * x.shape[1] == 0:
            raise ValueError("input must contain at least one token")

        batch, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)
        gates, indices = self.router(x)
        flat_gates = gates.reshape(-1, self.num_experts)
        flat_indices = indices.reshape(-1, self.top_k)
        flat_output = torch.zeros_like(flat_x)

        # Dispatch each expert only the tokens selected for it.
        for expert_id, expert in enumerate(self.experts):
            token_rows, _ = (flat_indices == expert_id).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            expert_output = expert(flat_x.index_select(0, token_rows))
            weights = flat_gates[token_rows, expert_id].unsqueeze(-1)
            contribution = (expert_output.float() * weights).to(flat_output.dtype)
            flat_output.index_add_(0, token_rows, contribution)

        return flat_output.reshape(batch, seq_len, dim)

    @torch.no_grad()
    def routing_counts(self, x: torch.Tensor) -> torch.Tensor:
        """Return how many token choices each expert received."""
        if x.ndim != 3 or x.shape[-1] != self.n_embd:
            raise ValueError(f"expected input [batch, sequence, {self.n_embd}], got {tuple(x.shape)}")
        _, indices = self.router(x)
        return torch.bincount(indices.reshape(-1), minlength=self.num_experts)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--n-embd", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-mult", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noisy-router", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) <= 0:
        parser.error("batch-size, seq-len and steps must be positive")
    if args.lr <= 0:
        parser.error("lr must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    if args.require_a100 and (gpu_name is None or "A100" not in gpu_name):
        raise RuntimeError(f"A100 required; found {gpu_name or device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    with torch.device(device):
        model = SparseMoE(
            n_embd=args.n_embd,
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_mult=args.hidden_mult,
            noisy_router=args.noisy_router,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    x = torch.randn(args.batch_size, args.seq_len, args.n_embd, device=device)
    total_tokens = args.batch_size * args.seq_len
    times_ms: list[float] = []
    last_loss = 0.0

    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if device.type == "cuda" and dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(x)
                loss = output.float().square().mean()
        else:
            output = model(x)
            loss = output.float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"non-finite gradient in {name}")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)
        if output.shape != x.shape:
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        last_loss = loss.item()
        # Deterministic post-update loads; noisy routing is used during training.
        model.eval()
        counts = model.routing_counts(x)
        model.train()
        if int(counts.sum()) != total_tokens * args.top_k:
            raise RuntimeError("routing count does not equal tokens * top_k")
        print(
            f"step={step + 1}/{args.steps} loss={last_loss:.6f} "
            f"step_ms={times_ms[-1]:.2f} load_min={int(counts.min())} load_max={int(counts.max())}",
            flush=True,
        )

    average_ms = sum(times_ms) / len(times_ms)
    report = {
        "status": "PASS",
        "device": str(device),
        "gpu": gpu_name,
        "torch": torch.__version__,
        "precision": args.precision,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "workload": "MoE layer on fixed random inputs, squared-output loss; not language-model training",
        "parameters": parameter_count(model),
        "parameters_millions": round(parameter_count(model) / 1e6, 2),
        "configuration": vars(args),
        "input_shape": [args.batch_size, args.seq_len, args.n_embd],
        "final_loss": last_loss,
        "average_step_ms": round(average_ms, 2),
        "tokens_per_second": round(total_tokens * 1000 / average_ms, 2),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
        if device.type == "cuda"
        else None,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
