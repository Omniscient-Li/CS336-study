"""Single-process DeepSeek-style MoE with shared/routed experts.

This educational layer uses GELU experts, sigmoid affinities, top-k routing,
and bias-only load balancing. It is not the complete DeepSeek-V3 architecture:
SwiGLU, group-limited routing, expert parallelism and a language model are absent.
The residual connection is included here, as in the original example.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Two-layer nonlinear FFN expert."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        if d_model <= 0 or hidden <= 0:
            raise ValueError("d_model and hidden must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate="tanh")))


class DeepSeekMoE(nn.Module):
    """Shared experts plus sparsely activated routed experts.

    Bias affects selection only. Gates normalize the original sigmoid affinity
    scores. Pass return_counts=True to collect actual forward routing loads;
    call update_bias(counts=counts) once per optimizer step, never in forward.
    """

    def __init__(
        self,
        d_model: int,
        n_routed_exp: int,
        n_shared_exp: int = 1,
        top_k: int = 8,
        routed_hidden: int = 2048,
        shared_hidden: Optional[int] = None,
        bias_lr: float = 0.01,
        fp16_router: bool = False,
        dropout: float = 0.0,
        routed_scale: float = 1.0,
    ):
        super().__init__()
        if d_model <= 0 or n_routed_exp <= 0 or routed_hidden <= 0:
            raise ValueError("model/expert dimensions and routed expert count must be positive")
        if n_shared_exp < 0:
            raise ValueError("n_shared_exp cannot be negative")
        if not 1 <= top_k <= n_routed_exp:
            raise ValueError("top_k must be between 1 and n_routed_exp")
        if shared_hidden is not None and shared_hidden <= 0:
            raise ValueError("shared_hidden must be positive")
        if not math.isfinite(bias_lr) or bias_lr < 0:
            raise ValueError("bias_lr must be finite and non-negative")
        if not math.isfinite(routed_scale) or routed_scale <= 0:
            raise ValueError("routed_scale must be finite and positive")

        self.d_model = d_model
        self.n_routed = n_routed_exp
        self.n_shared = n_shared_exp
        self.top_k = top_k
        self.bias_lr = bias_lr
        self.fp16_router = fp16_router
        self.routed_scale = routed_scale
        self.routed_hidden = routed_hidden
        self.shared_hidden = routed_hidden if shared_hidden is None else shared_hidden
        self.routed = nn.ModuleList(
            [ExpertFFN(d_model, routed_hidden, dropout) for _ in range(n_routed_exp)]
        )
        self.shared = nn.ModuleList(
            [ExpertFFN(d_model, self.shared_hidden, dropout) for _ in range(n_shared_exp)]
        )
        self.centroids = nn.Parameter(torch.empty(n_routed_exp, d_model))
        nn.init.normal_(self.centroids, std=d_model ** -0.5)
        self.register_buffer("bias", torch.zeros(n_routed_exp))

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected [batch, sequence, {self.d_model}], got {tuple(x.shape)}")
        if x.shape[0] * x.shape[1] == 0:
            raise ValueError("input must contain at least one token")
        return x.reshape(-1, self.d_model)

    def _route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp16_router and x_flat.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = F.linear(x_flat, self.centroids).float()
        else:
            with torch.autocast(device_type=x_flat.device.type, enabled=False):
                logits = F.linear(x_flat.float(), self.centroids.float())
        scores = logits.sigmoid()
        indices = (scores + self.bias.float()).topk(self.top_k, dim=-1).indices
        selected_scores = scores.gather(-1, indices)
        gates = selected_scores / selected_scores.sum(-1, keepdim=True).clamp_min(1e-20)
        return gates * self.routed_scale, indices

    def forward(
        self, x: torch.Tensor, return_counts: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        flat_x = self._flatten_input(x)
        shared_out = torch.zeros_like(flat_x)
        for expert in self.shared:
            shared_out = shared_out + expert(flat_x).to(flat_x.dtype)

        gates, indices = self._route(flat_x)
        routed_out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.routed):
            token_rows, choice_rows = (indices == expert_id).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            expert_out = expert(flat_x.index_select(0, token_rows))
            weights = gates[token_rows, choice_rows].unsqueeze(-1)
            contribution = (expert_out.float() * weights).to(routed_out.dtype)
            routed_out.index_add_(0, token_rows, contribution)

        output = x + (shared_out + routed_out).reshape_as(x)
        if return_counts:
            counts = torch.bincount(indices.detach().reshape(-1), minlength=self.n_routed)
            return output, counts
        return output

    @torch.no_grad()
    def update_bias(
        self, x: Optional[torch.Tensor] = None, *, counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Increase bias for under-used experts and decrease it for over-used ones.

        Prefer counts from the corresponding forward pass. The x fallback
        recomputes routing using current weights, for standalone experiments.
        Accumulate counts over microbatches before updating once per step.
        """
        if (x is None) == (counts is None):
            raise ValueError("provide exactly one of x or counts")
        if counts is None:
            _, indices = self._route(self._flatten_input(x))
            counts = torch.bincount(indices.reshape(-1), minlength=self.n_routed)
        if counts.shape != (self.n_routed,):
            raise ValueError("counts must have one entry per routed expert")
        loads = counts.to(device=self.bias.device, dtype=torch.float32)
        if not torch.isfinite(loads).all() or (loads < 0).any() or loads.sum() <= 0:
            raise ValueError("counts must be finite, non-negative and have a positive sum")
        self.bias.add_((loads.mean() - loads).sign().to(self.bias.dtype), alpha=self.bias_lr)
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--routed-experts", type=int, default=64)
    parser.add_argument("--shared-experts", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--shared-hidden", type=int, default=None)
    parser.add_argument("--bias-lr", type=float, default=0.001)
    parser.add_argument("--routed-scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fp16-router", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) <= 0:
        parser.error("batch-size, seq-len and steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    if args.require_a100 and (gpu is None or "A100" not in gpu):
        raise RuntimeError(f"A100 required; found {gpu or device}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    # FP32 weights/AdamW states, mixed precision expert computation.
    with torch.device(device):
        model = DeepSeekMoE(
            args.d_model, args.routed_experts, args.shared_experts, args.top_k,
            args.hidden, args.shared_hidden, args.bias_lr, args.fp16_router,
            args.dropout, args.routed_scale,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)
    original_centroids = model.centroids.detach().clone()
    times_ms, losses, load_history = [], [], []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=device.type, dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            output, counts = model(x, return_counts=True)
            loss = output.float().square().mean()
        if output.shape != x.shape or not torch.isfinite(loss):
            raise RuntimeError(f"invalid output/loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"non-finite gradient in {name}")
        scaler.step(optimizer)
        scaler.update()
        if int(counts.sum()) != args.batch_size * args.seq_len * args.top_k:
            raise RuntimeError("expert counts must sum to tokens * top_k")
        model.update_bias(counts=counts)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)
        losses.append(loss.item())
        load_history.append(counts.tolist())
        print(
            f"step={step + 1}/{args.steps} loss={losses[-1]:.6f} "
            f"step_ms={times_ms[-1]:.2f} load_min={int(counts.min())} load_max={int(counts.max())}",
            flush=True,
        )
    if args.top_k > 1 and torch.equal(original_centroids, model.centroids):
        raise RuntimeError("router weights did not update")
    total = sum(parameter.numel() for parameter in model.parameters())
    active = (
        model.centroids.numel() + args.top_k * 2 * args.d_model * args.hidden
        + args.shared_experts * 2 * args.d_model * model.shared_hidden
    )
    warm_times = times_ms[1:] or times_ms
    report = {
        "status": "PASS", "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "configuration": vars(args), "parameters": total,
        "active_parameters_per_token": active, "parameter_dtype": str(model.centroids.dtype),
        "input_shape": list(x.shape), "losses": losses,
        "step_times_ms": times_ms, "mean_step_ms_after_first": sum(warm_times) / len(warm_times),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "expert_loads": load_history, "final_router_bias": model.bias.tolist(),
        "workload": "MoE layer on fixed random inputs with squared-output loss, not language-model training",
    }
    report_text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(report_text + "\n", encoding="utf-8")
    print(report_text, flush=True)


if __name__ == "__main__":
    main()
