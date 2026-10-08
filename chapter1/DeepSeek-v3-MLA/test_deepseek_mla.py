"""Run CPU or A100 correctness tests: python test_deepseek_mla.py --device cuda --require-a100."""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


spec = importlib.util.spec_from_file_location("deepseek_mla", Path(__file__).with_name("DeepSeek-MLA.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DeepSeekAttention = module.DeepSeekAttention


def reference_rope(x):
    dim = x.shape[-1]
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = torch.outer(torch.arange(x.shape[2], device=x.device).float(), theta)
    frequencies = torch.polar(torch.ones_like(angles), angles)
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(pairs * frequencies).flatten(-2).to(x.dtype)


def reference_attention(model, x):
    batch, length, _ = x.shape

    def heads(tensor, dim):
        return tensor.reshape(batch, length, model.num_heads, dim).transpose(1, 2)

    latent = model.W_dkv_content(x)
    q_content = heads(model.W_q_content(x), model.d_head)
    k_content = heads(model.W_uk_content(latent), model.d_head)
    value = heads(model.W_uv_content(latent), model.d_head)
    q_rope = reference_rope(heads(model.W_q_pos(x), model.d_rope))
    k_rope = reference_rope(model.W_k_pos(x).unsqueeze(1)).expand(-1, model.num_heads, -1, -1)
    query = torch.cat((q_content, q_rope), -1).float()
    key = torch.cat((k_content, k_rope), -1).float()
    # Use PyTorch SDPA with concatenated Q/K as an independent attention oracle.
    context = F.scaled_dot_product_attention(
        query, key, value.float(), is_causal=True,
        scale=1.0 / math.sqrt(model.d_head + model.d_rope),
    )
    context = context.to(value.dtype).transpose(1, 2).reshape(batch, length, model.d_model)
    return model.W_o(context)


def check_reference(device, dtype, config):
    d_model, heads, latent, rope, length = config
    model = DeepSeekAttention(d_model, heads, latent, rope, max_seq_len=length).to(device=device, dtype=dtype)
    reference = copy.deepcopy(model)
    x = torch.randn(2, length, d_model, device=device, dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    actual = model(x)
    expected = reference_attention(reference, ref_x)
    tolerance = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=tolerance * 4, rtol=tolerance * 4)
    for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=tolerance * 4, rtol=tolerance * 4, msg=name)

    # Changing future tokens must leave the prefix output unchanged.
    model.eval()
    with torch.no_grad():
        changed = x.detach().clone()
        split = max(1, length // 2)
        changed[:, split:] = torch.randn_like(changed[:, split:]) * 3
        torch.testing.assert_close(model(changed)[:, :split], actual.detach()[:, :split], atol=0, rtol=0)

    return {"dtype": str(dtype), "config": config, "max_output_error": (actual - expected).abs().max().item()}


def check_validation(device):
    invalid_configs = [(15, 4, 8, 4), (16, 0, 8, 4), (16, 4, 0, 4), (16, 4, 8, 3), (16, 4, 8, 0)]
    for config in invalid_configs:
        try:
            DeepSeekAttention(*config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid dimensions accepted: {config}")
    model = DeepSeekAttention(16, 4, 8, 2, max_seq_len=8).to(device)
    for shape in [(1, 9, 16), (1, 0, 16), (1, 3, 15), (1, 16)]:
        try:
            model(torch.randn(*shape, device=device))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid input accepted: {shape}")


def check_rope(device, dtype):
    # Long positions catch accidental conversion of frequencies to BF16.
    rope = module.RotaryPositionalEncoding(16, max_seq_len=1024).to(device=device, dtype=dtype)
    x = torch.randn(2, 3, 1024, 16, device=device, dtype=dtype)
    torch.testing.assert_close(rope(x), reference_rope(x), atol=1e-5 if dtype == torch.float32 else 0.02, rtol=0.01)


def check_training(device):
    model = DeepSeekAttention(128, 4, 32, 16, dropout=0.2, max_seq_len=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    old_weight = model.W_q_content.weight.detach().clone()
    x = torch.randn(2, 64, 128, device=device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(x)
            loss = output.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
    assert not torch.equal(old_weight, model.W_q_content.weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-a100", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    if args.require_a100 and (gpu is None or "A100" not in gpu):
        raise RuntimeError(f"A100 required; actual device: {gpu or device}")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtypes = [torch.float32, torch.float16, torch.bfloat16] if device.type == "cuda" else [torch.float32]
    # Equal/different content and RoPE dimensions, singleton and non-tile sequence lengths.
    configs = [(512, 8, 128, 64, 64), (128, 4, 32, 16, 257), (32, 4, 8, 2, 1)]
    results = []
    for dtype in dtypes:
        check_rope(device, dtype)
        for config in configs:
            result = check_reference(device, dtype, config)
            results.append(result)
            print(json.dumps(result), flush=True)
    check_validation(device)
    check_training(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"status": "PASS", "device": str(device), "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "reference_cases": len(results), "checks": ["forward", "input_and_parameter_gradients", "causality", "rope", "invalid_inputs", "optimizer_steps_with_dropout_and_autocast"]}), flush=True)


if __name__ == "__main__":
    main()
