import torch
from run_masked_normalize import run_masked_normalize


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if normalize_constant is None:
        per_example = (policy_log_probs * response_mask).sum(1) / response_mask.sum(1).clamp_min(1)
    else:
        if normalize_constant <= 0:
            raise ValueError("normalize_constant must be positive")
        per_example = run_masked_normalize(policy_log_probs, response_mask, dim=1,
                                          normalize_constant=normalize_constant)
    loss = -per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}
