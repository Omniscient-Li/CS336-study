from typing import Callable

import torch
def run_compute_group_normalized_rewards(
    reward_fn : Callable ,
    rollout_responses : list[str] ,
    repeated_ground_truths : list[str] ,
    group_size : int ,
    advantage_eps : float ,
    normalize_by_std : bool ,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    计算每个 rollout 的 reward，并按组进行归一化。每 group_size 个连续的 rollouts 属于同一组，归一化时会使用该组内的 reward 进行中心化（减均值）和可选的标准差归一化。
    """
    if group_size < 2 or advantage_eps <= 0:
        raise ValueError("group_size must be >= 2 and advantage_eps must be positive")
    if len(rollout_responses) != len(repeated_ground_truths) or not rollout_responses or len(rollout_responses) % group_size:
        raise ValueError("Responses and answers must have matching lengths divisible by group_size")
    raw_rewards = torch.tensor([
        reward_fn(r , g)["reward"] for r , g in zip(rollout_responses , repeated_ground_truths)
    ])

    r = raw_rewards.reshape(-1 , group_size)
    centerd = r - r.mean(-1 , keepdim = True)
    if normalize_by_std :
        normalized = centerd / (r.std(dim = 1 , unbiased = True , keepdim = True) + advantage_eps)
    else:
        normalized = centerd

    return normalized.flatten(), raw_rewards, {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(unbiased=False).item(),
        "mixed_group_fraction": (r.std(dim=1) > 0).float().mean().item(),
    }
