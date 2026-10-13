import torch
import torch.nn.functional as F
from run_compute_entropy import run_compute_entropy
def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    outputs = model(input_ids, return_dict=True, use_cache=False)
    logits = outputs.logits # (batch_size, sequence_length, vocab_size)

    probs = F.log_softmax(logits.float(), dim=-1)
    labels = labels.unsqueeze(-1) # (batch_size, sequence_length, 1)
    log_probs = torch.gather(probs, dim=-1, index=labels) # (batch_size, sequence_length)
    log_probs = log_probs.squeeze(-1) # (batch_size, sequence_length)
    if return_token_entropy:
        # Entropy is a logged diagnostic; retain gradients only for log-probs.
        with torch.no_grad():
            entropy = run_compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    else:
        return {"log_probs": log_probs}
