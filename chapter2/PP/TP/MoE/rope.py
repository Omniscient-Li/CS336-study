import math
import torch
from model_args import DeepSeekV3ModelArgs

def precompute_freqs_cis(
        args : DeepSeekV3ModelArgs ,
) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

     # Basic RoPE frequency calculation (dim/2,)
    freqs = 1.0 / (
        base ** (torch.arange(0 , dim  , 2 , dtype = torch.float32) / dim)
    )

    # Create position indices (seqlen,)
    t = torch.arange(seqlen)

    # Outer product: [positions] × [frequencies] (seqlen, dim/2)
    freqs = torch.outer(t , freqs)

    # Convert to complex expoentials: e^(i*freq*pos) (seqlen, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs) , freqs)
    return freqs_cis

def apply_rotary_emb(
        x : torch.Tensor , 
        freqs_cis : torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor x.
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim/2)
    Returns:
        Tensor with rotary embeddings applied, same shape as x.
    """
    # x: [B, S, H, D]
    dtype = x.dtype
    # x: (B, S, H, D/2, 2) -> (B, S, H, D/2)
    x = torch.view_as_complex(x.float().view(*x.shape[:-1] , -1 , 2))
    freqs_cis = freqs_cis.view(1 , x.size(1) , 1 , x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

