"""
LLaMA 2 model implementation from scratch in PyTorch.
Based on the LLaMA 2 paper (Touvron et al., 2023).

Key components:
- RMSNorm: Root Mean Square Layer Normalization
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU FeedForward
- KV Cache for efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    """LLaMA model configuration.
    
       Default values correspond to LLaMA-2 7B.
    """
    dim : int = 4096
    n_layers : int = 32
    n_heads : int = 32 # Number of heads for the queries
    n_kv_heads : Optional[int] = None # Number of heads for K and V
    vocab_size : int = -1 # This will be set when we laod the tokenizer
    multiple_of : int = 256
    ffn_dim_multiplier : Optional[float] = None 
    norm_eps : float = 1e-5

    #Needed for KV cache
    max_batch_size : int = 32
    max_seq_len : int = 2048

    device : str = None



def precompute_theta_pos_frequencies(head_dim : int , seq_len : int , device : str , theta : float = 10000.0):
    """Precompute the complex rotation frequencies for RoPE.
    
        For each position m and each dimension pair i, we compute:
            freqs[m, i] = m * (1 / theta^(2i / head_dim))
    
        These are stored as complex numbers (cos(freqs), sin(freqs)) for efficient
        rotation via complex multiplication.
    
    Args:
        head_dim: dimension of each attention head (must be even)
        seq_len:  maximum sequence length
        device:   target device
        theta:    base frequency (default 10000.0 as in the original paper)
    
    Returns:
        Complex tensor of shape (Seq_Len, Head_Dim / 2)
    """
    #As written in the paper , the dimension of the embedding muast be even.
    assert head_dim % 2 == 0 , "Dimension must be even"
    """
    Bulid the theta parameters
    According to the formula : theta_i = 10000 ^ (-2(i - 1) / dim) for i = [1 , 2 , ... , head_dim / 2]
    Shape : (Head_Dim / 2)
    """
    theta_numerator = torch.arange(0 , head_dim , 2).float()
    # Shape : (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions ( the "m parameters")
    # Shape : (Seq_Len)
    m = torch.arange(seq_len , device = device)
    # Multiply each theta by each position using the outer product
    # Shape (Seq_Len) out_product * (Head_Dim / 2) -> (Seq_Len , Head_Dim / 2)
    freqs = torch.outer(m , theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta) , where R = 1 as follows:
    #(Seq_Len , Head_Dim / 2) -> (Seq_Len , Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs) , freqs)
    return freqs_complex


def apply_rotary_embeddings(x : torch.Tensor , freqs_complex : torch.Tensor , device : str):
    """Apply RoPE rotation to query or key tensors.
    
        Steps:
            1. Reshape x so consecutive dimension pairs become complex numbers
            2. Broadcast the precomputed frequencies and multiply (this rotates)
            3. Convert back to real numbers and reshape to original shape
    
        Args:
            x:              query or key tensor (B, Seq_Len, H, Head_Dim)
            freqs_complex:  precomputed complex frequencies (Seq_Len, Head_Dim/2)
            device:         target device
    
        Returns:
            Rotated tensor of same shape as x
    """
    #(B , Seq_Len , H , Head_Dim) -> (B , Seq_Len , H , Head_Dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape( *x.shape[: -1] , -1 , 2))
    #(Seq_Len , Head_Dim / 2) -> (1 , Seq_Len , 1 , Head_Dim / 2) 
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    #(B , Seq_Len , H , Head_Dim / 2) * (1 , Seq_Len , 1 , Head_Dim / 2) = (B , Seq_Len , H , Head_Dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B , Seq_Len , H , Head_Dim / 2) -> (B , Seq_Len , H , Head_Dim / 2 , 2)
    x_out = torch.view_as_real(x_rotated)
    # (B , Seq_Len , H , Head_Dim / 2 , 2) -> (B , Seq_Len , H , Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
        Unlike LayerNorm, RMSNorm does NOT subtract the mean — it only rescales
        by the root-mean-square of the activations.  LLaMA applies this BEFORE
        each sub-layer (pre-norm), not after.
    
        Math:
        RMSNorm(x) = x / RMS(x) * gamma
         where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self , dim : int , eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self , x : torch.Tensor):
        # (B , Seq_Len , Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1 , keepdim = True) + self.eps)

    def forward(self , x : torch.Tensor):
        # (Dim) * (B , Seq_Len , Dim) = (B , Seq_Len , Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    


def repeat_kv(x : torch.Tensor , n_rep : int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads (GQA).
    
        Args:
            x:      keys or values (B, Seq_Len, N_KV_Heads, Head_Dim)
            n_rep:  repetition factor (n_heads_q // n_kv_heads)
    
        Returns:
            Repeated tensor (B, Seq_Len, N_KV_Heads * n_rep, Head_Dim)
    """
    batch_size , seq_len , n_kv_heads , head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x[: , : , : , None , : ]
            .expand(batch_size , seq_len , n_kv_heads , n_rep , head_dim)
            .reshape(batch_size , seq_len , n_kv_heads * n_rep , head_dim)
        )
    


class SelfAttention(nn.Module):
    """Multi-head / Grouped Query Attention with RoPE and KV cache.
    
        KV cache: During autoregressive generation, we process one token at a time.
        Instead of recomputing keys and values for all past tokens, we store them
        in a cache. At each step we only compute K, V for the new token, then
        concatenate with the cached ones. This reduces computation from O(n²) to O(n).
    """
    def __init__(self , args : ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated to match the head of the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head 
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim , args.n_heads * self.head_dim , bias = False)
        self.wk = nn.Linear(args.dim , self.n_kv_heads * self.head_dim , bias = False)
        self.wv = nn.Linear(args.dim , self.n_kv_heads * self.head_dim , bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim , args.dim , bias = False)

        self.cache_k = torch.zeros(args.max_batch_size , args.max_seq_len , self.n_kv_heads , self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size , args.max_seq_len , self.n_kv_heads , self.head_dim)

    def forward(self , x : torch.Tensor , start_pos : int , freqs_complex : torch.Tensor):
        batch_size , seq_len , _ = x.shape # (B , 1 , Dim)

        #(B , 1 , Dim) -> (B , 1 , H_Q * Head_Dim)
        xq = self.wq(x)
        #(B , 1 , Dim) -> (B , 1 , H_KV * Head_Dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B , 1, H_Q * Head_Dim) -> (B , 1 , H_Q , Head_Dim)
        xq = xq.view(batch_size , seq_len , self.n_heads_q , self.head_dim)
        # (B , 1, H_KV * Head_Dim) -> (B , 1 , H_KV , Head_Dim)
        xk = xk.view(batch_size , seq_len , self.n_kv_heads , self.head_dim)
        # (B , 1, H_KV * Head_Dim) -> (B , 1 , H_KV , Head_Dim)
        xv = xv.view(batch_size , seq_len , self.n_kv_heads , self.head_dim)

        # Dose not change the shape of the tensors
        xq = apply_rotary_embeddings(xq , freqs_complex , device = x.device)
        xk = apply_rotary_embeddings(xk , freqs_complex , device = x.device)

        #Replace the entry in the cache for this token
        self.cache_k[:batch_size , start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size , start_pos : start_pos + seq_len] = xv

        #Retrieve all the cached keys and values so far
        # (B , Seq_Len_KV , H_KV , Head_Dim)
        keys = self.cache_k[:batch_size , 0 : start_pos + seq_len]
        values = self.cache_v[:batch_size , 0 : start_pos + seq_len]

        #Repeat the heads of the K and V to each number of heads of queries
        keys = repeat_kv(keys , self.n_rep)
        values = repeat_kv(values , self.n_rep)

        # (B , 1 , H_Q , Head_Dim) -> (B , H_Q , 1 , Head_Dim)
        xq = xq.transpose(1 , 2)
        keys = keys.transpose(1 , 2)
        values = values.transpose(1 , 2)

        # (B , H_Q , 1 , Head_Dim) @ (B , H_Q , Head_Dim , Seq_Len_KV) -> (B , H_Q , 1 , Seq_Len_KV)
        scores = torch.matmul(xq , keys.transpose(2 , 3)) / math.sqrt(self.head_dim)
        scores = torch.softmax(scores.float() , dim = -1).to(xq.dtype)

        # (B , H_Q , 1 , Seq_Len_KV) @ (B , H_Q , Seq_Len_KV , Head_Dim) -> (B , H_Q , 1 , Head_Dim)
        out = torch.matmul(scores , values)

        # (B , H_Q , 1 , Head_Dim) -> (B , 1 , H_Q , Head_Dim)
        out = out.transpose(1 , 2).contiguous().view(batch_size , seq_len , -1)
        return self.wo(out)

 
class FeedForward(nn.Module):
    """SwiGLU FeedForward network.
    
        Uses the SwiGLU activation:
            output = W2 @ (SiLU(W1(x)) * W3(x))
    
        The hidden dimension is computed as:
            hidden = 4 * dim  →  2/3 of that  →  round up to multiple_of
        This is roughly 8/3 * dim, which is the standard for LLaMA.
    """
    def __init__(self , args : ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = ((hidden_dim + args.multiple_of - 1) // args.multiple_of) * args.multiple_of

        self.w1 = nn.Linear(args.dim , hidden_dim , bias = False)
        self.w2 = nn.Linear(hidden_dim , args.dim , bias = False)
        self.w3 = nn.Linear(args.dim , hidden_dim , bias = False)

    def forward(self , x : torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    """A single transformer block with pre-norm.
    
        Architecture (pre-norm):
            x = x + SelfAttention(RMSNorm(x))
            x = x + FeedForward(RMSNorm(x))
    """
    def __init__(self , args : ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        #Normalization before the self attention
        self.attention_norm = RMSNorm(args.dim , eps = args.norm_eps)
        #Normalization before the feed forward block
        self.ffn_norm = RMSNorm(args.dim , eps = args.norm_eps)

    def forward(self , x : torch.Tensor , start_pos : int , freqs_complex : torch.Tensor):
        #(B , Seq_Len , Dim) + (B , Seq_Len , Dim) -> (B , Seq_Len , Dim)
        h = x + self.attention.forward(self.attention_norm(x) , start_pos , freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

      

#基础transformer模型
class Transformer(nn.Module):
    """LLaMA Transformer model.
    
        Architecture:
            Embedding → [EncoderBlock × N] → RMSNorm → Output Projection
    
        The output projection maps back to vocabulary space for next-token prediction.
        Weights between the embedding layer and the output projection can be tied
        (but are not in our implementation — LLaMA keeps them separate).
    """
    def __init__(self , args : ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1 # Vocab_size must be set
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size , args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim , eps = args.norm_eps)
        self.output = nn.Linear(args.dim , self.vocab_size , bias = False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads , self.args.max_seq_len * 2 , device = self.args.device)

    def forward(self, tokens : torch.Tensor , start_pos : int):
        #(B , Seq_Len)
        batch_size , seq_len = tokens.shape
        assert seq_len == 1 # Only one token at a time can be processed

        #(B , =Seq_Len) -> (B , Seq_Len , Dim)
        h = self.tok_embeddings(tokens)

        #Retrive the pairs (m , theta) corresponding to the positions [start_pos : start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        #Consecutively apply the transformer blocks
        for layer in self.layers:
            h = layer(h , start_pos , freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output