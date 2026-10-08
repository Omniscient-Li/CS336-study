import argparse
import torch
import torch.nn as nn
import math

class RotaryPositionalEncoding(nn.Module):
    """
    Helper module to apply Rotary Positional Encoding (RoPE).
    This is not added to the embeddings but is applied directly to
    the Query and Key vectors.
    """
    def __init__(self , d_head , max_seq_len = 2048):
        super().__init__()
        if d_head <= 0 or d_head % 2:
            raise ValueError("RoPE dimension must be positive and even")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.d_head = d_head
        self.max_seq_len = max_seq_len

    def forward(self , x):
        # x shape : [batch , num_heads , seq_len , d_head]
        if x.ndim != 4 or x.shape[-1] != self.d_head:
            raise ValueError("RoPE input must have shape [batch, heads, sequence, d_head]")
        seq_len = x.shape[2]
        if not 0 < seq_len <= self.max_seq_len:
            raise ValueError("sequence length must be between 1 and max_seq_len")

        # Keep frequencies in FP32 even after model.half()/bfloat16().
        theta = 10000.0 ** (
            -torch.arange(0, self.d_head, 2, device=x.device, dtype=torch.float32)
            / self.d_head
        )
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        angles = positions[:, None] * theta[None, :]
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs.unbind(-1)
        cos, sin = angles.cos(), angles.sin()
        rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1)
        return rotated.flatten(-2).to(x.dtype)


class DeepSeekAttention(nn.Module):
    """
    Simplified causal MLA with decoupled RoPE and a shared positional key.
    This version materializes content K/V; it does not implement latent KV
    caching, weight absorption, or DeepSeek's query compression/YaRN.
    """
    def __init__(self , d_model , num_heads , d_latent , d_rope , dropout = 0.0 , max_seq_len = 2048):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model and num_heads must be positive, and d_model divisible by num_heads")
        if d_latent <= 0:
            raise ValueError("d_latent must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_latent = d_latent
        self.d_rope = d_rope
        self.max_seq_len = max_seq_len

        # --- A: Content Path (Pure MLA) ---
        self.W_q_content = nn.Linear(d_model , d_model)
        self.W_dkv_content = nn.Linear(d_model , d_latent)
        self.W_uk_content = nn.Linear(d_latent , d_model)
        self.W_uv_content = nn.Linear(d_latent , d_model)

        # --- B: Position Path (RoPE Applied) ---
        # Decoupled MLA shares the positional key across all heads.
        self.W_k_pos = nn.Linear(d_model , d_rope)
        self.W_q_pos = nn.Linear(d_model , d_rope * num_heads)

        # RoPE module to apply the rotations
        self.rope = RotaryPositionalEncoding(d_rope , max_seq_len)

         # --- C: Final Output Projection ---
        self.W_o = nn.Linear(d_model , d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask' , torch.triu(
            torch.ones(1 , 1 , max_seq_len , max_seq_len, dtype=torch.bool) , diagonal = 1
        ), persistent=False)

    def forward(self , x):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("input must have shape [batch, sequence, d_model]")
        batch_size , seq_len , _ = x.shape
        if not 0 < seq_len <= self.max_seq_len:
            raise ValueError("sequence length must be between 1 and max_seq_len")

        # --- A: Content Path Calculation ---
        # This path is cache-friendly and position-agnostic.
        q_c = self.W_q_content(x).view(batch_size , seq_len , self.num_heads , self.d_head).transpose(1 , 2)
        c_kv = self.W_dkv_content(x)
        k_c = self.W_uk_content(c_kv).view(batch_size , seq_len , self.num_heads , self.d_head).transpose(1 , 2)
        v_c = self.W_uv_content(c_kv).view(batch_size , seq_len , self.num_heads , self.d_head).transpose(1, 2)

        # --- B: Position Path Calculation ---
        q_r_unrotated = self.W_q_pos(x).view(batch_size , seq_len , self.num_heads , self.d_rope).transpose(1 , 2)
        k_r_unrotated = self.W_k_pos(x).unsqueeze(1)

        # Apply RoPE to the positional Query and Key vectors
        q_r = self.rope(q_r_unrotated)
        k_r = self.rope(k_r_unrotated)

        # --- C: Combining Paths for Final Attention Score ---
        # Accumulate logits and softmax in FP32 for FP16/BF16 stability.
        with torch.autocast(device_type=x.device.type, enabled=False):
            content_scores = q_c.float() @ k_c.float().transpose(-2, -1)
            position_scores = q_r.float() @ k_r.float().transpose(-2, -1)
            attn_scores = (content_scores + position_scores) / math.sqrt(self.d_head + self.d_rope)

         # --- D: Final Steps (Masking, Softmax, Output) ---
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len], float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights).to(v_c.dtype)

        # The final context vector is computed using only the content value matrix (v_c)
        context_vector = (attn_weights @ v_c).transpose(1 , 2).contiguous().view(
            batch_size , seq_len , self.d_model
        )
        output = self.W_o(context_vector)
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    layer = DeepSeekAttention(512, 8, 128, 64).to(args.device)
    x = torch.randn(4, 64, 512, device=args.device, requires_grad=True)
    output = layer(x)
    output.square().mean().backward()
    print(f"Device: {x.device}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("Forward and backward successful!")


if __name__ == "__main__":
    main()
