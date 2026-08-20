import torch
import torch.nn as nn
# 优先相对导入（hw3 作为包导入时），失败则回退为同目录导入（直接按文件跑时）
try:
    from .rope import RoPE
except ImportError:
    from rope import RoPE

class CausalMultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self , d_model : int , n_heads : int , max_seq_len : int , theta : float , device = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(theta , self.head_dim , max_seq_len)
        #这里要注意，我们使用多头注意力机制的时候，每个head的维度是d_model // n_heads，我们应当对每个head进行RoPE

    def attention(self , Q : torch.Tensor , K : torch.Tensor , V : torch.Tensor , mask : torch.Tensor | None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q , K.transpose(-2 , -1) / torch.sqrt(torch.tensor(d_k)))
        if mask is not None:
            # mask 为 triu(上三角)：True = 未来位置，必须填 -1e9 屏蔽（+1e-9 在 softmax 里等于没屏蔽）
            scores = scores.masked_fill(mask , -1e9)
        attn_weights = torch.softmax(scores , dim = -1)
        return torch.matmul(attn_weights , V)

    def forward(self , x , wq , wk , wv , wo , token_positions) -> torch.Tensor:
        batch_size , seq_len , d_model = x.shape

        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = q.view(batch_size , seq_len , self.n_heads , self.head_dim)
        k = k.view(batch_size , seq_len , self.n_heads ,self.head_dim)
        v = v.view(batch_size , seq_len , self.n_heads , self.head_dim)

        q = q.transpose(1 , 2)
        k = k.transpose(1 , 2)
        v = v.transpose(1 , 2)

        q = self.rope(q , token_positions)
        k = self.rope(k , token_positions)

        #现在的形状是(batch_size, n_heads, seq_len, head_dim)
        # 创建mask，用于防止当前位置的token看到未来的token。
        mask = torch.triu(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        out = out @ wo.T
        return out