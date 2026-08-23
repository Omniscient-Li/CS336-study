import torch
import torch.nn as nn
from rope import RoPE


class CausalMultiHeadAttentionNoWeight(nn.Module):
    def __init__(self , d_model : int , n_heads : int , max_seq_len : int , theta : float , device = None):
        super().__init__() # 必须先初始化基类，才能往 self 上挂 nn.Linear 等子模块
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(theta , self.head_dim , max_seq_len) #使用多头注意力机制的时候，每个head的维度是d_model // n_heads，应当对每个head进行RoPE

        self.wq = nn.Linear(d_model , d_model , bias = False)
        self.wk = nn.Linear(d_model , d_model , bias = False)
        self.wv = nn.Linear(d_model , d_model , bias = False)
        self.wo = nn.Linear(d_model , d_model , bias = False)

    def attention(self , Q : torch.Tensor , K : torch.Tensor , V : torch.Tensor , mask : torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q , K.transpose(-2 , -1)) / (d_k ** 0.5) # 缩放因子用普通浮点，避免每次 forward 都新建 CPU 张量
        if mask is not None:
            scores = scores.masked_fill(mask , float("-inf")) # 必须填 -inf：softmax(-inf)=0，被屏蔽位置贡献才真正为 0；填正值会让未来 token 反而主导注意力
        attn_weights = torch.softmax(scores , dim = -1)
        return torch.matmul(attn_weights , V)

    def forward(self , x , token_positions) -> torch.Tensor:
        batch_size , seq_len , d_model = x.shape

        q = self.wq(x) # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = self.wk(x) # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = self.wv(x) # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = q.view(batch_size , seq_len , self.n_heads , self.head_dim)
        k = k.view(batch_size , seq_len , self.n_heads , self.head_dim)
        v = v.view(batch_size , seq_len , self.n_heads , self.head_dim)

        q = q.transpose(1 , 2)
        k = k.transpose(1 , 2)
        v = v.transpose(1 , 2)

        q = self.rope(q , token_positions)
        k = self.rope(k , token_positions)

        mask = torch.triu(torch.ones(seq_len , seq_len , dtype = torch.bool , device = x.device) , diagonal = 1) # torch.boll -> torch.bool
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        out = self.attention(q , k , v , mask)
        out = out.transpose(1 , 2)
        out = out.contiguous().view(batch_size , seq_len , d_model)

        return self.wo(out)