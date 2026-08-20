import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""
原始的输入形状是(batch_size, seq_len, d_model)
0.对于输入，要先用W_q, W_k, W_v 线性变换得到q,k,v
1.如果有n_heads个头，就把最后一个维度切分成n_heads份，q,k,v每一部分都切分成 d_model//n_heads 的维度，
2.对于每个头，对于q,k,v都去做attention操作。
3.最后把所有的头按照最后一个维度concat起来，然后做一次线性变换。
"""

class CausalMultiHeadAttention(nn.Module):
    """
    CausalMultiHeadAttention 是因果多头注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    每个头的公式都是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        wq: (d_model, d_k) 查询的权重
        wk: (d_model, d_k) 键的权重
        wv: (d_model, d_v) 值的权重
        wo: (d_model, d_model) 输出的权重
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self , d_model , n_heads):
        super(CausalMultiHeadAttention , self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def attention(self , Q : torch.Tensor , K : torch.Tensor , V : torch.Tensor , mask : torch.Tensor | None = None):
        # 缩放因子必须是每头的维度 head_dim = Q.shape[-1]，不是 d_model
        d_k = Q.shape[-1]
        scores = torch.matmul(Q , K.transpose(-2 , -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            # mask 是 triu(对角线上移1) 得到的：True = 未来位置(j>i)，False = 可以关注的位置(j<=i)
            # 所以是"mask 本身为 True 的位置"填 -1e9（屏蔽未来），不是 mask==0 的位置
            scores = scores.masked_fill(mask , -1e9)
        attn_weights = torch.softmax(scores , dim = -1)  #对key这一维度进行softmax归一化
        return torch.matmul(attn_weights , V) # 将attn_weights与value相乘得到最终的输出

    def forward(self , x , wq , wk , wv , wo) -> torch.Tensor:
        batch_size , seq_len , d_model = x.shape

        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = q.view(batch_size , seq_len , self.n_heads , self.head_dim)
        k = k.view(batch_size , seq_len , self.n_heads , self.head_dim)
        v = v.view(batch_size , seq_len , self.n_heads , self.head_dim)

        q = q.transpose(1 , 2)
        k = k.transpose(1 , 2)
        v = v.transpose(1 , 2)
        #现在的形状是(batch_size, n_heads, seq_len, head_dim)
        # 创建因果mask：triu(diagonal=1) 的上三角为 True，即 (i,j) 中 j>i 的未来位置。
        # 这样在 attention 里 masked_fill(mask, -1e9) 就把"未来"屏蔽掉，只留下 j<=i（过去和自身）
        mask = torch.triu(torch.ones(seq_len , seq_len , dtype = torch.bool , device = x.device) , diagonal = 1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self.attention(q , k , v , mask)
        out = out.transpose(1 , 2)
        out = out.contiguous().view(batch_size , seq_len , d_model)
        out = out @ wo.T
        return out