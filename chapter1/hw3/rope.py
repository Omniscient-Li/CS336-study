import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    把 d_k 维向量看成 d_k/2 个二维子空间（第 (0,1) 维一对、第 (2,3) 维一对、……）。第 i 对坐标 (x_0, x_1) 在二维平面上旋转角度 m·freq_i：

    x_0' = x_0·cos(m·freq_i) − x_1·sin(m·freq_i)
    x_1' = x_0·sin(m·freq_i) + x_1·cos(m·freq_i)
    其中 freq_i = θ^(−2i/d_k)，位置 m 的旋转角度 = m × freq_i。
    i=0 的维度对：freq = θ^0 = 1，相邻 token 的角度差最大 → 编码"相邻位置"的精细差异（高频，像钟的秒针）
    i 靠后的维度对：freq ≈ 1/θ，随距离变化极慢 → 编码长距离依赖（低频，像时针）
    不同频率叠加在一起，原理上类似傅里叶分解——每个"频道"负责不同的距离尺度
    为什么是相对位置：旋转是正交变换，(R_m·q)^T·(R_n·k) = q^T·R_{m-n}·k，m 和 n 同时出现在差值里。所以两个 token 的注意力分数只由它们的距离决定——语言模型的注意力恰好最关心相对距离
    公式是：
    out = x * cos(theta * position) - x * sin(theta * position)
    Args:
        theta (float): 底数超参数
        d_k (int): 输入的维度，也就是d_model
        max_seq_len (int): 最大序列长度
        device (torch.device): 设备
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        token_positions: (batch_size, seq_len) 每个token的位置信息
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even") #每两个维度配成一对做二维旋转，奇数维度没法配对。实际训练中 d_k 都是偶数（64、128）
        self.theta = theta  #这个是RoPE的底数超参数，不是直接的角度
        self.d_k = d_k #d_k就是d_model,即嵌入之后的稠密向量，它必须为偶数
        self.max_seq_len = max_seq_len
        self.device = device
        #计算频率 ：freq_i = 1 / θ^(−2i/d_k)
        freqs = 1.0 / (self.theta ** (torch.arange(0 , self.d_k , 2).float() / self.d_k))
        """
        torch.arange(0, d_k, 2) → [0, 2, 4, ..., d_k-2]，共 d_k/2 个（步长 2）
        除以 d_k → [0, 2/d, 4/d, ..., (d_k-2)/d]，在 [0, 1) 上均匀分布
        theta ** 这个 → [θ^0, θ^(2/d), θ^(4/d), ...]，指数线性增长
        取倒数 → [1, θ^(-2/d), θ^(-4/d), ..., ≈1/θ]，频率按几何级数递减，从 1 一路降到约 1/θ
        θ 就是控制衰减速度的底数（原始论文取 10000；θ 越大最低频越低，对超长序列的外推越友好）
        """
        #记录每个token的位置信息
        positions = torch.arange(self.max_seq_len)
        #计算正弦与余弦
        sinusoids = torch.outer(positions , freqs)  #outer是外积，即每个位置都与每个频率相乘 shape: [max_seq_len, d_k//2]
        self.register_buffer("cos_cache" , sinusoids.cos() , persistent = False) 
        #register_buffer：这个张量不是 nn.Parameter，所以不参与梯度、不被 optimizer 更新；但注册后它会跟着 model.to(device) / model.to(dtype) 自动搬设备转精度，这是它和普通 self.xxx = tensor 的区别
        # persistent=False：不写入 state_dict()。cos/sin 表是从 theta/d_k/max_seq_len 纯公式推出来的，checkpoint 里不用存，加载模型后 __init__ 会重算一遍——省存储空间
        self.register_buffer("sin_cache" , sinusoids.sin() , persistent = False)

    def forward(self , x : torch.Tensor , token_positions : torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions] # 高级索引查表：给出一批位置编号，直接取出这些行。比如 token_positions=[0,3,7] 就取出第 0、3、7 行的 cos/sin。
        sin = self.sin_cache[token_positions]

        # 让 cos/sin 的前导维数与 x 对齐，支持 (seq,) 或 (batch, seq) 两种 token_positions
        # 查表后 cos 形状是 (..., seq, d_k//2)，x 是 (batch, seq, d_k)，广播前补齐缺失的 batch 维。
        # 不用写死 unsqueeze(0)：token_positions 为 (batch, seq) 时 cos 已经是三维，再 unsqueeze 会多出一维导致形状错
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
        while sin.dim() < x.dim():
            sin = sin.unsqueeze(0)

        x_part1 = x[... , 0 :: 2] # 偶数维：每对的第一个坐标 x_0 ... 表示保留所有前导维度，0::2 是步长 2 切片
        x_part2 = x[... , 1 :: 2] # 奇数维：每对的第二个坐标 x_1

        output1 = x_part1 * cos - x_part2 * sin # 偶数位置乘以cos，奇数位置乘以sin
        output2 = x_part1 * sin + x_part2 * cos # 偶数位置乘以sin，奇数位置乘以cos
        # 逐元素并行：每一对的 x_0' 和 x_1'。cos/sin 在这里和 x_part 广播相乘——每个 batch、每个 token、每个维度对用的角度都不同（token 位置决定行，维度对决定列）
        out = torch.stack([output1 , output2] , dim = -1)# [batch, seq_len, d_k//2, 2] 
        out = out.flatten(-2)  # [batch, seq_len, d_k]
        return out
