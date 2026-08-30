import torch

class Autograd_function_pytorch(torch.autograd.Function):
    """
    FlashAttention 的 PyTorch 自定义 autograd Function 实现（分块 online-softmax 版本）。
    torch.autograd.Function是pytorch给"自定义前向+反向算子"准备的基类。普通的nn.Module使用现成的算子拼网络

    输入形状：
    Query (batch_size, Nq, d)
    Key   (batch_size, Nk, d)
    Value (batch_size, Nk, d)

    前向返回 O（官方接口只返回一个输出）：
    O: (batch_size, Nq, d)  注意力输出
    L: (batch_size, Nq)     每行的 log-sum-exp（softmax 分母取对数），不返回、通过 save_for_backward 传给反向
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        #ctx:第一个参数固定是context（上下文），forward里算出来，backward需要的中间数据放进去
        batch_size = q.shape[0]
        Nq = q.shape[1]  # q 的序列长度
        Nk = k.shape[1]  # k 的序列长度
        d = q.shape[2]   # 特征维度
        """
        FlashAttention 的核心思想：注意力得分矩阵 S 的形状是 (Nq, Nk)，序列越长它越大（例如 4096×4096）。不一次算整个 S，而是切成 64×64 的小块，一次只算一块。64 来自 GPU 的 SRAM 缓存容量——一块 64×64 的浮点矩阵刚好塞进缓存，再大就要溢到慢速内存。
        """

        Bq = 64  # q 的分块大小
        Bk = 64  # k 的分块大小
        Tq = Nq // Bq
        Tk = Nk // Bk

        # 初始化张量
        O = torch.zeros_like(q)  # (batch_size, Nq, d) 创建一个形状、dtype、device 都和 q 完全一样的全零张量。形状 (batch, Nq, d)。O 是最终输出（注意力结果），现在先占位，后面分块填充。
        L = torch.zeros(batch_size, Nq, device=q.device)  # (batch_size, Nq) L 是每行 softmax 分母的对数 形状 (batch, Nq)——注意没有 d 维，因为 L 是"每行一个标量"（每行存一个 log-sum-exp 值）。device=q.device 保证和 q 在同一张卡上（GPU 上的 q 要配 GPU 上的 L，否则后面赋值报错）。

        # 对每个批次分别处理
        for b in range(batch_size):
            q_batch = q[b]  # (Nq, d)
            k_batch = k[b]  # (Nk, d)
            v_batch = v[b]  # (Nk, d)
            """
            标准 softmax 分两步：先找整行最大值 m，再算 exp(S-m)/Σexp(S-m)。但分块后 k 的块是一个一个流进来的，最大值可能在还没看到的块里。online softmax 的解法是维护一个可以"事后修正"的状态 (m, l, O)：

            m：目前为止见过的行最大值
            l：Σexp(S - m)——在"当前基准 m"下的 exp 行和
            O：Σexp(S - m)·v——同样基准下的加权和
            """

            for i in range(Tq):  # 外层循环：q 的分块
                max_S_ij_last = torch.full((Bq, 1), float('-inf'), device=q.device)  # (Bq, 1) 行最大值 m
                l_ij = torch.zeros(Bq, 1, device=q.device)  # (Bq, 1) exp的行和 状态量 l：到目前为止 Σexp(S − m) 的行和（m 是当前最大值）。
                O_ij = torch.zeros(Bq, d, device=q.device)  # (Bq, d) 未归一化的输出 状态量O：Σexp(S − m)·v 的加权和（同一个基准 m 下）。初始零矩阵。注意它是 (64, d)——每行是 d 维向量，对应 64 个 query 位置各自的"未归一化输出"。
                q_i = q_batch[i * Bq : (i + 1) * Bq, :]
                for j in range(Tk):  # 内层循环：K, V 的分块
                    # causal 时 q 块 i 前面的 k 块 j（j > i）整块都是"未来"位置，直接跳过
                    if is_causal and j > i:
                        continue
                    k_j = k_batch[j * Bk : (j + 1) * Bk, :]
                    v_j = v_batch[j * Bk : (j + 1) * Bk, :]
                    S_ij = q_i @ k_j.T / d ** 0.5  # (Bq, Bk)
                    # causal 时块内还要 mask 掉上三角：位置 (r, c) 有效当且仅当 i*Bq+r >= j*Bk+c
                    if is_causal:
                        row_idx = torch.arange(Bq, device=q.device)[:, None] + i * Bq
                        col_idx = torch.arange(Bk, device=q.device)[None, :] + j * Bk
                        S_ij = S_ij.masked_fill(row_idx < col_idx, float('-inf'))
                    max_S_ij_now = torch.max(max_S_ij_last, torch.max(S_ij, dim=1)[0].unsqueeze(1))  # (Bq, 1)
                    # 接下来需要 S 的每一行都减去对应行的最大值（online softmax 的稳定化）
                    P_ij = torch.exp(S_ij - max_S_ij_now)  # (Bq, Bk)
                    # 旧累加量按 exp(m_old - m_new) 缩放，再并入当前块
                    l_ij = torch.exp(max_S_ij_last - max_S_ij_now) * l_ij + torch.sum(P_ij, dim=1).unsqueeze(1)
                    O_ij = torch.exp(max_S_ij_last - max_S_ij_now) * O_ij + P_ij @ v_j  # (Bq, d)
                    max_S_ij_last = max_S_ij_now
                # 内循环结束，l_ij 就是该 q 块各行 softmax 分母的和，除以它完成归一化
                O_i = O_ij / l_ij  # (Bq, d) 广播除法，等价于 torch.diag(1/l) @ O_ij 但更省内存
                L_i = max_S_ij_now.squeeze(1) + torch.log(l_ij.squeeze(1))  # log-sum-exp: m + log(l)

                # 将结果存储到对应的批次和分块位置
                O[b, i * Bq : (i + 1) * Bq, :] = O_i
                L[b, i * Bq : (i + 1) * Bq] = L_i

        # 保存前向传播需要的张量（官方测试会从 saved_tensors 里按形状 (batch, Nq) 找到 L）
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal

        # 官方接口：forward 只返回 O 这一个输出；L 通过 save_for_backward 传给反向
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        """
        return: dq, dk, dv
        """
        batch_size, Nq, d = Q.shape
        _, Nk, _ = K.shape
        Bq = 64
        Bk = 64
        Tq = Nq // Bq
        Tk = Nk // Bk

        # softmax 反传的经典技巧：dS = P ⊙ (dP - D)，其中 D 是每行 (dO ⊙ O) 的行和
        D = torch.sum(O * dO, dim=-1)  # (batch_size, Nq)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for i in range(Tq):
            # 按 Bq 分块在第一个循环里处理（和前向一样的分块方式）
            q_i = Q[:, i * Bq : (i + 1) * Bq, :]  # (batch_size, Bq, d)
            dO_i = dO[:, i * Bq : (i + 1) * Bq, :]  # (batch_size, Bq, d)
            L_i = L[:, i * Bq : (i + 1) * Bq]  # (batch_size, Bq)
            D_i = D[:, i * Bq : (i + 1) * Bq]  # (batch_size, Bq)

            for j in range(Tk):
                # 前向里被跳过的块（j > i）反向里也没有任何梯度贡献
                if is_causal and j > i:
                    continue
                # 按 Bk 分块在第二个循环里处理
                k_j = K[:, j * Bk : (j + 1) * Bk, :]
                v_j = V[:, j * Bk : (j + 1) * Bk, :]

                S_ij = q_i @ k_j.transpose(-2, -1) / (d ** 0.5)  # (batch_size, Bq, Bk)
                # 前向里 mask 过的位置 exp 出来必须是 0（梯度也是 0），掩码要和前向完全一致
                if is_causal:
                    row_idx = torch.arange(Bq, device=Q.device)[:, None] + i * Bq
                    col_idx = torch.arange(Bk, device=Q.device)[None, :] + j * Bk
                    S_ij = S_ij.masked_fill(row_idx < col_idx, float('-inf'))
                # 用前向存的 log-sum-exp L 重算 P = softmax(S)（不保存全部 P，省显存）
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (batch_size, Bq, Bk)

                # dV = P^T @ dO：V 的第 j 个分块在前向里和每个 q 分块都作用过，所以这里是 += 累计
                dV_j = P_ij.transpose(-2, -1) @ dO_i  # (batch_size, Bk, d)
                dV[:, j * Bk : (j + 1) * Bk, :] += dV_j

                dP_ij = dO_i @ v_j.transpose(-2, -1)  # (batch_size, Bq, Bk)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # softmax 的反向

                dQ_i_j = (dS_ij @ k_j) / (d ** 0.5)
                dQ[:, i * Bq : (i + 1) * Bq, :] += dQ_i_j

                dK_j = (dS_ij.transpose(-2, -1) @ q_i) / (d ** 0.5)
                dK[:, j * Bk : (j + 1) * Bk, :] += dK_j

        # forward 的参数是 (q, k, v, is_causal)，所以返回四个梯度；is_causal 是 bool，梯度为 None
        return dQ, dK, dV, None
