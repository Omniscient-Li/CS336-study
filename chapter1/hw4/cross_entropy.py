import torch


class CrossEntropyloss:
    def __init__(self , inputs , targets):
        self.inputs = inputs #logits , shape : [batch_size , vocab_size]
        self.targets = targets # true , shape : [batch_size]
        self.vocab_size = inputs.shape[1]
        self.batch_size = inputs.shape[0]

    def forward(self):
        # 数值稳定的 log-softmax：作业要求"log 和 exp 能约掉就约掉"，不能先算 softmax 再取 log
        # （大 logit 下 exp 会下溢成 0，log(0) = -inf——这正是官方测试大输入样例报 inf 的原因）
        # 第一步：减去每个样本最大的 logit，保证 exp 不溢出（和 hw3 softmax 同一个技巧）
        max_logits = self.inputs.max(dim = 1 , keepdim = True).values
        # log Σᵢ exp(xᵢ − max)：对数-求和-指数，先减 max 再 exp 再 log，全程数值稳定
        logsumexp = max_logits + torch.log(torch.sum(torch.exp(self.inputs - max_logits) , dim = 1 , keepdim = True))
        # 恒等式：log(softmax(x)) = x − logsumexp(x)，一步得到对数概率，无需经过 softmax
        log_probs = self.inputs - logsumexp
        p = log_probs[range(self.batch_size) , self.targets]
        """
        高级索引（fancy indexing），从概率分布中取出正确类别的概率。

        range(batch_size) = [0, 1, 2]      ← 行索引
        targets           = [0, 1, 3]      ← 列索引

        y_pred = [[0.52, 0.04, 0.21, 0.09, 0.03],   ← 第 0 行
                 [0.01, 0.65, 0.12, 0.08, 0.14],    ← 第 1 行
                 [0.01, 0.05, 0.03, 0.82, 0.09]]    ← 第 2 行

        p = y_pred[(0,0), (1,1), (2,3)]
          = [y_pred[0][0], y_pred[1][1], y_pred[2][3]]
          = [0.52,         0.65,         0.82]
        三个样本各自正确类别的概率分别是 52%、65%、82%
        """
        # 官方测试对拍的是 F.cross_entropy（默认 mean）：loss = −1/N × Σ(−log pᵢ)
        return -torch.mean(p)
        
