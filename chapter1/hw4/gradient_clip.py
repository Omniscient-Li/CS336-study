import torch

class GradientClip:
    def __init__(self , parameters , max_l2_norm , epsilon = 1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.epsilon = epsilon

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None] # ① 收集 模型里不是每个参数都有梯度——冻结层（requires_grad=False）或本次前向没参与计算的参数，p.grad 是 None。直接对 None 做 flatten 会崩，所以先过滤。
        all_grads = torch.cat([grad.flatten() for grad in grads]) # ② 拼成一个向量 即把每个梯度张量拉平成向量、首尾相接，再算这一个长向量的范数。flatten() 拉平、cat 拼接，正是这个公式的直接实现。注意这里所有梯度共享同一个缩放系数，所以必须算总的范数，而不是每个参数各自的范数
        grad_l2 = torch.norm(all_grads , 2) # ③ 算总 L2 范数
        if grad_l2 > self.max_l2_norm: # ④ 超限才动
            clip_coeff = self.max_l2_norm / (grad_l2 + self.epsilon) # ⑤ 算缩放系数
            for grad in grads:
                grad.mul_(clip_coeff) # ⑥ 整体缩放
        