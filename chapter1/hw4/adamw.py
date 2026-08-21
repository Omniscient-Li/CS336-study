from torch import optim
import torch

class AdamW(optim.Optimizer):
    def __init__(self , params , lr , betas , eps , weight_decay):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        """
        继承 optim.Optimizer 是为了免费获得参数组（param_groups）机制：params 可以是普通张量列表，也可以是 [{'params': ..., 'lr': 0.01}, {'params': ..., 'lr': 0.001}] 这样的字典列表——不同层用不同学习率，基类帮你管理。
        defaults 字典存储所有超参数的默认值
        params 是要优化的参数（通常是 model.parameters()）
        父类 Optimizer 会把这些参数组织成 param_groups——支持为不同层设置不同的学习率
        注意 betas 是一个二元组，惯例是 (β₁, β₂) = (0.9, 0.999)：β₁ 控制惯性大小，β₂ 控制对"波动历史"的记忆长度。
        """
        super().__init__(params , defaults)

    @torch.no_grad() #优化器的更新不在计算图里——我们直接用 .data 就地改参数。这个装饰器保证整个 step 过程中不会不小心构建出 autograd 图（否则每次更新都会把图连起来，内存爆炸）。
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups: #遍历每个参数组
            for p in group['params']: #遍历组内每个参数张量
                if p.grad is None: #跳过没有梯度的参数
                    continue
                grad = p.grad.data
                state = self.state[p] #每个参数独立的优化器状态字典
                """
                self.state 是一个 dict，以参数张量本身为 key。这确保了同一个参数在多次 step 调用中，动量和 RMS 状态是持久的。
                
                """
                #print(state)
                #初始化状态
                if len(state) == 0: #第一次调用step
                    state['step'] = 0 #当前步数计数器
                    state['m'] = torch.zeros_like(p.data) #一阶动量，和参数同形状的全零张量
                    state['v'] = torch.zeros_like(p.data) #二阶动量，和参数同形状的全零张量
                    """
                    torch.zeros_like(p.data) 创建与参数形状、数据类型、设备完全相同的全零张量。
                    比如 weight 是 [512, 2048] float32，那 m 和 v 也是 [512, 2048] float32
                    """

                m , v = state['m'] , state['v']
                beta1 , beta2 = group['betas']

                state['step'] += 1

                #Adam更新
                m.mul_(beta1).add_(grad , alpha = 1 - beta1) #mul_ 和 add_ 都是inplace操作，就地修改
                """
                m = m * beta1 -> mul_(beta1): 旧动量按 β₁ 衰减
                m = m + (1 - beta1) * grad -> add_(grad, alpha=1-beta1): 加入当前梯度
  
                """
                v.mul_(beta2).add_(grad.pow(2), alpha =1 - beta2)
                #修正偏差
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1 
                
                denom = (v / bias_correction2).sqrt().add_(group['eps']) 
                p.data.addcdiv_(m, denom, value=-step_size) #先乘后除
                """
                addcdiv_ 的语义：tensor.addcdiv_(c, a, b) = tensor += c × a / b
                所以这两行等价于
                v̂ = v / (1 - β₂^t)    v 的偏差修正
                denom = √v̂ + ε         分母：RMS + 小常数防除零
                p = p - (lr / bias_correction1) × m / denom 完整更新

                """

                # 解耦权重衰减,直接减，不要修改梯度
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                """
                等价于：p = p - lr × λ × p = p × (1 - lr × λ)
                这是 AdamW 区别于 Adam 的核心：weight decay 直接作用于参数本身，不经过动量 m 和 RMS v
                普通 Adam:   g' = g + λ·θ     → 权重衰减被"污染"了，和自适应学习率耦合
                AdamW:       θ' = θ - lr·λ·θ  → 干净、独立的参数收缩，和自适应学习率完全解耦

                """