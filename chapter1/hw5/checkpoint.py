import torch

def save_checkpoint(model , optimizer , iteration , out):
    """
    model	模型实例（如 TransformerBlock）
    optimizer	优化器实例（如 AdamW），包含动量 m、RMS v 等状态
    iteration	当前训练步数，用于恢复学习率调度和偏差修正
    out	保存路径
    """
    torch.save({
        'model_state_dict' : model.state_dict() , 
        'optimizer_state_dict' : optimizer.state_dict() , 
        'iteration' : iteration
    } , out)
    """
    ① model.state_dict() — 模型的所有权重参数
    这是一个 OrderedDict，key 是层的名称，value 是参数张量
    注意：state_dict() 只保存可学习参数（nn.Parameter），不保存 buffer（除非 persistent=True）和模型结构信息。这意味着加载时需要先用代码重建同结构的模型。
    ② optimizer.state_dict() — 优化器的全部状态
    为什么要保存优化器状态？ 如果丢失这些状态，从断点继续训练时优化器等于"失忆"了，前几千步的动量积累白费，训练轨迹完全不同
    ③ iteration — 训练步数 
    一个整数。恢复训练后学习率调度器需要知道当前是第几步
    """


def load_checkpoint(src , model , optimizer):
    checkpoint = torch.load(src)  #反序列化
    model.load_state_dict(checkpoint['model_state_dict']) #恢复模型权重
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #恢复模型优化器状态
    iteration = checkpoint['iteration'] 
    return iteration