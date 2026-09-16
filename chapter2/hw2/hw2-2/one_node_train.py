import torch
import torch.optim as optim
from torchvision import datasets , transforms
import numpy as np
import random
from model import SimpleNet

def train(model , device , train_loader , optimizer , epoch):
    model.train()
    for batch_idx , (data , target) in enumerate(train_loader):
        data , target = data.to(device) , target.to(device)
        optimizer.zero_grad() # 清理上一轮的梯度
        output = model(data) # 前向传播
        loss = torch.nn.functional.nll_loss(output , target) # 计算loss
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # 设置随机种子确保可重现性
    seed = 1234
    torch.manual_seed(seed) # CPU上的tensor采样
    torch.cuda.manual_seed(seed) # 当前GPU上的采样
    torch.cuda.manual_seed_all(seed) # 所有GPU（多卡使用）
    np.random.seed(seed) # Numpy采样
    random.seed(seed)

    """
    设置确定性行为 benchmark=False：默认 True 时 cuDNN 会在运行时对卷积算法做基准测试挑最快的——但选择结果随机器而异，破坏可复现性
    deterministic=True：强制用确定性算法（某些 cuDNN 算子的默认实现是非确定的，反向尤其明显 
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device : {device}")

    transform = transforms.Compose([
        transforms.ToTensor() ,
        transforms.Normalize((0.1307 , ) , (0.3081 , ))
    ])
    dataset1 = datasets.MNIST('../data' , train = True , download = True , 
                              transform = transform)
    # 设置确定性的数据加载器
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(dataset1 , batch_size = 64 , shuffle = True , generator = generator)

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters() , lr = 0.001)

    for epoch in range(1 , 3):
        train(model , device , train_loader , optimizer , epoch)

        torch.save(model.state_dict() , "mnist_simple.pt")

if __name__ == '__main__':
    main() 