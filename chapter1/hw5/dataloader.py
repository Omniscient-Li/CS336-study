import random
import numpy as np
from typing import List
import torch

class DataLoader:
    def __init__(self , data : List[int] , batch_size : int , context_length : int , shuffle = True):
        """
        context_length 就是 hw3 里 TransformerLM 的 context_length——一个样本就是喂给模型的一个 (batch, seq) 矩阵里的一行。
        data 可以是 Python list 也可以是 numpy 数组，后面切片两种都兼容
        """
        self.data = data # 一整条长 token 序列（一维）
        self.data_len = len(data)
        self.batch_size = batch_size # 每个 batch 含几个样本
        self.shuffle = shuffle
        self.context_length = context_length # 每个样本多长（= 模型能看到的最长上下文）

    def get_train_batch_data(self):
        """
        随机获取一个batch的数据，y 正好是 x 向右移动一个位置的结果。模型在看到 x 的第 i 个位置的输入时，需要努力预测出 y 在第 i 个位置的词元，这正是我们想要的“预测下一个词”的效果。
        """
        # 合法起始索引范围是 [0, data_len - context_length - 1]（共 data_len - context_length 个）。
        # randint 的高界是开区间，所以要传 data_len - context_length——
        # 多减 1 会让最后一个合法起始位置永远采不到，官方测试的 max(start)==92 断言会失败
        idxs = np.random.randint(0 , self.data_len - self.context_length , size = (self.batch_size ,))
        # 一次采 batch_size 个随机位置 i，每个位置切一段长度为 context_length 的序列
        x = np.stack([self.data[i : i + self.context_length] for i in  idxs])
        y = np.stack([self.data[i + 1 : i + self.context_length + 1] for i in idxs])
        # np.stack 把它们叠成一个 (batch_size, context_length) 的二维数组——正是模型要的输入形状
        return torch.tensor(x) , torch.tensor(y)

    def get_valid_batch_data_iter(self):
        """
        验证集的区别之处在于，不需要随机选择，而是直接从数据集中按顺序获取所有数据，并用迭代器返回
        """
        # 总 batch 数：每个 batch 覆盖 batch_size + context_length 个 token，推导可得 floor((data_len - context_length) / batch_size)
        start_num = (self.data_len - self.context_length) // self.batch_size # 表示有多少个batch
        for i in range(start_num):
            bias = i * self.batch_size # 表示每一个batch开始的位置
            # 切片里必须用循环变量 j：让样本在 batch 内逐位前进（stride=1 的滑窗），
            # 否则 batch 内 batch_size 行全是同一段数据（原来的 bug）
            x = np.stack([self.data[bias + j : bias + j + self.context_length] for j in range(self.batch_size)])
            y = np.stack([self.data[bias + j + 1 : bias + j + self.context_length + 1] for j in range(self.batch_size)])
            yield torch.tensor(x) , torch.tensor(y)

    def __len__(self):
        """
        返回数据集的batch数量
        """
        return self.data_len // self.batch_size
 