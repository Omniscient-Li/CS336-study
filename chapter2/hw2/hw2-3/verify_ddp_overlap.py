"""
DDPOverlapBucketed 验证脚本（对照实验，模仿官方 test_ddp.py 的验证逻辑）。

- 2 个进程（gloo + CPU，官方测试同款配置）
- 非并行基线：20 个样本一次前向/反向/step
- DDP：每个 rank 各看 10 个样本（disjoint）→ finish_gradient_synchronization → step
- 断言：5 轮后 DDP 参数与基线逐元素一致（allclose）
- 小桶（0.01 MB）强制把 6 个参数切成 4 个桶，检验 bucketing 逻辑
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from copy import deepcopy
from torchvision import datasets, transforms

from ddp_overlap_bucketed import DDPOverlapBucketed, SimpleNet


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    device = torch.device('cpu')

    # --- 数据：MNIST 前 20 个样本（已在 A100 上缓存） ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST('../data', train=True, download=False, transform=transform)
    all_x = torch.stack([ds[i][0] for i in range(20)])
    all_y = torch.tensor([ds[i][1] for i in range(20)])

    # --- 不同初始化（seed=rank），验证构造时广播 ---
    torch.manual_seed(rank)
    baseline = SimpleNet()  # 非并行基线
    ddp = DDPOverlapBucketed(deepcopy(baseline), bucket_size_mb=0.01)

    # 构造广播验证：两 rank 参数必须完全一致
    for p in ddp.parameters():
        ref = p.data.clone()
        dist.broadcast(ref, src=0)
        assert torch.allclose(p.data, ref), f'rank {rank}: parameters not broadcast-synced'
    if rank == 0:
        assert all(torch.allclose(a.data, b.data) for a, b in zip(ddp.parameters(), baseline.parameters()))
        print(f'[rank 0] broadcast check passed, {len(list(ddp.parameters()))} params, '
              f'{sum(1 for b in ddp.buckets if b["params"])} buckets created')

    loss_fn = F.nll_loss
    opt_ddp = torch.optim.SGD(ddp.parameters(), lr=0.1)
    opt_base = torch.optim.SGD(baseline.parameters(), lr=0.1)

    for i in range(5):
        opt_ddp.zero_grad()
        opt_base.zero_grad()

        # 非并行基线：全量 20 个样本
        out = baseline(all_x)
        loss = loss_fn(out, all_y)
        loss.backward()
        opt_base.step()

        # DDP：每个 rank 10 个样本（disjoint）
        offset = rank * 10
        out_d = ddp(all_x[offset:offset + 10])
        loss_d = loss_fn(out_d, all_y[offset:offset + 10])
        loss_d.backward()
        ddp.finish_gradient_synchronization()
        opt_ddp.step()

        # 每轮后 DDP 必须 == 基线
        for bp, dp in zip(baseline.parameters(), ddp.parameters()):
            ref = bp.data.clone()
            dist.broadcast(ref, src=0)
            diff = (ref - dp.data).abs().max().item()
            assert torch.allclose(ref, dp.data, atol=1e-6), f'rank {rank} step {i}: DDP diverged (max diff {diff:.2e})'

    if rank == 0:
        print('VERIFY PASSED: 5 steps, DDP parameters match non-parallel baseline exactly')
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
