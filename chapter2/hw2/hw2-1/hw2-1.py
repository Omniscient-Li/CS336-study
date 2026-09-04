import torch
import torch.distributed as dist
import time
import os
import argparse
import csv
import itertools
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import spawn

def setup(master_addr , master_port , rank , world_size , backend):
    # 初始化环境分布
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)# 设置主节点IP地址以及端口，其他节点需要通过这个地址连接到主节点
    # 手动创建 TCPStore：Windows 版 torch 默认请求 libuv 但没编译 libuv 会报错，显式 use_libuv = False
    store = dist.TCPStore(master_addr , master_port , world_size , rank == 0 , use_libuv = False)
    # 根据后端初始化进程组，rank是当前进程的rank，world_size是总进程数，backend: 这是指定通信后端的参数。常见的后端有：gloo(CPU), nccl(GPU), mpi等。
    dist.init_process_group(backend , rank = rank , world_size = world_size , store = store)

def cleanup():
    # 清理分布式环境
    dist.destroy_process_group()
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_all_reduce(rank , world_size , tensor_size_mb , backend , device , master_addr , master_port , queue = None):
    """
    评测函数主体
    """
    # 1. 设置环境和设备
    setup(master_addr , master_port , rank , world_size , backend)
    if device == 'cuda':
        # 将进程绑定在对应的GPU
        # 单卡机器跑多进程时 rank 可能超过 GPU 数量，取模映射回可用设备
        torch.cuda.set_device(rank % torch.cuda.device_count())
        # 清理内存防止冲突
        torch.cuda.empty_cache()

    # 2. 创建测试数据
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    # float32是4字节
    num_elements = tensor_size_bytes // 4
    tensor_data = torch.randn(num_elements , device = device)

    # 3. 预热
    for _ in range(5):
        dist.all_reduce(tensor_data , op = dist.ReduceOp.SUM)
        # 如果是GPU ， 需要同步等待完成
        if device == 'cuda':
            torch.cuda.synchronize()

    dist.barrier()
    # 4. 正式计时
    start_time = time.time()
    for _ in range(20):
        dist.all_reduce(tensor_data , op = dist.ReduceOp.SUM)

    # GPU需要同步保证操作完成
    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    duration = end_time - start_time
    avg_time = duration / 20

    # 计算带宽（所有 rank 都需要，以便返回结构一致）
    # 带宽 = 数据大小 / 时间（简化计算）
    bandwidth_gbps = (tensor_size_bytes / avg_time) / 1e9

     # 只在主进程打印结果
    if rank == 0:
        print(f"Backend: {backend}, Device: {device}, World Size: {world_size}, Tensor Size: {tensor_size_mb}MB")
        print(f"Average time per all-reduce: {avg_time * 1000:.4f} ms")
        print(f"Achieved Bandwidth: {bandwidth_gbps:.4f} GB/s\n")

    local_result = {
            'rank': rank,
            'world_size': world_size,
            'backend': backend,
            'device': device,
            'tensor_size_mb': tensor_size_mb,
            'avg_time_ms': avg_time * 1000.0,
            'bandwidth_gbps': bandwidth_gbps
        }
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)

    # 最后清理环境
    cleanup()
    # spawn 不会把函数返回值传回主进程，rank 0 通过 Queue 把结果送回去
    if rank == 0:
        if queue is not None:
            queue.put(gathered_results)
        return gathered_results


def parse_args():
    parser = argparse.ArgumentParser(description="单机多进程 all-reduce 基准测试")
    parser.add_argument("--world-sizes", type=str, default="2,4,6",
                        help="逗号分隔的进程数列表，如 2,4,6")
    parser.add_argument("--tensor-sizes", type=str, default="1,10,100,1000",
                        help="逗号分隔的张量大小列表（MB），如 1,10,100,1000")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="cpu 用 gloo 后端，cuda 用 nccl 后端")
    parser.add_argument("--backend", type=str, default=None, choices=["gloo", "nccl"],
                        help="默认按 device 自动配对；可覆盖（如单卡多进程 nccl 会报 Duplicate GPU，改用 --backend gloo）")
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=29500,
                        help="起始端口，每组配置 +1 避免 TIME_WAIT 冲突")
    return parser.parse_args()


def main():
    args = parse_args()
    world_sizes = [int(x) for x in args.world_sizes.split(",")]
    tensor_sizes_mb = [int(x) for x in args.tensor_sizes.split(",")]
    # 设备与后端自动配对：GPU 用 nccl，CPU 用 gloo（--backend 可覆盖）
    backend = args.backend or ("nccl" if args.device == "cuda" else "gloo")

    all_rows = []
    configs = list(itertools.product(world_sizes, tensor_sizes_mb))
    for idx, (world_size, size_mb) in enumerate(configs):
        print(f"\n[{idx + 1}/{len(configs)}] world_size={world_size}, tensor={size_mb}MB, "
              f"backend={backend}, device={args.device}")
        # 必须用 spawn 上下文创建 Queue，否则跨平台传递会出问题
        queue = mp.get_context("spawn").Queue()
        # 每组配置用独立端口，避免上一个进程组销毁后端口 TIME_WAIT 导致偶发失败
        port = args.master_port + idx
        spawn(
            benchmark_all_reduce,
            args=(world_size, size_mb, backend, args.device, args.master_addr, port, queue),
            nprocs=world_size,
            join=True,
        )
        # spawn 不传回函数返回值，rank 0 的结果通过 Queue 回传
        gathered = queue.get(timeout=600)
        all_rows.extend(gathered)

    # 汇总表：同一配置下对多个 rank 取平均（官方建议聚合各 rank 的测量）
    print("\n" + "=" * 70)
    print(f"{'world_size':>10} {'tensor(MB)':>10} {'avg_time(ms)':>14} {'bandwidth(GB/s)':>16}")
    print("-" * 70)
    summary = []
    for world_size in world_sizes:
        for size_mb in tensor_sizes_mb:
            rows = [r for r in all_rows
                    if r["world_size"] == world_size and r["tensor_size_mb"] == size_mb]
            mean_time = sum(r["avg_time_ms"] for r in rows) / len(rows)
            mean_bw = sum(r["bandwidth_gbps"] for r in rows) / len(rows)
            summary.append({"world_size": world_size, "tensor_size_mb": size_mb,
                            "avg_time_ms": mean_time, "bandwidth_gbps": mean_bw})
            print(f"{world_size:>10} {size_mb:>10} {mean_time:>14.4f} {mean_bw:>16.4f}")

    # 保存两个 CSV：逐 rank 明细 + 聚合汇总
    with open("all_reduce_benchmark_detail.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open("all_reduce_benchmark_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print("=" * 70)
    print("已保存 all_reduce_benchmark_detail.csv / all_reduce_benchmark_summary.csv")


if __name__ == "__main__":
    main()
