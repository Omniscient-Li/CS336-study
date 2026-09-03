import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # chapter2/hw1/ 上三级才是仓库根
for hw in ["hw3", "hw4", "hw5", "hw7"]:
    sys.path.insert(0, str(REPO_ROOT / "chapter1" / hw))

import wandb
import time
import torch
import json
import pickle
import os
import argparse
import numpy as np
import torch.nn.functional as F
from dataloader import DataLoader            # hw5
from transformermodule import TransformerModule  # hw7
from adamw import AdamW                      # hw4
from lr_cosine_shedule import CosineSchedule # hw4
# from transformer import get_linear_schedule_with_warmup
import timeit
import torch.cuda.nvtx as nvtx



timestamp = time.strftime("%Y%m%d_%H%M%S")

def wandb_init(d_model, d_ff, n_layers, n_heads, batch_size, epochs, train_steps):
    # 无 wandb 账号时（A100 上）设置 WANDB_MODE=disabled，跳过登录；本地有账号则正常登录
    if os.environ.get("WANDB_MODE", "") != "disabled":
        wandb.login()
    run = wandb.init(project="cs336_final_train", 
                    config = {
                    # Experiment
                    "experiment_name": f"tinystories_17M_{timestamp}",
                    "total_tokens_processed": 327_680_000,
                    
                    # Data
                    "train_data_path": "../data/TinyStoriesV2-GPT4-train.txt",
                    "valid_data_path": "../data/TinyStoriesV2-GPT4-valid.txt",
                    # 直接用 A100 上 chapter1/hw7 已编码好的数据（save_encode_ids.py 的产物），不依赖启动目录
                    "encoded_ids_train_path": str(REPO_ROOT / "chapter1" / "hw7" / "owt_encoded_ids_train.pkl"),
                    "encoded_ids_valid_path": str(REPO_ROOT / "chapter1" / "hw7" / "owt_encoded_ids_valid.pkl"),
                    "vocab_path": "vocab.json",
                    "merges_path": "merges.txt",

                    # Model
                    "vocab_size": 10000,
                    "context_length": 256,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "rope_theta": 10000.0,

                    # Training
                    "batch_size": batch_size, # Adjust based on your GPU memory
                    # "learning_rate": 3e-5,
                    #学习率退火相关参数
                    "initial_lr": 3e-5,
                    "max_learning_rate": 3e-5,
                    "min_learning_rate": 1e-5,
                    "lr_warmup_steps": 2000,
                    "cosine_cycle_iters": 10000,

                    #优化器相关参数
                    "weight_decay": 0.1,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "eps": 1e-8,

                    #梯度裁剪
                    "grad_clip": 1.0,

                    #训练相关参数
                    "epochs": epochs,
                    "train_steps": train_steps,
                    
                    # Logging & Checkpointing
                    "log_interval": 20,
                    "val_interval": 20,
                    # 验证集全量 22 万批要 9+ 小时（final_train.py 实测），只评估前 500 批
                    "max_val_batches": 500,
                    "checkpoint_interval": 60,
                    "checkpoint_dir": "checkpoints",
                }
                )
    config = run.config
    return config


def train(config,device,train_steps):
    # data_path = config["train_data_path"]
    vocab_size = config["vocab_size"]
    encoded_ids_train_path = config["encoded_ids_train_path"]
    encoded_ids_valid_path = config["encoded_ids_valid_path"]
    #直接导入编码后的数据
    with open(encoded_ids_train_path, "rb") as f:
        train_encode_ids = pickle.load(f)
    with open(encoded_ids_valid_path, "rb") as f:
        valid_encode_ids = pickle.load(f)
    # save_encode_ids.py 存的是 np.int32（省内存），CUDA 的 embedding 只收 int64 索引，转一下（final_train.py 同款坑）
    train_encode_ids = np.asarray(train_encode_ids, dtype=np.int64)
    valid_encode_ids = np.asarray(valid_encode_ids, dtype=np.int64)

    train_data_loader = DataLoader(train_encode_ids, config["batch_size"],config["context_length"],shuffle=True) # 训练集导入
    valid_data_loader = DataLoader(valid_encode_ids, config["batch_size"],config["context_length"],shuffle=True) # 验证集导入
    os.makedirs("checkpoints", exist_ok=True)  # torch.save 不会自动建目录，先建好
  
    # 加载模型
    model = TransformerModule(config["d_model"], config["n_heads"], config["d_ff"], config["context_length"], config["rope_theta"], config["n_layers"], vocab_size, device).to(device)
    # 加载优化器
    lr_scheduler = CosineSchedule(config["max_learning_rate"], config["min_learning_rate"], config["lr_warmup_steps"], config["cosine_cycle_iters"]) # 学习率退火   
    optimizer = AdamW(model.parameters(), config["initial_lr"], (config["adam_beta1"], config["adam_beta2"]), config["eps"], config["weight_decay"])
    # 损失直接用 F.cross_entropy（hw4 的 CrossEntropyloss 只支持二维输入，final_train.py 同款用法）
    # print("模型加载完成")
    # 训练循环
    model.train()
    global_step = 0  # 初始化全局步数

    for epoch in range(config["epochs"]):#每个epoch代表训练完了一整个所有批次的数据，不能轻易和step混淆
        if (epoch+1) >= 10:
            timer_func = timeit.default_timer
            start_time = timer_func()
        for step in range(train_steps):
            # 更新学习率
            new_lr = lr_scheduler(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            # print("epoch",epoch,"lr",new_lr)
            x,y = train_data_loader.get_train_batch_data()
            x = x.to(device)
            y = y.to(device)
            logits = model(x)  # shape: (batch, seq, vocab_size)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            # start_backward_time = timer_func()
            loss.backward() 
            # end_backward_time = timer_func()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()

            global_step += 1 # 增加全局步数

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} LR {new_lr:.6f} Loss: {loss.item()}")
        if (epoch+1) >= 10:
            end_time = timer_func()
        # print(f"Epoch {epoch} took {end_time - start_time:.2f} seconds")
            # 嵌套引号的 f-string 只在 Python 3.12+ 合法，拆出来写兼容所有版本
            time_fname = f"{config['d_model']}_{config['d_ff']}_{config['n_layers']}_{config['n_heads']}_train_time.txt"
            with open(time_fname, "a") as f:
                f.write(f"Epoch {epoch} took {end_time - start_time:.2f} seconds\n")
        # print("训练完了")
        wandb.log({"epoch": epoch, "loss": loss.item()})
        if (epoch+1) % config["val_interval"] == 0:
            model.eval()
            # 验证集全量 22 万批要 9+ 小时（final_train.py 实测），只评估前 max_val_batches 批
            max_val_batches = config.get("max_val_batches", 500)
            with torch.no_grad():
                for i, (x, y) in enumerate(valid_data_loader.get_valid_batch_data_iter()):
                    if i >= max_val_batches:
                        break
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                    wandb.log({"epoch": epoch, "loss": loss.item()})
            model.train()  # 验证完恢复训练模式（final_train.py 同款）
        # print("经过验证集")
        if (epoch+1) % config["checkpoint_interval"] == 0:
            # torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 可以加上'loss': loss.item()等
            }, f"checkpoints/model_epoch_{epoch}_{timestamp}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

    torch.save(model.state_dict(), f"checkpoints/model_final_{timestamp}.pth")
    print("Final checkpoint saved")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")  # A100 单卡是 cuda:0；作者原值 cuda:6 是 8 卡机器的
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--no-rmsnorm', dest='use_rmsnorm', action='store_false', help="Disable RMSNorm and use LayerNorm instead")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)


    parser.set_defaults(use_rmsnorm=True)
    args = parser.parse_args()

    device = args.device
    epochs = args.epochs
    train_steps = args.train_steps
    batch_size = args.batch_size
    d_model = args.d_model
    d_ff = args.d_ff
    n_layers = args.n_layers
    n_heads = args.n_heads

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    config = wandb_init(d_model, d_ff, n_layers, n_heads, batch_size, epochs, train_steps)

    train(config,device,train_steps)