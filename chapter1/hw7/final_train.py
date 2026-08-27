import sys
from pathlib import Path

# 先把 hw1~hw5 加进 sys.path：tokenizer_encode、adamw、dataloader 等顶层导入的模块散落在
# 不同 hw 目录里。用 __file__ 定位仓库根，不管在哪个目录启动脚本都能找到（和 final_inference.py 同款）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # chapter1/hw7/ 上三级才是仓库根
for hw in ["hw1", "hw2", "hw3", "hw4", "hw5"]:
    sys.path.insert(0, str(REPO_ROOT / "chapter1" / hw))

import os
import wandb
import time
import torch
import json
import pickle
import argparse
import numpy as np
import torch.nn.functional as F
from tokenizer_encode import Tokenizer
from pair_all_bpe_tokenzier import run_train_bpe

from adamw import AdamW
from dataloader import DataLoader
from transformermodule import TransformerModule
from transformermodule_withoutrmsnorm import TransformerModuleWithoutRMSNorm
from lr_cosine_shedule import CosineSchedule

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--train_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument('--no-rmsnorm', dest='use_rmsnorm', action='store_false', help="Disable RMSNorm and use LayerNorm instead")
# parser.add_argument("--no_rope", dest='use_rope', action='store_false', help="Disable RoPE and use learned position embeddings instead")
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


timestamp = time.strftime("%Y%m%d_%H%M%S")
wandb.login()
run = wandb.init(project="cs336_final_train", 
                 config = {
                # Experiment
                "experiment_name": f"tinystories_17M_{timestamp}",
                "total_tokens_processed": 327_680_000,
                
                # Data
                "train_data_path": "../data/TinyStoriesV2-GPT4-train.txt",
                "valid_data_path": "../data/TinyStoriesV2-GPT4-valid.txt",
                "vocab_path": "vocab.json",
                "merges_path": "merges.txt",

                # Model
                # 注意：当前 vocab.pkl 只有 1000 个条目（hw1 训练 BPE 时用的 vocab_size=1000），
                # 这里必须和词表一致，否则模型 90% 的输出是词表里不存在的 id。
                # TODO: 正式作业运行前用 hw1 重新训练 vocab_size=10000 的 BPE，再改回 10000 并重新编码数据
                "vocab_size": 1000,
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
                # 验证集 7.1M token ≈ 22 万个 batch，全量评估要 9+ 小时（实测踩坑）；
                # 只评估前 max_val_batches 批（每批 8192 token，500 批 = 4M token ≈ 2 分钟，足够有代表性）
                "max_val_batches": 500,
                "checkpoint_interval": 60,
                "checkpoint_dir": "checkpoints",
            }
            )
config = run.config

# 最终 checkpoint 要写进这个目录；不提前建好的话最后 torch.save 会 FileNotFoundError
checkpoint_dir = REPO_ROOT / "chapter1" / "hw7" / config["checkpoint_dir"]
os.makedirs(checkpoint_dir, exist_ok=True)




device = torch.device(device if torch.cuda.is_available() else "cpu")
data_path = config["train_data_path"]
vocab_size = config["vocab_size"]
# # 训练BPE分词器
# special_tokens = ["<|endoftext|>"]
# vocab, merges = run_train_bpe(data_path, vocab_size, special_tokens)
# print("已经训练好BPE分词器")

# 从 vocab.pkl 加载词汇表
# with open("vocab.pkl", "rb") as f:
#     # pickle.load 会自动恢复字典，并且值是 bytes 类型
#     vocab = pickle.load(f)

# 从 merges.pkl 加载合并规则
# with open("merges.pkl", "rb") as f:
#     # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
#     merges = pickle.load(f)
# special_tokens = ["<|endoftext|>"]  

# tokenizer = Tokenizer(vocab, merges, special_tokens)
# 加载训练数据
# with open(data_path, "r",encoding="utf-8") as f:
#     original_data = f.read()
# encode_ids = tokenizer.encode(original_data)
# encode_ids = torch.tensor(encode_ids, dtype=torch.long)
# print("数据加载完成")
#直接导入编码后的数据（绝对路径，不依赖启动目录；save_encode_ids.py 也写到同样位置）
with open(REPO_ROOT / "chapter1" / "hw7" / "owt_encoded_ids_train.pkl", "rb") as f:
    train_encode_ids = pickle.load(f)
with open(REPO_ROOT / "chapter1" / "hw7" / "owt_encoded_ids_valid.pkl", "rb") as f:
    valid_encode_ids = pickle.load(f)
# save_encode_ids.py 存的是 np.int32（省内存），但 CUDA 的 embedding/cross_entropy 只收 int64 索引，
# 转成 int64 再喂给 DataLoader（CPU 上 int32 不报错，这个坑只在 GPU 上暴露）
train_encode_ids = train_encode_ids.astype(np.int64)
valid_encode_ids = valid_encode_ids.astype(np.int64)

train_data_loader = DataLoader(train_encode_ids, config["batch_size"],config["context_length"],shuffle=True) # 训练集导入
valid_data_loader = DataLoader(valid_encode_ids, config["batch_size"],config["context_length"],shuffle=True) # 验证集导入
# 加载模型
# use_rmsnorm=True 时用带 RMSNorm 的 TransformerModule；--no-rmsnorm 时才用 LayerNorm 版本（之前反了）
if args.use_rmsnorm:
    model = TransformerModule(config["d_model"], config["n_heads"], config["d_ff"], config["context_length"], config["rope_theta"], config["n_layers"], vocab_size, device).to(device)
else:
    model = TransformerModuleWithoutRMSNorm(config["d_model"], config["n_heads"], config["d_ff"], config["context_length"], config["rope_theta"], config["n_layers"], vocab_size, device).to(device)
# 加载优化器
lr_scheduler = CosineSchedule(config["max_learning_rate"], config["min_learning_rate"], config["lr_warmup_steps"], config["cosine_cycle_iters"]) # 学习率退火   
optimizer = AdamW(model.parameters(), config["initial_lr"], (config["adam_beta1"], config["adam_beta2"]), config["eps"], config["weight_decay"])
# 损失直接用 PyTorch 自带的交叉熵：hw4 的 CrossEntropyloss 只支持二维 [batch, vocab]，
# 这里的 logits 是 (batch, seq, vocab)，要先 view(-1, vocab_size) 展平再算
print("模型加载完成")
# 4. 训练循环
model.train()
global_step = 0  # 初始化全局步数
for epoch in range(config["epochs"]):
    # for step in range(len(train_data_loader)):
    # 学习率更新移到 step 循环内部
    # print("epoch",epoch)
    for step in range(args.train_steps):
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])  # config 里配了 1.0，之前没执行
        optimizer.step()
        global_step += 1 # 增加全局步数

        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step} LR {new_lr:.6f} Loss: {loss.item()}")
    # print("训练完了")
    wandb.log({"epoch": epoch, "train_loss": loss.item()})
    # print("经过了wandblog")
    if (epoch+1) % config["val_interval"] == 0:
        model.eval()
        val_losses = []  # 收集所有 batch 的 loss，最后取平均 log 一次（之前每个 batch 都 log 且都叫 "loss"）
        max_val_batches = config.get("max_val_batches", 500)
        print(f"Epoch {epoch}: 开始验证集评估（只取前 {max_val_batches} 批；全量 22 万批要 9+ 小时，别等）")
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_data_loader.get_valid_batch_data_iter()):
                if i >= max_val_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch} val_loss: {val_loss:.4f}（{len(val_losses)} 批的平均）")
        wandb.log({"epoch": epoch, "val_loss": val_loss})
        model.train()  # 验证完切回训练模式（之前忘了，后面 epoch 会在 eval 模式下训练）
    # print("经过验证集")
    if (epoch+1) % config["checkpoint_interval"] == 0:
        # torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 可以加上'loss': loss.item()等
        }, checkpoint_dir / f"model_epoch_{epoch}_{timestamp}.pth")
        print(f"Checkpoint saved at epoch {epoch}")

torch.save(model.state_dict(), checkpoint_dir / f"model_final_{timestamp}.pth")
print("Final checkpoint saved")