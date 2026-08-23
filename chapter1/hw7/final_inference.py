import sys
from pathlib import Path

# 先把 hw1~hw5 加进 sys.path：本脚本顶层导入的模块（tokenizer_encode、RMSnorm、SwiGLU 等）
# 散落在不同 hw 目录里。用 __file__ 定位仓库根，不管在哪个目录启动脚本都能找到
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # chapter1/hw7/ 上三级才是仓库根
for hw in ["hw1", "hw2", "hw3", "hw4", "hw5"]:
    sys.path.insert(0, str(REPO_ROOT / "chapter1" / hw))

import argparse
import pickle
import torch

from transformermodule import TransformerModule          
from inference import decode_token                        
from tokenizer_encode import Tokenizer                    

# 定义模型结构（参数必须和训练时 final_train.py 的配置一字不差：768/3072/12/12）
# 注意：当前 vocab.pkl 只有 1000 个条目，所以这里是 1000；正式作业运行前重训 10000 的 BPE 后改成 10000
vocab_size = 1000
context_length = 256
d_model = 768
d_ff = 3072
n_layers = 12
n_heads = 12
rope_theta = 10000.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="用训练好的 Transformer LM 自回归生成文本")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="模型权重路径，训练结束后的文件名形如 checkpoints/model_final_20260823_xxxxxx.pth（时间戳由 final_train.py 自动生成）")
parser.add_argument("--prompt", type=str, default="baby shark",
                    help="给模型的输入文本。可选经典开头：'Once upon a time...'、狄更斯 'It was the best of times...'")
parser.add_argument("--max_tokens", type=int, default=200, help="最多生成的 token 数")
parser.add_argument("--temperature", type=float, default=1.0, help="温度缩放：越小越接近贪心解码")
parser.add_argument("--top_p", type=float, default=0.9, help="top-p 核采样阈值")
args = parser.parse_args()

model = TransformerModule(d_model, n_heads, d_ff, context_length, rope_theta, n_layers, vocab_size, device).to(device)

# 加载权重，自动兼容两种格式：
# 最终模型      -> final_train.py 里 torch.save(model.state_dict())，加载出来直接就是 state_dict
# 中途 checkpoint -> {'epoch':..., 'model_state_dict':..., 'optimizer_state_dict':...} 字典，要取 ['model_state_dict']
ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)

# 加载分词器（和训练用的是同一份）
with open(REPO_ROOT / "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open(REPO_ROOT / "merges.pkl", "rb") as f:
    merges = pickle.load(f)

special_tokens = ["<|endoftext|>"]

tokenizer = Tokenizer(vocab, merges, special_tokens)
input_ids = tokenizer.encode(args.prompt)

# 推理：传 Python list 而不是张量——decode_token 内部会自己包张量并放到模型所在设备
output_ids = decode_token(input_ids, model, tokenizer,
                          max_tokens_to_generate=args.max_tokens,
                          top_p=args.top_p, temperature=args.temperature)

output_text = tokenizer.decode(output_ids[0].cpu().tolist())
print(output_text)
