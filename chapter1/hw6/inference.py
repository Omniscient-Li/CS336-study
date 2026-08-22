import torch

def top_p_sampling(probabilities , top_p = 0.9):
    """
    top_p 核采样 
    1.对所有token按原始概率降序排序
    2.选择前面的token，直到累计概率 >= p（丢弃"之前累计概率已 >= p"的 token）
    3.丢弃剩余token，重新归一化所选集合的gailv
    4.从中随机选取下一个token
    """
    sorted_probabilities , idx = torch.sort(probabilities , dim = -1 , descending = True) #按概率降序排序
    cumulative_probabilities = torch.cumsum(sorted_probabilities , dim = -1) #累积概率
    # 只丢掉"它之前的累计概率已经 > p"的 token：恰好把累计概率推过 p 的那个 token 会被保留，
    # 且永远至少保留概率最大的 1 个 token。旧写法 mask = cumsum > p 会把跨过 p 的 token 一起丢掉，
    # 若 top-1 自己就超过 p，则全部被清零 → 除零得 NaN → multinomial 报错
    mask = (cumulative_probabilities - sorted_probabilities) > top_p
    sorted_probabilities[mask] = 0 #被丢弃的 token 概率置 0

    #归一化
    sorted_probabilities.div_(sorted_probabilities.sum(dim = -1 , keepdim = True))

    #随机选择一个概率大于0的token
    next_token_idx = torch.multinomial(sorted_probabilities , 1)
    next_token_idx = torch.gather(idx , dim = -1 , index = next_token_idx)
    #返回下一个token
    return next_token_idx

def temperature_scaling(logits , temperature = 1.0):
    """
    温度缩放
    1.将Logits除以温度参数
    2.温度参数越大，概率分布越平滑，输出越随机多样；越小越接近贪心解码
    """
    probabilities = torch.softmax(logits[:,-1,:]/temperature,dim=-1)
    return probabilities

def decode_token(input_tokens , model , tokenizer , max_tokens_to_generate , top_p = 0.9 , temperature = 1.0):
    """
    自回归解码：给模型一个 prompt，逐个生成后续 token，直到生成满 max_tokens_to_generate 个
    或提前生成出 <|endoftext|>（EOS）。
    1.将输入token传入模型，获取logits（只取最后一个位置的 logits 预测下一个词）
    2.对Logits进行温度缩放，得到概率分布
    3.对概率分布进行top_p采样，得到下一个token
    4.将下一个token拼回输入序列，重复1-3

    input_tokens : token id 列表（如 tokenizer.encode(文本) 的结果），返回 (1, 序列长度) 的 id 张量，
                  可以 tokenizer.decode(输出[0].tolist()) 还原成文本
    tokenizer    : hw2 的 Tokenizer，用来查 <|endoftext|> 的 id
    """
    model.eval() #设置为评估模式不要dropout

    input_tokens = torch.tensor(input_tokens).unsqueeze(0)  # 加 batch 维：(seq,) → (1, seq)
    # 查 EOS 的 id：和 hw2 encode 里特殊 token 的查法一致（词表的键是 utf-8 字节）。
    # 用 .get 是防止词表里没训练进这个特殊 token 时直接 KeyError
    eos_token_id = tokenizer.bytes_to_id.get("<|endoftext|>".encode("utf-8"))
    with torch.no_grad(): #不计算梯度
        for _ in range(max_tokens_to_generate):
            logits = model(input_tokens)
            probabilities = temperature_scaling(logits,temperature)
            next_token_idx = top_p_sampling(probabilities,top_p)
            # 生成出 EOS 就提前结束；EOS 不拼进结果，否则 decode 时会把 "<|endoftext|>" 原样印出来
            # （旧代码拿整个张量和字符串比较，永远返回 False，等于没有这个判断）
            if eos_token_id is not None and next_token_idx.item() == eos_token_id:
                break
            input_tokens = torch.cat([input_tokens,next_token_idx],dim=-1) # 将下一个token添加到input_ids中

    return input_tokens
