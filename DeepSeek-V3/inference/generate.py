import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

def sample(logits, temperature=1.0):
    """
    从给定的 logits 中采样一个索引，基于温度参数调整概率分布。

    参数:
    logits (Tensor): 未归一化的预测分数，用于计算概率分布。
    temperature (float): 温度参数，用于控制概率分布的随机性。默认值为1.0。

    返回:
    Tensor: 采样得到的索引，表示最可能的类别。
    """
    # 根据温度参数调整logits，以改变softmax的概率分布
    logits = logits / max(temperature, 1e-5)

    # 计算调整后的logits的softmax，得到概率分布
    probs = torch.softmax(logits, dim=-1)

    # 通过除以一个指数分布的随机数，然后取argmax来采样索引
    # 这种方法可以引入额外的随机性，避免总是选择概率最大的类别
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
        model: Transformer,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id:int,
        temperature=1.0) -> List[List[int]]:
    """
    使用Transformer模型生成文本。

    参数:
    - model: Transformer模型实例。
    - prompt_tokens: 提示词的token序列，用于模型生成的起始。
    - max_new_tokens: 最大生成token数量。
    - eos_id: 结束标记的token ID，生成到此token时停止。
    - temperature: 用于控制生成随机性的温度值。高温时生成更随机，低温时生成更确定。

    返回:
    - 生成的token序列。
    """
    # 计算每个提示词的长度，并确保最长的提示词长度不超过模型的最大序列长度。
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    # 计算总的序列长度，包括提示词和新生成的token，但不超过模型的最大序列长度。
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    # 初始化tokens张量，用于存储提示词和生成的序列，初始值为-1。
    tokens = torch.full((len(prompt_tokens), total_len), -1 ,dtype=torch.long, device="cuda")
    # 将提示词的token填充到tokens张量中。
    for i , t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    # 初始化prev_pos为0，表示当前处理的位置。
    prev_pos = 0
    # 初始化finished标志，表示每个序列是否已完成生成。
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    # 创建prompt_mask，标记tokens中哪些位置是已知的提示词。
    prompt_mask = tokens != -1
    # 逐个位置生成新的token，直到达到最大长度或所有序列都完成生成。
    for cur_pos in range(min(prompt_lens), total_len):
        # 使用模型预测下一个token的概率分布。
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        # 根据温度值选择下一个token，温度大于0时进行采样，否则选择概率最高的token。
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # 根据prompt_mask决定是否使用生成的token还是保留原始token。
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        # 更新finished标志，如果当前位置是新生成的且为结束标记，则标记为完成。
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        # 如果所有序列都已完成生成，则停止循环。
        if finished.all():
            break
    # 提取生成的token序列，去除提示词和结束标记后的部分。
    comletion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        comletion_tokens.append(toks)
    return comletion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained("ckpt_path")
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_special_tokens=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size
        prompts_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                        add_generation_prompt=True) for prompt in prompts]
        completions_tokens = generate(model, prompts_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.backend_decode(completions_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)