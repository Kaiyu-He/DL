import json
import os
import random
import sys

import tqdm

from data import get_data_detect_ai, get_eval_data_detect_ai

os.environ["CUDA_VISIBLE_DEVICES"] = "7,8"
import torch
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.init import Init, BaseArgs
from train.train_hf import generate_data,Dataset

def generate(model, tokenizer, input_text, max_new_tokens=512):
    if not isinstance(input_text, list):
        input_text = [input_text]
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs_ids = tokenizer(input_text, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        outputs_ids = model.generate(
            input_ids=inputs_ids.input_ids,
            attention_mask=inputs_ids.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    outputs_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_ids.input_ids, outputs_ids)
    ]
    outputs_text = []
    for output_ids in outputs_ids:
        outputs_text.append(tokenizer.decode(output_ids, skip_special_tokens=True))
    return outputs_text


def get_loss(model, tokenizer, prompts: list, answers: list = None, batch_size=1):
    inputs_ids = []
    labels = []
    max_length = 0
    if answers is None:
        for prompt in prompts:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)
            inputs_ids.append(prompt_ids["input_ids"])
            labels.append(prompt_ids["input_ids"])
            max_length = max(max_length, len(prompt_ids["input_ids"]))
    else:
        for prompt, answer in zip(prompts, answers):
            question = tokenizer(prompt, add_special_tokens=False)
            prompt_ids = tokenizer(answer, add_special_tokens=False)
            prompt_ids['input_ids'] = question['input_ids'] + prompt_ids['input_ids']
            inputs_ids.append(prompt_ids["input_ids"])
            labels.append([-100] * len(question["input_ids"]) + prompt_ids["input_ids"][len(question["input_ids"]):])
            max_length = max(max_length, len(prompt_ids["input_ids"]))
    for i in range(len(inputs_ids)):
        inputs_ids[i] = inputs_ids[i] + [tokenizer.pad_token_id] * (max_length - len(inputs_ids[i]))
        labels[i] = labels[i] + [-100] * (max_length - len(labels[i]))
    if answers is None:
        losses = []
    else:
        losses = torch.empty(0).to('cuda')
    inputs_ids = torch.tensor(inputs_ids).to('cuda')
    labels = torch.tensor(labels).to('cuda')
    num_batches = (len(inputs_ids) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        end_idx = min(end_idx, len(labels))
        batch_labels = labels[start_idx:end_idx]
        batch_input = inputs_ids[start_idx:end_idx]
        with torch.no_grad():
            output = model(
                input_ids=batch_input,
            )
            logits = output.logits[:, :-1, :]
            batch_labels = batch_labels[:, 1:].view(-1)
            batch_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch_labels,
                ignore_index=-100,
                reduction="none"
            )
            batch_loss = batch_loss.view((end_idx - start_idx), -1)
            if answers is None:
                for loss in batch_loss:
                    losses.append(loss.tolist())
                continue
            batch_loss = (batch_loss.sum(dim=1) / (batch_loss != 0).sum(dim=1))
            losses = torch.cat((losses, batch_loss), dim=0)
    return losses

def get_prompt():
    with open("/netcache/hekaiyu/project/data/prompt.json", "r")as f:
        return json.load(f)['prompt'][0]
class ModelArgs(BaseArgs):
    config_path: str = "/netcache/hekaiyu/project/config/qwen2.5-lora.yml"
    def __init__(self):
        super().__init__()

if __name__ == '__main__':

    args = ModelArgs().parse_args()
    model, tokenizer = Init(args)
    prompt_path = "/netcache/hekaiyu/project/data/detect_ai.json"
    prompt = get_prompt()
    dataset = get_eval_data_detect_ai(args.data, prompt_path)
    dataset.shuffle()
    for data in tqdm.tqdm(dataset, total=len(dataset)):
        input_text = data['input']
        print(data['output'])
        response = generate(model, tokenizer, [input_text], max_new_tokens=20)
        print(response)
        print(tokenizer("是", add_special_tokens=False))
        response = get_loss(model, tokenizer, [input_text], ["是"])
        print(tokenizer("否", add_special_tokens=False))
        print(response)
        response = get_loss(model, tokenizer, [input_text], ["否"])
        print(response)
        # response = get_loss(model, tokenizer, [input_text])
        # print(response)
        print("---------------")
