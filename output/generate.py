import json
import os
import random
import sys

import jieba
from rouge import Rouge

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_eval_data
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

def get_prompt():
    with open("/netcache/hekaiyu/project/prompt/summary.json", "r")as f:
        return json.load(f)['prompt'][0]

def metric(output, answer):
    rouge = Rouge()
    output = ' '.join(jieba.cut(output))
    answer = ' '.join(jieba.cut(answer))
    score = rouge.get_scores(output, answer)[0]['rouge-l']['f']
    return score

if __name__ == '__main__':
    dataset = get_eval_data(['CLTS: 500'], "/netcache/hekaiyu/project/prompt/summary.json")
    # model_name = "Qwen2.5"
    # model_path = "/netcache/huggingface/Qwen2.5-7B-Instruct"
    # lora_path = "/netcache/hekaiyu/project/save/summary/qwen"
    # model_name = "llama3"
    # model_path = "/netcache/hekaiyu/model/llama-3-chinese-8b-instruct-v3"
    # lora_path = "/netcache/hekaiyu/project/save/summary/llama"
    model_name = "deepseek"
    model_path = "/netcache/hekaiyu/model/deepseek-llm-7b-chat"
    lora_path = "/netcache/hekaiyu/project/save/summary/deepseek"
    print(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if len(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num = 0
    with open(f"./output/{model_name}.jsonl", "r")as f:
        for _ in f:
            num += 1
    score = []
    pbar = tqdm.tqdm(total=len(dataset))
    with open(f"./output/{model_name}.jsonl", "a")as f:
        for i, data in enumerate(dataset):
            if i < num:
                pbar.update(1)
                continue
            input_text = data['input']
            output = generate(model, tokenizer, input_text, max_new_tokens=128)[0]
            print(output)
            output = output.split('分析：')[0]
            try:
                score.append(metric(output, data['output']))
            except:
                score.append(0)
            f.write(json.dumps({
                'output': output,
                'answer': data['output']
            }, ensure_ascii=False) + '\n')
            pbar.update(1)
            pbar.set_description_str(f"{sum(score)/len(score)}-{score[-1]}")

    print(f"final_score: {sum(score)/len(score)}")
