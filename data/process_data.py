import json
import os.path
import random
from pathlib import Path

import numpy as np
import torch


class CLTSDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="./data/CLTS/train"):
        if not os.path.exists(data_path):
            raise f"{data_path} 数据集文件不存在"
        self.data_path = data_path
        self.data = {
            "input": [],
            "output": []
        }
        src_path = next(Path(data_path).glob("*.src"))
        with open(src_path) as f:
            for line in f:
                line = line.strip()
                line = line.replace(" ", "")
                line = line[:line.rfind('。') + 1]
                self.data['input'].append(line)
        tgt_path = next(Path(data_path).glob("*.tgt"))
        with open(tgt_path) as f:
            self.data['output'] = [line.strip().replace(" ", "") for line in f]

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        return {
            "input": self.data['input'][idx],
            "output": self.data["output"][idx],
        }


class deepseek_news_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path="./data/AI_content/deepseek/train"):
        if not os.path.exists(data_path):
            raise f"{data_path} 数据集文件不存在"
        self.data_path = data_path
        self.data = {
            "input": [],
        }
        path = next(Path(data_path).glob("*.jsonl"))
        with open(path, "r") as f:
            for line in f:
                line = json.loads(line)
                self.data['input'].append(line['message'])

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        return {
            "input": self.data['input'][idx],
            "output": "",
        }


class TotalDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_path):
        self.data = {
            "input": [],
            "output": []
        }
        try:
            with open(prompt_path, "r") as f:
                self.prompt = json.load(f)['prompt']
        except:
            self.prompt = ["<|content|>"]
        self.indices = []

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        idx = self.indices[idx]
        input_ids = self.prompt[random.randint(0, len(self.prompt) - 1)]
        input_ids = input_ids.replace("<|content|>", self.data['input'][idx])
        return {
            "input": input_ids,
            "output": self.data["output"][idx],
        }

    def add(self, d):
        self.data['input'].append(d['input'])
        self.data['output'].append(d['output'])
        self.indices.append(len(self.data["input"]))
    def shuffle(self):
        random.shuffle(self.indices)

def get_data(args):
    dataset = TotalDataset("./prompt.json")
    for i in args:
        try:
            name, total = i.split(':', 1)
            total = int(total)
        except:
            name = i
            total = -1
        if name.find("CLTS") >= 0:
            sub_dataset = CLTSDataset()
            num = 0
            for d in sub_dataset:
                if num == total:
                    break
                num += 1
                dataset.add(d)
    return dataset


def get_data_detect_ai(args, prompt_path="./detect_ai.json"):
    dataset = TotalDataset(prompt_path)
    for i in args:
        try:
            name, total = i.split(':', 1)
            total = int(total)
        except:
            name = i
            total = -1
        if name.find("CLTS") >= 0:
            sub_dataset = CLTSDataset()
            num = 0
            for d in sub_dataset:
                if num == total:
                    break
                num += 1
                d['output'] = '否'
                dataset.add(d)
        elif name.find("deepseek") >= 0:
            sub_dataset = deepseek_news_dataset()
            num = 0
            for d in sub_dataset:
                if num == total:
                    break
                num += 1
                d['output'] = '是'
                dataset.add(d)
    return dataset
def get_eval_data_detect_ai(args, prompt_path="./detect_ai.json"):
    dataset = TotalDataset(prompt_path)
    for i in args:
        try:
            name, total = i.split(':', 1)
            total = int(total)
        except:
            name = i
            total = -1
        if name.find("CLTS") >= 0:
            sub_dataset = CLTSDataset("./data/CLTS/test")
            num = 0
            for d in sub_dataset:
                if num == total:
                    break
                num += 1
                d['output'] = '否'
                dataset.add(d)
        elif name.find("deepseek") >= 0:
            sub_dataset = deepseek_news_dataset("./data/AI_content/deepseek/test")
            num = 0
            for d in sub_dataset:
                if num == total:
                    break
                num += 1
                d['output'] = '是'
                dataset.add(d)
    return dataset
