import argparse
import json
import os
import time

import torch
import numpy as np
import tqdm
from datasets import load_dataset
import wandb
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelArgs(BaseArgs):
    config_path: str = "/netcache/hekaiyu/project/config/qwen2.5-lora.yml"
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    args = ModelArgs().parse_args()
    lora_path = args.lora_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remoteg_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    llm = LLM(
        model=model_path,
        enable_lora=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8
    )

    acc, err, num, step = 0, 0, 0, 0
    eval = np.array([[0, 0], [0, 0]])
    result = {}

    true_sample = []
    wrong_sample = []
    # with open("/home/hekaiyu/sft_ds_lora/true.jsonl","w"):
    #     pass
    # with open("/home/hekaiyu/sft_ds_lora/wrong.jsonl","w"):
    #     pass
    wandb.init(name=args.name)
    start = time.time()
    for item in tqdm.tqdm(dataset):
        # print("!!!!"+data)
        print(item["type"])

        print()
        if len(lora_path):
            outputs = llm.generate(
                [item["inputs_text"]],
                sampling_params,
                lora_request=LoRARequest("sql_adapter", 1, lora_path=lora_path) if len(lora_path) > 0 else None
            )
        else:
            outputs = llm.generate(
                [item["inputs_text"]],
                sampling_params,
            )
        for output in outputs:
            print(output)
            prompt = output.prompt
            output_text = output.outputs[0].text
            if output_text.find("Yes") >= 0:
                output_text = 1
            else:
                if output_text.find("No") >= 0:
                    output_text = 0
                else:
                    output_text = -1
            num += 1
            print(output_text)
            if output_text == item["label"]:
                true_sample.append(item["dataset"])
                if item['type'] in result:
                    result[f"{item['type']}"]["acc"] += 1
                else:
                    result[f"{item['type']}"] = {
                        "acc": 1,
                        "wrong": 0,
                        "error": 0,
                    }

                eval[int(output_text)][int(item["label"])] += 1
            elif output_text != -1:
                wrong_sample.append(item["dataset"])
                if item['type'] in result:
                    result[f"{item['type']}"]["wrong"] += 1
                else:
                    result[f"{item['type']}"] = {
                        "acc": 0,
                        "wrong": 1,
                        "error": 0,
                    }
                eval[int(output_text)][int(item["label"])] += 1
            else:
                eval[0][int(item["label"])] += 1
                err += 1
                if item['type'] in result:
                    result[f"{item['type']}"]["error"] += 1
                else:
                    result[f"{item['type']}"] = {
                        "acc": 0,
                        "wrong": 0,
                        "error": 1,
                    }
            step += 1
            epsilon = 1e-7
            if step % 1 == 0:
                # print(step)
                result["accuracy"] = eval
                result["F1"] = 2 * eval[1][1] / (2 * eval[1][1] + eval[0][1] + eval[1][0] + epsilon)
                result["acc"] = (eval[1][1] + eval[0][0]) / (eval[0][0] + eval[0][1] + eval[1][0] + eval[1][1])
                result["error_num"] = err
                print(result)
                error = []

    # write_to_file(true_sample, "/home/hekaiyu/sft_ds_lora/true.jsonl")
    # write_to_file(wrong_sample, "/home/hekaiyu/sft_ds_lora/wrong.jsonl")
    end = time.time()
    wandb.log({"acc": result["acc"],
               "F1": result["F1"],
               "err": result["error_num"],
               "select_pos_true": eval[1][1] / (eval[1][1] + eval[1][0]),
               "select_neg_true": eval[0][0] / (eval[0][0] + eval[0][1]),
               "FP": eval[1][0],
               "FN": eval[0][1],
               "TP": eval[1][1],
               "TN": eval[0][0],
               "run_time": end - start
               })
    print(len(result))
    for key, value in result.items():
        if type(value) == dict:
            wandb.log({f"{key}": value["acc"] / (value["acc"] + value["wrong"])})
    wandb.finish()
