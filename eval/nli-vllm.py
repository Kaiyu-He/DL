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




def write_to_file(datasets, file_path):
    with open(file_path, 'a') as f:
        for item in datasets:
            f.write(json.dumps(item))
            f.write('\n')


def get_dataset(data_path):
    dataset = load_dataset("json", data_files=data_path)
    prompts = []
    with open("/home/hekaiyu/sft_ds_lora/prompt.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            if "instruction" in item:
                prompts.append(item["instruction"])

    def text_prepocess_function(example):
        sentence1 = example["sentence1"].replace("\n", " ")
        sentence2 = example["sentence2"].replace("\n", " ")
        num = 1
        template = (
            f"{prompts[num]}\n\n"
            f"## Claim: {sentence2}\n\n"
            f"## References: {sentence1}\n\n"
            f"## Output:"
        )
        messages = [
            {"role": "user", "content": template},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if "label_manual" in example:
            label = example["label_manual"]
        else:
            label = example["label"]
        return {
            "dataset": example,
            "inputs_text": input_text,
            "label": int(label),
            "type": example["type"] if "type" in example else ""
        }

    dataset = dataset.map(text_prepocess_function, remove_columns=dataset["train"].column_names)
    return dataset["train"]


os.environ["WANDB_API_KEY"] = "666bff72d04cfb09fb241f5152a186617627651b"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/netdisk/hekaiyu/model/Qwen2-1.5B-Instruct",
        help=""
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/netcache/hekaiyu/result/Qwen2-1.5B-deepseek/checkpoint-9000",
        help=""
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Qwen2-1.5B-deepseek",
        help=""
    )
    parser.add_argument(
        "--data_path",
        type=str,
        # default="/netcache/hekaiyu/dataset-nli/eval-manual.jsonl",
        default="/netcache/hekaiyu/dataset-nli/train-deepseek.jsonl",
        help=""
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        # default="/netcache/hekaiyu/dataset-nli/eval-manual.jsonl",
        default="nli",
        help=""
    )
    args = parser.parse_known_args()[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["WANDB_PROJECT"] = args.wandb_project
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remoteg_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=3072)

    llm = LLM(
        model=model_path,
        enable_lora=True,
        tensor_parallel_size=4 if args.name.find("14B") >= 0 else 2,
        gpu_memory_utilization=0.8
    )
    data_path = args.data_path
    lora_path = args.lora_path

    dataset = get_dataset(data_path)
    print(dataset)
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
