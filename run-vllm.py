import json
import os
import jieba
from rouge import Rouge
from vllm import SamplingParams, LLM
from vllm.lora.request import LoRARequest

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import tqdm
from transformers import AutoTokenizer
from data import get_eval_data
def generate(model, tokenizer, input):
    if len(lora_path):
        outputs = model.generate(
            [input],
            sampling_params,
            lora_request=LoRARequest("sql_adapter", 1, lora_path=lora_path)
        )
    else:
        outputs = model.generate(
            [input],
            sampling_params,
        )
    output_text = outputs[0].outputs[0].text
    return output_text

def metric(output, answer):
    rouge = Rouge()
    output = ' '.join(jieba.cut(output))
    answer = ' '.join(jieba.cut(answer))
    print(rouge.get_scores(output, answer))
    score = rouge.get_scores(output, answer)[0]['rouge-l']['f']
    return score

if __name__ == '__main__':
    dataset = get_eval_data(['CLTS: 500'], "/netcache/hekaiyu/project/prompt/summary.json")
    dataset.shuffle()
    # model_name = "Qwen2.5"
    # model_path = "/netcache/huggingface/Qwen2.5-7B-Instruct"
    # lora_path = "/netcache/hekaiyu/project/save/summary/qwen"
    # model_name = "llama3"
    # model_path = "/netcache/hekaiyu/model/llama-3-chinese-8b-instruct-v3"
    # lora_path = "/netcache/hekaiyu/project/save/summary/llama"
    model_name = "deepseek"
    model_path = "/netcache/hekaiyu/model/deepseek-llm-7b-chat"
    lora_path = "/netcache/hekaiyu/project/save/summary/deepseek"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remoteg_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    model = LLM(
        model=model_path,
        enable_lora=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9
    )
    with open(f"./output/{model_name}.jsonl", "w"):
        pass
    score = []
    pbar = tqdm.tqdm(total=len(dataset))
    with open(f"./output/{model_name}.jsonl", "a") as f:
        for data in dataset:
            input_text = data['input']
            output = generate(model, tokenizer, input_text)[0]
            print(output)
            output = output.split('分析：')[0]
            print(output)
            if not output:
                output = ""
            score.append(metric(output, data['output']))
            f.write(json.dumps({
                'output': output,
                'answer': data['output']
            }, ensure_ascii=False) + '\n')
            pbar.update(1)
            pbar.set_description_str(f"{sum(score) / len(score)}-{score[-1]}")

    print(f"final_score: {sum(score) / len(score)}")
