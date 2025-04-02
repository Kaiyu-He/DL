import json
import os.path
import random
import sys
from pathlib import Path

import requests
import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from data import get_eval_data_detect_ai

def deepseek_query(query):
    from openai import OpenAI
    client = OpenAI(api_key="sk-df3548002777404da2b96456cb349046", base_url="https://api.deepseek.com")
    message = [
        {
            "role": "user",
            "content": f"{query}"
        }
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=message,
        temperature=1,
        response_format={
            'type': 'json_object'
        }
    )
    text = response.choices[0].message.content
    return text

def generate(input, max_new_token=20) -> dict:
    try:
        response = requests.post(
            'http://210.75.240.138:14425/generate',
            json={
                "input": input,
                "max_new_token": max_new_token,
            },
        )
        return response.json()
    except Exception as e:
        print(e)


def metric(prompt):
    acc, null, tot = 0, 0, 0

    dataset = get_eval_data_detect_ai(["CLTS: 500"], prompt)
    pbar = tqdm.tqdm(total=len(dataset))
    for data in dataset:
        text = data
        output = generate(text['input'])['response'][0]
        if output.find(text['output']) >= 0:
            acc += 1
        if not output.find('是') >= 0 and not output.find('否') >= 0:
            null += 1
        tot += 1
        pbar.update(1)
        pbar.set_description_str(f"CLTS: acc: {acc}/{tot}/{null}-{acc / tot}")
    acc1, null1, tot1 = 0, 0, 0
    dataset = get_eval_data_detect_ai(["deepseek: 500"], prompt)
    pbar = tqdm.tqdm(total=len(dataset))
    for data in dataset:
        text = data
        output = generate(text['input'])['response'][0]
        if output.find(text['output']) >= 0:
            acc1 += 1
        if not output.find('是') >= 0 and not output.find('否') >= 0:
            null1 += 1
        tot1 += 1
        pbar.update(1)
        pbar.set_description_str(f"deepseek: acc: {acc1}/{tot1}/{null1}-{acc1 / tot1}")
    acc += acc1
    null += null1
    tot += tot1
    return acc/tot


def _update(prompt):
#     query = f"""# 角色设定
# 你是一位专业的「提示词优化专家」，专注于帮助用户改进和优化他们的AI提示词。你精通各类大模型的工作原理，擅长通过精准的语言调整来提升生成内容的质量、相关性和创造性。
#
# # 能力
# - 分析用户提供的原始提示词，找出模糊、歧义或低效的部分
# - 提供具体的优化建议和修改方案
# - 解释每个优化点的理由和预期效果
# - 能对提示词缺失的描述进行一定的补充
# - 回答格式为 {{"prompt": ..., "analyse": ...}}
#
# # 任务
# - 优化和完善提示词，要求模型能够能好的完成文本摘要任务
# - 文本的内容用 "<|content|>" 的形式插入
# - 要求提示词中必须包含 "<|content|>"
#
# # 用户提示词
# {prompt}
#
# # 优化后提示词
# """
    query = f"""# 角色设定
你是一位专业的「提示词优化专家」，专注于帮助用户改进和优化他们的AI提示词。你精通各类大模型的工作原理，擅长通过精准的语言调整来提升生成内容的质量、相关性和创造性。

# 能力
- 分析用户提供的原始提示词，找出模糊、歧义或低效的部分
- 提供具体的优化建议和修改方案
- 解释每个优化点的理由和预期效果
- 能对提示词缺失的描述进行一定的补充
- 请以 JSON 格式返回以下信息： {{"prompt": ..., "analyse": ...}}

# 任务
- 优化和完善提示词，使模型能够能好的完成AI撰写文本检测任务
- 文本的内容用 "<|content|>" 的形式插入
- 要求提示词中必须包含 "<|content|>"
- 要求优化后的提示词与原有的提示词有明显的变动

# 提示词
```
{prompt}```
# 回答
"""
    # prompt = generate([query], 512)['response'][0]
    # print(prompt)
    # prompt = prompt.split("{", 1)[1].split("}", 1)[0]
    # prompt = json.loads(f"{{{prompt}}}")['prompt']
    # print(prompt)
    prompt = deepseek_query(query)
    prompt = json.loads(prompt)['prompt']
    # print(prompt)
    return prompt


def _write(prompt: str, score, prompts, save_dir):
    if score <= prompts['prompt'][-1][1]:
        return prompts
    if len(prompts['prompt']) < 10:
        prompts['prompt'].append((prompt, score))
    else:
        prompts['prompt'][-1] = (prompt, score)
    prompts['prompt'].sort(key=lambda k: -k[1])
    with open(save_dir, "w") as f:
        f.write(json.dumps(prompts, ensure_ascii=False, indent=4) + '\n')


def updata_prompt(save_dir, original_path):
    with open(original_path, "r") as f:
        prompts = json.load(f)
    # orig = prompts['prompt'][0]
    # prompts = {
    #     "prompt": [(orig, metric(orig))]
    # }
    times = 0
    while times <= 1000:
        times += 1
        now = _update(prompts['prompt'][random.randint(0,len(prompts['prompt'])-1)][0])
        print(now)
        score = metric(now)
        _write(now, score, prompts, save_dir)


if __name__ == '__main__':
    updata_prompt("/netcache/hekaiyu/project/prompt/detect_ai1.json",
                  "/netcache/hekaiyu/project/prompt/detect_ai2.json")
