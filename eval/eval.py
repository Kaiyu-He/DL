import json
import requests
import tqdm
from data import get_eval_data_detect_ai

def generate(input, max_new_token=20) -> dict:
    try:
        response = requests.post(
            'http://210.75.240.138:10908/generate',
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
    return acc/len(dataset)

if __name__ == "__main__":
    with open("/netcache/hekaiyu/project/prompt/detect_ai.json", "r") as f:
        prompts = json.load(f)
    prompt = prompts['prompt'][0]
    metric(prompt)