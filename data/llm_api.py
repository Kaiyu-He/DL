import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import tqdm

from data import get_data

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deepseek_query(query):
    from openai import OpenAI
    client = OpenAI(api_key="api-key", base_url="https://api.deepseek.com")
    message = [
        {
            "role": "user",
            "content": f"{query}"
        }
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=message,
        # response_format={
        #     'type': 'json_object'
        # }
    )
    text = response.choices[0].message.content
    return text


# def doubao_query(query):
#     from volcenginesdkarkruntime import Ark
#     client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3",
#                  api_key="")
#     completion = client.chat.completions.create(
#         model="ep-20250203192027-bt2bp",
#         messages=[
#             {"role": "user", "content": query},
#         ],
#     )
#     response = completion.choices[0].message.content
#     return response

def process(summary):
    start = time.time()
    query = f"""# 角色设定
你是一位专业的文章编辑助手，擅长根据提供的摘要快速生成文章详细内容。

# 要求
1. 严格基于提供的摘要生成内容
2. 可以适当地根据真实的内容补充背景、时间等信息
3. 撰写的新闻内容仅包含一行 400-600 字的文本， 不包含标题等其他多余信息
4. 确保信息准确性和语言专业性

# 摘要
{summary}

# 新闻内容

"""
    text = deepseek_query(query)
    with file_lock:
        with open(save_path, "a") as f:
            f.write(json.dumps({
                "summary": summary,
                "message": text
            }, ensure_ascii=False) + '\n')
    pbar.update(1)
    # print(time.time()-start)

if __name__ == "__main__":
    save_path = "./data/AI_content/deepseek/news.jsonl"
    dataset = get_data(['CLTS: 10000'])
    processed = {}
    restart = False
    try:
        if restart:
            raise "重新生成"
        with open(save_path, "r") as f:
            for line in f:
                line = json.loads(line)
                processed[line['summary']] = 1
    except Exception as e:
        print(e)
        with open(save_path, "w"):
            pass
    file_lock = threading.Lock()
    pbar = tqdm.tqdm(total=len(dataset))
    with ThreadPoolExecutor(max_workers=40) as executor:
        for data in dataset:
            if data['output'] in processed:
                pbar.update(1)
                continue
            future = executor.submit(process, data['output'])
