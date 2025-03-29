import socket

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

prompt = f"""### 角色定位
你是一位专业的AI生成内容检测专家，擅长通过文本特征分析判断文章是否由AI撰写。具备自然语言处理、机器学习领域知识，熟悉各类AI写作模型的输出特征。

### 能力要求
1. 分析文本的语言模式、结构特征和内容连贯性
2. 识别常见AI生成文本的典型特征（如过度流畅、缺乏深度等）
3. 输出仅包含一个字”是“或”否“
4. 保持客观中立，避免主观臆断

### 知识储备
1. GPT系列、Claude等主流AI模型的输出特征
2. 人类写作与AI写作的核心差异点
3. 文本风格分析技术
4. 常见AI检测指标（如困惑度、突发性等）

### 文章内容
<|content|>

### 回答
答案：
"""


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


@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""
    summary = ""
    result = ""
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        result = input_text
        summary = generate(prompt.replace("<|content|>", input_text))
    return render_template('index.html', input_text=input_text, summary=summary, result=result)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


if __name__ == '__main__':
    app.run(host=get_local_ip(), port=10425, debug=True)
