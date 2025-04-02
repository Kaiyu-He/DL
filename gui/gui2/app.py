import socket
import requests
import logging
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./templates', static_folder='./templates')

prompt = f"""### 角色定位
你是一位专业的 AI 生成内容检测专家，擅长通过文本特征分析判断文章是否由 AI 撰写。具备自然语言处理、机器学习领域知识，熟悉各类 AI 写作模型的输出特征。

### 能力要求
1. 分析文本的语言模式、结构特征和内容连贯性
2. 识别常见 AI 生成文本的典型特征（如过度流畅、缺乏深度等）
3. 输出仅包含一个字”是“或”否“
4. 保持客观中立，避免主观臆断

### 知识储备
1. GPT 系列、Claude 等主流 AI 模型的输出特征
2. 人类写作与 AI 写作的核心差异点
3. 文本风格分析技术
4. 常见 AI 检测指标（如困惑度、突发性等）

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
        response.raise_for_status()  # 检查响应状态码
        return response.json()
    except requests.RequestException as e:
        logging.error(f"请求 API 时出错: {e}")
        return {"error": "请求 API 时出错，请稍后重试"}
    except ValueError as e:
        logging.error(f"解析响应 JSON 时出错: {e}")
        return {"error": "解析响应 JSON 时出错，请检查 API 配置"}


@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""
    summary = ""
    result = ""
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        if input_text:
            summary = "暂无"
            identify = generate(prompt.replace("<|content|>", input_text))['response']
            print(identify)
            if identify == '是':
                result = "该文本很可能为 AI 撰写"
            else:
                result = "该文本为人类撰写"
        else:
            summary = {"error": "请输入有效的文章内容"}
    return render_template('index.html', input_text=input_text, summary=summary, result=result)


@app.route('/page2')
def page2():
    return render_template('page2.html')


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
    app.run(host=get_local_ip(), port=10526, debug=True)