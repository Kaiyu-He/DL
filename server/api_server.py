import socket
import math
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
import threading

sys.path.append(str(Path(__file__).parent.parent))
from eval.init import Init, BaseArgs
from eval.generate import generate, get_loss

app = Flask(__name__)
processing_lock = threading.Lock()


@app.route('/generate', methods=['POST'])
def process_generate():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    inputs = request.get_json()
    if not isinstance(inputs, dict):
        return jsonify({"error": "Input must be a dictionary"}), 400
    processing_lock.acquire()
    try:
        if "input" not in inputs:
            raise "请求不包含input输入文本"
        if "max_new_token" not in inputs:
            output = generate(model, tokenizer, inputs['input'])
        else:
            output = generate(model, tokenizer, inputs['input'], inputs['max_new_token'])
        response = {
            "response": output
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 确保释放锁
        processing_lock.release()


@app.route('/get_loss', methods=['POST'])
def process_get_loss():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    inputs = request.get_json()
    if not isinstance(inputs, dict):
        return jsonify({"error": "Input must be a dictionary"}), 400
    processing_lock.acquire()
    try:
        if "input" not in inputs:
            raise "请求不包含input输入文本"
        if "answer" not in inputs:
            raise "请求不包含answer输入文本"
        output = get_loss(model, tokenizer, inputs['input'], inputs['answer'])
        if type(output) == list:
            response = {
                "loss": output
            }
        else:
            response = {
                "loss": output.tolist(),
            }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 确保释放锁
        processing_lock.release()


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


class ModelArgs(BaseArgs):
    config_path: str = "/netcache/hekaiyu/project/config/qwen2.5-lora.yml"

    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7,8"
    args = ModelArgs().parse_args()
    model, tokenizer = Init(args)
    app.run(host=get_local_ip(), port=14425, threaded=True)
