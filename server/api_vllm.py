import socket
import os

from transformers import AutoTokenizer
from vllm import SamplingParams, LLM
from vllm.lora.request import LoRARequest

os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
import sys
from pathlib import Path
from flask import Flask, request, jsonify
import threading

sys.path.append(str(Path(__file__).parent.parent))

app = Flask(__name__)
processing_lock = threading.Lock()


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
            output = generate(model, tokenizer, inputs['input'])
        response = {
            "response": output
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

if __name__ == '__main__':

    model_path = "/netcache/huggingface/Qwen2.5-7B-Instruct"
    lora_path = "/netcache/hekaiyu/project/save"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remoteg_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    model = LLM(
        model=model_path,
        enable_lora=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8
    )
    app.run(host=get_local_ip(), port=10908, threaded=True)
