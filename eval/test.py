import json
import os
import numpy as np
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import tqdm
import wandb
import torch


def write_to_file(datasets, file_path):
    content = '\n'.join(datasets)
    with open(file_path, 'a') as f:
        f.write(content + '\n')


def get_dataset(data_path, tokenizer, prompt_type, model_type):
    dataset = load_dataset("json", data_files=data_path)

    def text_prepocess_function(examples):
        inputs = []
        for comment in examples["comment"]:

            if prompt_type == "original":
                input_text = f"判断一下评论是好评还是差评：{comment}\n答案：（好评/差评）"
            comment = "评论：" + comment.replace('\n', ' ')
            system = "请对以下评论进行判断，仅回答 '好评' 或 '差评' 两个字。"
            if model_type == "map-neo":
                comment = '<<SYS>>\n' + system + '\n<</SYS>>\n\n' + comment
            if prompt_type == "zero-shot":
                chat = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{comment}"}
                ]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False)
            if prompt_type == "two-shot":
                chat = [
                    {"role": "system", "content": "请对以下评论进行判断，仅回答 '好评' 或 '差评' 两个字。"},
                    {"role": "user",
                     "content": f"评论：一般。因为我不吃辣。所以只吃了几个不辣的菜。所以发言权不大。朋友吃的很high，应该不错吧。酸梅汤超大扎\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：都晓得在春熙路随便找个地方歇哈脚都要花脱几十块钱 这家店我们从找到后 就一直在这耍 下午可以坐一下午 环境可以 很清静 那么闹市区的地方，人居然一直都不多 沙发也软软的 不过东西的味道就一般了 招行的卡可以打折\n"},
                    {"role": "assistant", "content": f"好评"},
                    {"role": "user", "content": f"评论：{comment}"}
                ]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False)
            if prompt_type == "few-shot":
                chat = [
                    {"role": "system", "content": "请对以下评论进行判断，仅回答 '好评' 或 '差评' 两个字。"},
                    {"role": "user",
                     "content": f"评论：公司在这里借了会议室给大家培训用的，第一次去给人印象觉得他们会议室还挺宽敞额，之后再次去到那里，把灯统统打开，才发现：说大也不算很大，100个平方米，桌子旧了点还有高低脚额，负责会议室额师傅，不能说他好，但是你也不好说他不好，你让他做的事情他都做了，可是做到位了吗。。。貌似没有。。。地毯上脏兮兮额，不只是会议室。。。桌面脏兮兮额，怎么擦也擦不干净。。。椅子有些还不错，有些绿色绒面额有香烟烫掉的痕迹。。。\n总之，环境真滴是不咋的，但是还算比较便宜的，一天3800元人民币。。。不过估计以后不会再租这边，有些小失望\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：都晓得在春熙路随便找个地方歇哈脚都要花脱几十块钱 这家店我们从找到后 就一直在这耍 下午可以坐一下午 环境可以 很清静 那么闹市区的地方，人居然一直都不多 沙发也软软的 不过东西的味道就一般了 招行的卡可以打折\n"},
                    {"role": "assistant", "content": f"好评"},
                    {"role": "user",
                     "content": f"评论：一般。因为我不吃辣。所以只吃了几个不辣的菜。所以发言权不大。朋友吃的很high，应该不错吧。酸梅汤超大扎\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：头一次点评给这么差,但这里真的是太次了,各种的不灵,一进门排队拿号,吃饭等位,休息室各种的满员,服务员来一句花钱的房间都没了,洗澡总共十个喷子,依然是排队,人太多太乱,服务员态度极差,让拿什么都不乐意还皱着眉头,吃饭等到时已经基本全都光了,选这里就是一个错误,主要是觉得没来过想看看如何,但真就没去过这么次的地,本来是放松来的,结果真是各种的不满,出来结帐时同样听到别人在说跟我同样的感受,以后决不会在来了,拜拜..\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：味道太赞了 环境也不错 就是得加强宣传 位置比较隐蔽 以后还回去的 +20升级208的活动不错 可以吃到鹅肝牛排海胆什么的\n"},
                    {"role": "assistant", "content": f"好评"},
                    {"role": "user", "content": f"{comment}"}
                ]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False)
            if prompt_type == "few-shot2":
                chat = [

                    {"role": "user",
                     "content": f"评论：公司在这里借了会议室给大家培训用的，第一次去给人印象觉得他们会议室还挺宽敞额，之后再次去到那里，把灯统统打开，才发现：说大也不算很大，100个平方米，桌子旧了点还有高低脚额，负责会议室额师傅，不能说他好，但是你也不好说他不好，你让他做的事情他都做了，可是做到位了吗。。。貌似没有。。。地毯上脏兮兮额，不只是会议室。。。桌面脏兮兮额，怎么擦也擦不干净。。。椅子有些还不错，有些绿色绒面额有香烟烫掉的痕迹。。。\n总之，环境真滴是不咋的，但是还算比较便宜的，一天3800元人民币。。。不过估计以后不会再租这边，有些小失望\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：都晓得在春熙路随便找个地方歇哈脚都要花脱几十块钱\n这家店我们从找到后\n就一直在这耍\n下午可以坐一下午\n环境可以\n很清静\n那么闹市区的地方，人居然一直都不多\n沙发也软软的\n不过东西的味道就一般了\n招行的卡可以打折\n"},
                    {"role": "assistant", "content": f"好评"},
                    {"role": "user",
                     "content": f"评论：一般。因为我不吃辣。所以只吃了几个不辣的菜。所以发言权不大。朋友吃的很high，应该不错吧。酸梅汤超大扎\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：头一次点评给这么差,但这里真的是太次了,各种的不灵,一进门排队拿号,吃饭等位,休息室各种的满员,服务员来一句花钱的房间都没了,洗澡总共十个喷子,依然是排队,人太多太乱,服务员态度极差,让拿什么都不乐意还皱着眉头,吃饭等到时已经基本全都光了,选这里就是一个错误,主要是觉得没来过想看看如何,但真就没去过这么次的地,本来是放松来的,结果真是各种的不满,出来结帐时同样听到别人在说跟我同样的感受,以后决不会在来了,拜拜..\n"},
                    {"role": "assistant", "content": f"差评"},
                    {"role": "user",
                     "content": f"评论：味道太赞了 环境也不错 就是得加强宣传 位置比较隐蔽 以后还回去的 +20升级208的活动不错 可以吃到鹅肝牛排海胆什么的\n"},
                    {"role": "assistant", "content": f"好评"},
                    {"role": "user", "content": f"请对以下评论进行判断，仅回答 '好评' 或 '差评' 两个字。\n{comment}"}

                ]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False)
            if model_type == "map-neo":
                input_text = input_text
            if model_type == "Qwen2-7B":
                input_text = input_text + "<|im_start|>assistant: "
            if model_type == "llama":
                input_text = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs.append(input_text)
        labels = [label for label in examples["label"]]
        scores = [score for score in examples["score"]]
        return {
            "inputs_text": inputs,
            "labels": labels,
            "score": scores
        }

    # 对数据集应用预处理函数
    dataset = dataset.map(text_prepocess_function, batched=True, remove_columns=dataset["train"].column_names)
    return dataset


def eval(model, tokenizer, dataset, prompt_type, model_type):
    model.eval()
    error = []
    acc, err, num, step = 0, 0, 0, 0
    eval = np.array([[0, 0], [0, 0]])
    score = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    result = {}
    model.eval()
    for item in tqdm.tqdm(dataset, total=len(dataset)):
        print(item["inputs_text"])
        input_text = item["inputs_text"]
        inputs_ids = tokenizer(input_text, return_tensors='pt').to('cuda')
        outputs_ids = model.generate(
            input_ids=inputs_ids.input_ids,
            attention_mask=inputs_ids.attention_mask,
            max_new_tokens=20
        )
        outputs_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_ids.input_ids, outputs_ids)
        ]
        output_text = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)[0]
        print(output_text)
        output_label = 0
        if len(output_text) >= 2:
            if output_text.find("好") >= 0:
                output_label = 1
        if output_text.find("好评") == -1 and output_text.find("差评") == -1:
            err += 1
        print(output_label)
        print(item['labels'])
        print(item["score"])
        score[output_label][item["score"]] += 1
        num += 1
        if output_label == 0 or output_label == 1:
            eval[output_label][item['labels']] += 1
        else:
            err += 1
            error.append(json.dumps(item))
        step += 1
        epsilon = 1e-7
        if step % 10 == 0:
            result["score"] = score
            result["accuracy"] = eval
            result["F1"] = 2 * eval[1][1] / (2 * eval[1][1] + eval[0][1] + eval[1][0] + epsilon)
            result["error_num"] = err
            result["acc"] = (eval[0][0] + eval[1][1]) / num
            print(result)
            error = []
            result_serializable = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in
                                   result.items()}
    wandb.log(
        {"F1": result["F1"], "acc": result["acc"], "error": err, "model_type": model_type, "prompt_type": prompt_type})


def eval_bert(model, tokenizer, dataset_path):
    model.eval()

    def collate_fn(data):
        sents = [i['comment'] for i in data]
        labels = [i['label'] for i in data]
        data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=500,
                                           return_tensors='pt',
                                           return_length=False
                                           ).to("cuda")
        labels = torch.LongTensor(labels).to("cuda")
        return data, labels

    correct = 0
    total = 0
    test_dataset = load_dataset("json", data_files=dataset_path)["train"]
    loader_test = DataLoader(dataset=test_dataset,
                             batch_size=256,
                             collate_fn=collate_fn,
                             )
    eval = np.array([[0, 0], [0, 0]])
    for i, (data, label) in enumerate(loader_test):
        with torch.no_grad():
            out = model(data)
        out = out.argmax(dim=1)
        correct += (out == label).sum().item()
        total += len(label)
        for x, y in zip(out, label):
            eval[x][y] += 1
    epsilon = 1e-7
    F1 = 2 * eval[1][1] / (2 * eval[1][1] + eval[0][1] + eval[1][0] + epsilon)
    print(correct / total)
    wandb.log({"final_acc": correct / total, "final_F1": F1})




os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
data_path = "../dataset/test_dataset.jsonl"


def use_model(model_type, prompt_type):
    if model_type == "bert":
        model_path = "../model/result/bert/model-1.pt"
        model = torch.load(model_path)
        prompt_type = "bert"
    if model_type == "bert-all":
        model_path = "../model/result/bert/model-all.pt"
        model = torch.load(model_path)
        prompt_type = "bert"
    if model_type == "Qwen2-7B":
        model_path = f"/netcache/huggingface/Qwen2-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if model_type == "llama":
        model_path = "../model/llama-3-chinese"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if model_type == "map-neo":
        model_path = "../model/map-neo"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if model_type == "sft":
        model_type = "Qwen2-7B"
        model_path = f"/netcache/huggingface/Qwen2-7B-Instruct"
        lora_path = "../model/result/Qwen2-0/checkpoint-1250"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        model = PeftModel.from_pretrained(model, lora_path)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    wandb.init(
        name=f"{model_type} + {prompt_type}"
    )
    model.eval()
    if model_type == "bert":
        eval_bert(model, tokenizer, dataset_path=data_path)
    else:
        dataset = get_dataset(data_path, tokenizer=tokenizer, prompt_type=prompt_type, model_type=model_type)["train"]
        eval(model=model, dataset=dataset,tokenizer=tokenizer, model_type=model_type, prompt_type=prompt_type)
    wandb.finish()
    torch.cuda.empty_cache()

os.environ["WANDB_PROJECT"] = "test2"
os.environ["WANDB_API_KEY"] = "666bff72d04cfb09fb241f5152a186617627651b"
if __name__ == '__main__':

    # model_type = "Qwen2-7B"   bert 、bert-all 、Qwen2-7B、llama、map-neo、sft
    # prompt_type = "original"  original 、zero-shot、two-shot 、few-shot
    use_model(model_type="map-neo", prompt_type="zero-shot")
    use_model(model_type="map-neo", prompt_type="two-shot")
    use_model(model_type="map-neo", prompt_type="few-shot")


