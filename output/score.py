import json

from rouge import Rouge
import jieba

def concat_answer(answers):
    rouge = Rouge()
    final_answer = ["", 0]
    for sentence1 in answers:
        score = 0
        for sentence2 in answers:
            score += rouge.get_scores(sentence1, sentence2)[0]['rouge-l']['f']
        if score > final_answer[1]:
            final_answer = (sentence1, score)
    return final_answer[0]
def metric(output, answer):
    rouge = Rouge()
    output = ' '.join(jieba.cut(output))
    answer = ' '.join(jieba.cut(answer))
    score = rouge.get_scores(output, answer)[0]['rouge-l']['f']
    return score

if __name__ ==  "__main__":
    model_names = [
        # "Qwen2.5",
        "llama3",
    ]
    outputs = []
    answer = []
    for i, model_name in enumerate(model_names):
        outputs.append([])
        with open(f"/netcache/hekaiyu/project/output/{model_name}.jsonl", "r")as f:
            for line in f:
                line = json.loads(line)
                outputs[-1].append(line['output'])
                if i == 0:
                    answer.append(line['answer'])
    score = []
    for i in range(len(outputs[0])):
        concat = []
        for j in range(len(outputs)):
            concat.append(outputs[j][i])
        concat = concat_answer(concat)
        score.append(metric(concat, answer[i]))
    print(f"{sum(score)/len(score)}")

