from datasets import load_dataset

if __name__ == "__main__":
    data_path = "/netcache/hekaiyu/project/data/AI_content/deepseek"
    dataset = load_dataset("json",data_files=data_path + "/news.jsonl")['train']
    dataset = dataset.train_test_split(0.2)
    dataset.shuffle()
    print(dataset)
    dataset['train'].to_json(f"{data_path}/train/train.jsonl", force_ascii=False)
    dataset['test'].to_json(f"{data_path}/test/test.jsonl", force_ascii=False)