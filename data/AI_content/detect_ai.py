import json
import requests
import time
from tqdm import tqdm
from collections import defaultdict


def generate(input_text, max_new_token=20) -> dict:
    start = time.time()
    try:
        response = requests.post(
            'http://210.75.240.138:14425/generate',
            json={
                "input": input_text,
                "max_new_token": max_new_token,
            },
        )
        elapsed = time.time() - start
        return response.json()
    except Exception as e:
        elapsed = time.time() - start
        print(e)
        return {"response": [f"Error: {str(e)}"]}


def extract_first_yes_no(response_list):
    """从response列表中提取第一个'是'或'否'"""
    if not response_list:
        return None

    combined_text = "".join(str(item) for item in response_list)

    for char in combined_text:
        if char == '是':
            return '是'
        elif char == '否':
            return '否'
    return None


if __name__ == "__main__":
    # 1. 读取JSON数据
    with open("expanded_news.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 2. 准备结果存储结构
    results = {
        "statistics": {
            "total_articles": 0,
            "yes_count": 0,
            "no_count": 0,
            "accuracy": 0.0,
            "unrecognized": 0
        },
        "details": []
    }

    # 3. 处理每篇文章
    for item in tqdm(json_data, desc="Processing Articles", unit="article"):
        article_id = item["id"]  # 直接从原始数据中获取id字段
        article = item["article"]

        input_text = f"""### 角色定位
你是一位专业的AI生成内容检测专家，擅长通过文本特征分析判断文章是否由AI撰写。具备自然语言处理、机器学习领域知识，熟悉各类AI写作模型的输出特征。

### 能力要求
1. 分析文本的语言模式、结构特征和内容连贯性
2. 识别常见AI生成文本的典型特征（如过度流畅、缺乏深度等）
3. 输出仅包含一个字"是"或"否"
4. 保持客观中立，避免主观臆断

### 知识储备
1. GPT系列、Claude等主流AI模型的输出特征
2. 人类写作与AI写作的核心差异点
3. 文本风格分析技术
4. 常见AI检测指标（如困惑度、突发性等）

### 文章内容
{article}

### 回答
答案：
"""
        result = generate(input_text, max_new_token=20)
        response_list = result.get("response", [])
        first_result = extract_first_yes_no(response_list)

        # 更新统计
        results["statistics"]["total_articles"] += 1
        if first_result == '是':
            results["statistics"]["yes_count"] += 1
        elif first_result == '否':
            results["statistics"]["no_count"] += 1
        else:
            results["statistics"]["unrecognized"] += 1

        # 计算实时准确率
        current_accuracy = results["statistics"]["yes_count"] / results["statistics"]["total_articles"]
        results["statistics"]["accuracy"] = current_accuracy

        # 保存详细信息
        results["details"].append({
            "article_id": article_id,
            "article_content": article[:100] + "..." if len(article) > 100 else article,  # 保存部分内容供参考
            "api_result": result,
            "detection_result": first_result if first_result else "unrecognized",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })



    # 4. 保存结果到JSON文件
    output_filename = f"detection_ai_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成! 结果已保存到 {output_filename}")
    print("最终统计:")
    print(f"总文章数: {results['statistics']['total_articles']}")
    print(
        f"'是'的数量: {results['statistics']['yes_count']} ({(results['statistics']['yes_count'] / results['statistics']['total_articles']) * 100:.2f}%)")
    print(
        f"'否'的数量: {results['statistics']['no_count']} ({(results['statistics']['no_count'] / results['statistics']['total_articles']) * 100:.2f}%)")
    print(
        f"未识别数量: {results['statistics']['unrecognized']} ({(results['statistics']['unrecognized'] / results['statistics']['total_articles']) * 100:.2f}%)")
    print(f"最终准确率: {results['statistics']['accuracy']:.2%}")