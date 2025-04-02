# 文本概括

## 数据集

- [CLTS-Dataset](https://github.com/lxj5957/CLTS-Dataset?tab=readme-ov-file)
- [LCSTS](https://hf-mirror.com/datasets/hugcyp/LCSTS)
- 自行构建的数据集
  - 获取小红书或其他网址文档
  - deepseek对文章进行总结作为答案


## 模型选择

- Qwen2.5-7B-Instruction 0.46039065914775373
- deepseek-7B
- llama3-8B-Instruction

## 微调

- deepspeed 加速
- 提示词优化
- 100 step 保留一个检查点
- zero-shot or few-shot
- 微调策略
  - 直接混合训练

## 能力评估
- 域内任务
  - 评价指标
    - rouge-L
  - 策略1：
    - 三个模型同时得到输出
    - 相互计算rouge-L
    - 取最高得分的


## GUI设计

## 识别AI文本
- 使用AI构建数据集
  - 根据概括撰写主要内容
- 原数据集默认不为AI撰写的文本

- 识别
  - 整体识别
  - 划分句子级别识别，两-四句为一个单位？。
