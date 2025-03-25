# 文本概括

## 数据集

- [CLTS-Dataset](https://github.com/lxj5957/CLTS-Dataset?tab=readme-ov-file)
- [LCSTS](https://hf-mirror.com/datasets/hugcyp/LCSTS)
- 自行构建的数据集
  - 获取小红书或其他网址文档
  - deepseek对文章进行总结作为答案


## 模型选择

- Qwen2.5-7B-Instruction
- aya-expanse-8B
- llama3-8B-Instruction

## 微调

- deepspeed 加速
- 提示词优化
- 100 step 保留一个检查点
- zero-shot or few-shot
- 微调策略
  - 直接混合训练
  - 课程学习 
    - 先短文本 后长文本+deepseek构建文本
  - 多个prompt随机选取应用

## 能力评估
- 域内任务
  - 评价指标
    - rouge-L
    - chrf++
    - loss
  - 策略1：
    - 三个模型同时得到输出
    - 相互计算rouge-L
    - 取最高得分的
  - 策略2：
    - 文本过滤——删去无用内容 or 筛选核心内容
    - cot
- 域外任务
  - 数据集
- 其他能力测试
  - 问答
  - 情感分类
  - 多选题


## GUI设计

## 识别AI文本
- 使用AI构建数据集
  - 根据概括撰写主要内容
- 原数据集默认不为AI撰写的文本

- 识别
  - 整体识别
  - 划分句子级别识别，两-四句为一个单位？。

## 可解释性

- 删去不同的层，看掌握情况的变化
  - FFN 层的参数删去后会急剧下降？
- 微调后的该任务能力增长曲线，和其他能力的下降曲线
- 单条知识损失的变化情况。
  - 嵌入假事实？
