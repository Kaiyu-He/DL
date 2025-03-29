import numpy as np
import requests
import threading
import time

def generate(input, max_new_token=20) -> dict:
    start = time.time()
    try:
        response = requests.post(
            'http://210.75.240.147:14425/generate',
            json={
                "input": input,
                "max_new_token": max_new_token,
            },
        )
        elapsed = time.time() - start
        return response.json()
    except Exception as e:
        elapsed = time.time() - start
        print(e)

def get_loss(input_text= "", answer = None) -> dict:
    if type(input_text) == str:
        input_text = [input_text]
    if type(answer) == str:
        answer = [answer]

    start = time.time()
    try:
        response = requests.post(
            'http://210.75.240.147:14425/get_loss',
            json={
                "input": input_text,
                "answer": answer
            },
        )
        elapsed = time.time() - start
        return response.json()
    except Exception as e:
        elapsed = time.time() - start
        print(e)

if __name__ == "__main__":
    context1 = "中新网3月4日电 国台办发言人朱凤莲3月4日表示，由于民进党当局一再阻挠，1148名急需返乡的滞鄂台胞迄今无法回家。苏贞昌日前又公开散布“苏式谎言”，继续罔顾事实、颠倒黑白，谎称“卡关就卡在大陆”，“真不知人间还有羞耻二字。”朱凤莲说，疫情发生以来，大陆方面一方面全力照顾在大陆台胞的生活和疫情防控需要，另一方面充分考虑滞鄂台胞的实际需求和回家心愿，积极安排东航于2月3日运送首批247名台胞返回台湾，并于2月5日和此后多次提出尽快运送其他提出返乡要求台胞的合理安排，包括提出由两岸航空公司共同执飞临时航班的运送安排，以满足滞鄂台胞急切回家的愿望。但民进党当局却一而再、再而三变换借口，不断设置障碍，一再拖延阻挠。“2月15日，我办发言人已详细披露大陆方面持续做出运送台胞安排和为实现运送不懈努力的全过程和细节，具体情况清清楚楚，事实真相一目了然。”朱凤莲指出，民进党当局不断以各种借口阻止东航后续运送，有目共睹。苏贞昌自己就曾公开说过，不能让在湖北的台胞回去，是因为岛内防疫安置能量不足。更有甚者，民进党当局竟然将期待返乡就业、学习团聚等1148名台胞列入所谓“注记管制名单”，全面封堵了滞鄂台胞回家之路。事实反复证明，民进党当局根本就不想让在湖北的台胞回家，滞鄂台胞返乡之路受阻，“卡关”就卡在民进党当局的这些政客手中。朱凤莲强调，苏贞昌企图以自相矛盾的谎言转移视线、推卸责任，未免低估了广大台胞的智商。“我们奉劝他要有起码的道德底线，停止信口雌黄，停止造谣生事。我们质问他，敢不敢讲立即同意这1148名台胞返乡？”"
    context2 = "在当时那个特殊阶段，湖北地区面临诸多挑战与困难，许多台胞因种种客观因素滞留当地。这本是两岸携手共克时艰、展现同胞温情的时刻，然而民进党当局却背道而驰。国台办发言人朱凤莲于 3 月 4 日严肃发声，道出背后令人心寒的真相：整整 1148 名滞鄂台胞，他们怀着对家乡、对亲人的深切思念，急切渴望踏上归乡之路，其中不乏身体抱恙急需回到台湾接受后续治疗的同胞，有因长时间与家人分离精神焦虑渴望团聚抚慰心灵的民众，还有面临事业困境急需返乡重整旗鼓之人。"
    input_text = f"""### 角色定位
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
{context2}

### 回答
答案：
"""
    print(input_text)
    print(generate(input_text, max_new_token=20))
    print(get_loss(input_text, "是"))
    print(get_loss(input_text, "否"))
    loss1 = get_loss(context1)['loss'][0]
    print(loss1)
    print(np.var(loss1, ddof=0))
    loss2 = get_loss(context2)['loss'][0]
    print(loss2)
    print(np.var(loss2, ddof=0))