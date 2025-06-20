
import difflib
import pandas as pd
import json
import os
import difflib


import pandas as pd
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

@AgentServer.custom_action("auto_answer")
class QuestionMatcher(CustomAction):
    def __init__(self):
        super().__init__()
        self.question_bank = self.read_qa_excel("agent/题库收集.xlsx")
        self.similarity_threshold = 0.7  # 相似度阈值
        self.current_question = ""  # 保存当前问题
        self.current_answers = []  # 保存当前答案列表
        print(f"题库加载完成，共{len(self.question_bank)}道题目")

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        print("开始执行自定义动作：问题匹配")
        question = self.get_question(context)
        print(f"识别的问题: {question}")
        if not question:
            print("错误：未能识别到问题")
            return False

        answers = self.get_answer(context)
        print(f"识别的答案: {answers}")
        if not answers:
            print("错误：未能识别到答案")
            return False

        # 保存当前问题和答案，供后续使用
        self.current_question = question
        self.current_answers = answers
        correct_answer = self.find_question(question, answers)
        if correct_answer:
            print(f"找到正确答案: {correct_answer}")
            # 点击正确答案
            self.click_correct_answer(context, answers, correct_answer)
            return True
        else:
            print("未找到匹配的问题")
            return False
    
    def get_question(self, context: Context) -> str:
        question = ""
        img = context.tasker.controller.post_screencap().wait().get()
        result = context.run_recognition("披荆斩棘-识别题目", img)

        if result and result.filterd_results:
            print(f"识别到 {len(result.filterd_results)} 个文本片段")
            for r in result.filterd_results:
                print(f"文本片段: {r.text}")
                question = question + r.text
        else:
            print("警告：未能识别到题目文本")

        return question.strip()

    def get_answer(self, context: Context) -> list[dict[str, list]]:
        img = context.tasker.controller.post_screencap().wait().get()
        answers = []

        for i in range(1, 5):  # 自动循环识别四个答案
            result = context.run_recognition(f"披荆斩棘-识别选项_{i}", img)
            if result and result.best_result:
                answer_data = {
                    "text": result.best_result.text.strip(),
                    "box": result.best_result.box
                }
                answers.append(answer_data)
                print(f"答案{i}: {answer_data['text']}, 位置: {answer_data['box']}")

        return answers
    
    def find_question(self,question, answers):

        best_match = None
        max_sim = 0
        
        for item in self.question_bank:
            full_text = f"{item['q']} {' '.join(item['a'])}"
            input_text = f"{question} {' '.join([ans['text'] for ans in answers])}"
            sim = difflib.SequenceMatcher(None, input_text, full_text).ratio()
            if sim > max_sim:
                max_sim = sim
                best_match = item
        print(max_sim)
        return best_match['ans'] if max_sim > self.similarity_threshold else None  # 综合相似度阈值

    def get_and_save_correct_answer(self, context: Context) -> None:
        """
        获取正确答案的标识框，通过高度匹配找到正确答案并保存
        """
        img = context.tasker.controller.post_screencap().wait().get()
        result = context.run_recognition("披荆斩棘-识别正确答案", img)

        if not result or not result.best_result:
            print("警告：未能识别到正确答案标识")
            return

        correct_box = result.best_result.box
        print(f"正确答案标识框: {correct_box}")

        # 计算正确答案标识的中心Y坐标
        correct_y = (correct_box[1] + correct_box[3]) // 2
        print(f"正确答案标识中心Y坐标: {correct_y}")

        # 找到Y坐标最接近的答案
        min_distance = float('inf')
        correct_answer_text = ""
        correct_answer_index = -1

        for idx, answer in enumerate(self.current_answers):
            answer_box = answer['box']
            # 计算答案框的中心Y坐标
            answer_y = (answer_box[1] + answer_box[3]) // 2
            # 计算距离
            distance = abs(correct_y - answer_y)

            print(f"答案{idx + 1} '{answer['text']}' 中心Y坐标: {answer_y}, 距离: {distance}")

            if distance < min_distance:
                min_distance = distance
                correct_answer_text = answer['text']
                correct_answer_index = idx

        if correct_answer_text:
            print(f"识别出正确答案是第{correct_answer_index + 1}个: {correct_answer_text}")
            # 更新CSV文件中的正确答案
            self.update_correct_answer(self.current_question, correct_answer_text)
        else:
            print("错误：无法确定正确答案")
    
    
    def click_correct_answer(self, context: Context, answers: list[dict[str, object]], correct_answer: str):
        """
        点击正确答案
        """
        for answer in answers:
            if answer['text'] in correct_answer:
                # 计算答案框中心点
                box = answer['box']
                center_x = box[0] + box[2] // 2
                center_y = box[1] + box[3] // 2

                # 执行点击
                context.tasker.controller.post_click(center_x, center_y).wait()
                print(f"已点击答案: {correct_answer}")
                break
        else:
            print(f"警告：未找到正确答案 '{correct_answer}' 的位置")

    def stop(self):
        pass

    
    def read_qa_excel(self,file_path):
        #读取excel
        df = pd.read_excel(file_path, sheet_name=2)
        df=df.iloc[2:]
        #删除问题和正确答案任意一个为空的
        df = df.dropna(subset=[df.columns[2], df.columns[3], df.columns[4], df.columns[5], df.columns[6], df.columns[7]], how='any')
        #构造答题json文件
        results=[]
        for _,row in df.iterrows():
            question=row[2]
            answer = row[3]    # 第4列是答案
            options = [
                f"{row[4]}", 
                f"{row[5]}", 
                f"{row[6]}", 
                f"{row[7]}"
            ]
            if "全选" in answer:
                answer="/".join(options)
                # print(f"全选题目：{question}，答案：{answer}")
            item = {
            "q": str(question),  # 确保是字符串
            "ans": str(answer).strip(),
            "a": options
            }
            results.append(item)
        return results

