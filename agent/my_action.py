
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
        if not question:
            print("错误：未能识别到问题")
            return False
        

        answers = self.get_answer(context)
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
        
    def clean_text(self,text):
        # 创建翻译表，将所有标点符号和空格映射为None
        chinese_punctuation = '，。！？【】（）《》“”‘’；：、——·'
        translator = str.maketrans('', '', string.punctuation + chinese_punctuation + ' ')
        # 使用翻译表移除标点符号和空格
        cleaned_text = text.translate(translator)
        return cleaned_text
    
    def get_question(self, context: Context) -> str:
        question = ""
        img = context.tasker.controller.post_screencap().wait().get()
        result = context.run_recognition("披荆斩棘-识别题目", img)

        if result and result.filterd_results:
            print(f"识别到 {len(result.filterd_results)} 个文本片段")
            for r in result.filterd_results:
                question = question + r.text
        else:
            print("警告：未能识别到题目文本")
        question = self.clean_text(question)
        print(f"识别到的题目: {question}")
        return question.strip()

    def get_answer(self, context: Context) -> list[dict[str, list]]:
        img = context.tasker.controller.post_screencap().wait().get()
        answers = []

        for i in range(1, 5):  # 自动循环识别四个答案
            result = context.run_recognition(f"披荆斩棘-识别选项_{i}", img)
            if result and result.best_result:
                answer_text= result.best_result.text.strip()
                # 清理答案文本
                answer_text = self.clean_text(answer_text)
                answer_data = {
                    "text": answer_text,
                    "box": result.best_result.box
                }
                answers.append(answer_data)
                print(f"选项{i}: {answer_data['text']}, 位置: {answer_data['box']}")

        return answers
    
    def find_question(self,question, answers):

        best_match = None
        max_sim = 0
        
        for item in self.question_bank:
            full_text = f"{item['q']} {' '.join(item['a'])}"
            input_text = f"{question} {' '.join([ans['text'] for ans in answers])}"
            q_sim = difflib.SequenceMatcher(None, question, item['q']).ratio()
            sim = difflib.SequenceMatcher(None, input_text, full_text).ratio()
            #选用问题相似度和综合相似度的最小值
            sim=min(q_sim, sim)  # 综合相似度
            if sim > max_sim:
                max_sim = sim
                best_match = item
        print(f"最佳匹配题目: {best_match['q']}")
        print(f"正确答案: {best_match['ans']}")
        print("相似度",max_sim)
        return best_match['ans'] if max_sim > self.similarity_threshold else None  # 相似度阈值


    def click_correct_answer(self, context: Context, answers: list[dict[str, object]], correct_answer: str):
        """
        点击正确答案
        """
        for answer in answers:
            # if answer['text'] in correct_answer:
            if difflib.SequenceMatcher(None, answer['text'], correct_answer).ratio()>0.8 or answer['text'] in correct_answer:
                # 计算答案框中心点
                box = answer['box']
                print(f"找到答案 '{answer['text']}' 的位置: {box}")
                center_x = box[0] + box[2] // 2
                center_y = box[1] + box[3] // 2
                print(f"即将点击坐标: ({center_x}, {center_y})")
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
        df=df.iloc[1:]
        #删除问题和正确答案任意一个为空的
        df = df.dropna(subset=[df.columns[2], df.columns[3], df.columns[4], df.columns[5], df.columns[6], df.columns[7]], how='any')
        #构造答题json文件
        results=[]
        for _,row in df.iterrows():
            question=row.iloc[2]
            #问题去掉括号和空格
            question = self.clean_text(question)
            answer = self.clean_text(row.iloc[3])    # 第4列是答案
            options = [
                f"{self.clean_text(row.iloc[4])}", 
                f"{self.clean_text(row.iloc[5])}", 
                f"{self.clean_text(row.iloc[6])}", 
                f"{self.clean_text(row.iloc[7])}"
            ]
            if "全选" in answer:
                answer="/".join(options)
            item = {
            "q": str(question), 
            "ans": str(answer).strip(),
            "a": options
            }
            results.append(item)
        return results

