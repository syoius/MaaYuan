import difflib
import string
import pandas as pd
import json
import os
import difflib


import pandas as pd
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


@AgentServer.custom_action("AutoAnswer")
class AutoAnswer(CustomAction):
    def __init__(self):
        super().__init__()
        self.question_bank = self.read_qa_excel("agent/qadb.xlsx")
        self.similarity_threshold = 0.5  # 相似度阈值
        self.current_question = ""  # 保存当前问题
        self.current_answers = []  # 保存当前答案列表
        print(f"题库加载完成，共{len(self.question_bank)}道题目")

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        print("开始自动答题")
        question = self.get_question(context)
        if not question:
            print(f"错误：未能识别到问题")
            return False

        answers = self.get_answer(context)
        if not answers:
            print(f"错误：未能识别到答案")
            return False

        # 保存当前问题和答案，供后续使用
        self.current_question = question
        self.current_answers = answers
        correct_answer = self.find_question(question, answers)
        if correct_answer:
            # 点击正确答案
            if self.click_correct_answer(context, answers, correct_answer):
                return True
            else:
                print("点击答案失败，请手动点击")
                return False
        else:
            print(f"未找到匹配的问题")
            return False

    def clean_text(self, text):
        # 创建翻译表，将所有标点符号和空格映射为None
        chinese_punctuation = "，。！？【】（）《》“”‘’；：、——·〈〉……—"
        translator = str.maketrans(
            "", "", string.punctuation + chinese_punctuation + " \t\n\r\u3000"
        )
        # 使用翻译表移除标点符号和空格
        cleaned_text = text.translate(translator)
        return cleaned_text.strip()

    def get_question(self, context: Context) -> str:
        question = ""
        img = context.tasker.controller.post_screencap().wait().get()
        result = context.run_recognition("披荆斩棘-识别题目", img)

        if result and result.filterd_results:
            for r in result.filterd_results:
                question = question + r.text
        else:
            print(f"警告：未能识别到题目文本")
        question = self.clean_text(question)
        print(f"识别到的题目: {question}")
        return question.strip()

    def get_answer(self, context: Context) -> list[dict[str, list]]:
        img = context.tasker.controller.post_screencap().wait().get()
        answers = []

        for i in range(1, 5):  # 自动循环识别四个答案
            result = context.run_recognition(f"披荆斩棘-识别选项_{i}", img)
            if result and result.best_result:
                answer_text = result.best_result.text.strip()
                # 清理答案文本
                answer_text = self.clean_text(answer_text)
                answer_data = {"text": answer_text, "box": result.best_result.box}
                answers.append(answer_data)
                print(f"选项{i}: {answer_data['text']}")

        return answers

    def find_question(self, question, answers):

        best_match = None
        max_sim = 0

        for item in self.question_bank:
            # 截断题干到前 25 个字（如果超了）
            item_q = item["q"][:25] if len(item["q"]) > 25 else item["q"]

            full_text = f"{item_q} {' '.join(sorted(item['a']))}"
            input_text = (
                f"{question} {' '.join(sorted([ans['text'] for ans in answers]))}"
            )
            q_sim = difflib.SequenceMatcher(None, question, item_q).ratio()
            sim = difflib.SequenceMatcher(None, input_text, full_text).ratio()

            # 选用问题相似度和综合相似度的最小值
            sim = min(q_sim, sim)  # 综合相似度
            if sim > max_sim:
                max_sim = sim
                best_match = item

        print(f"最佳匹配题目: {best_match['q']}")
        print(f"正确答案为: {best_match['ans']}")
        print("相似度", max_sim)
        return (
            best_match["ans"] if max_sim > self.similarity_threshold else None
        )  # 相似度阈值

    def click_correct_answer(
        self, context: Context, answers: list[dict[str, object]], correct_answer: str
    ):
        """
        点击正确答案
        """
        # 获得相似度列表
        similarities = []
        for answer in answers:
            sim = difflib.SequenceMatcher(None, answer["text"], correct_answer).ratio()
            similarities.append((sim, answer))

        # 按相似度排序（高到低）
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 获取最高相似度和第二高
        if not similarities:
            print("没有可用选项")
            return False
        max_sim, best_match = similarities[0]
        second_sim = similarities[1][0] if len(similarities) > 1 else 0

        # 检查最高相似度是否超过阈值，并且与第二高相似度有明显差距
        if max_sim > self.similarity_threshold and (max_sim - second_sim > 0.05):
            box = best_match["box"]
            center_x = box[0] + box[2] // 2
            center_y = box[1] + box[3] // 2
            context.tasker.controller.post_click(center_x, center_y).wait()
            print(f"已点击选项: {best_match['text']}")
            return True
        else:
            print(f"警告：未能明确匹配到正确答案 '{correct_answer}'的选项")
            return False

    def stop(self):
        pass

    def read_qa_excel(self, file_path):
        # 读取第3个sheet并跳过第一行
        df = pd.read_excel(file_path, sheet_name=3).iloc[1:]

        # 删除问题和选项任意一个为空的
        df = df.dropna(subset=df.columns[2:8], how="any")

        results = []
        for _, row in df.iterrows():
            question = self.clean_text(row.iloc[2])
            options = [self.clean_text(row.iloc[i]) for i in range(4, 8)]
            answer = str(row.iloc[3])

            # 处理全选和多选
            if "全选" in answer:
                answer = options[0]
            elif "多选" in answer:
                answer = answer.split("/")[1]
            answer = self.clean_text(answer)

            results.append({"q": question, "ans": answer, "a": options})
        return results
