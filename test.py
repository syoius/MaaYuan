import pandas as pd
import random
from typing import List, Dict, Optional


class DecisionAgent:
    """基于Excel表格的决策代理"""

    def __init__(self, excel_file_path: str):
        """
        初始化决策代理

        Args:
            excel_file_path: Excel文件路径
        """
        self.excel_file_path = excel_file_path
        self.data = None
        self.load_data()

    def load_data(self):
        """加载Excel数据"""
        try:
            # 读取Excel文件
            self.data = pd.read_excel(self.excel_file_path, sheet_name=1)
            print(f"成功加载Excel文件，共{len(self.data)}行数据")
            print(f"列名: {list(self.data.columns)}")
        except Exception as e:
            print(f"加载Excel文件失败: {e}")
            raise

    def find_event_options(self, event_name: str) -> List[Dict]:
        """
        根据事件名称查找所有相关的选项

        Args:
            event_name: 事件名称

        Returns:
            包含所有选项的列表
        """
        if self.data is None:
            raise ValueError("数据未加载")

        # 查找匹配的事件
        matching_rows = self.data[self.data["事件名称"] == event_name]

        if matching_rows.empty:
            return []

        options = []
        for _, row in matching_rows.iterrows():
            # 提取选项文本和结果文本
            option_text = row.get("选项文本", "")
            ocr_text = row.get("OCR用", "")
            # result_text = row.get("结果文本", "")
            label = row.get("label", "")

            if pd.notna(option_text) and option_text.strip():
                options.append(
                    {
                        "option_text": option_text.strip(),
                        "ocr_text": ocr_text.strip(),
                        # "result_text": (
                        #     result_text.strip() if pd.notna(result_text) else ""
                        # ),
                        "label": label.strip() if pd.notna(label) else "",
                        "row_index": row.name,
                    }
                )

        return options

    def get_decision(self, event_name: str, decision_type: str = "贤明") -> Dict:
        """
        根据事件名称和决策类型获取决策结果

        Args:
            event_name: 事件名称
            decision_type: 决策类型 ("贤明" 或 "混沌")

        Returns:
            包含选择的选项信息的字典
        """
        # 查找事件的所有选项
        options = self.find_event_options(event_name)

        if not options:
            return {
                "success": False,
                "message": f'未找到事件"{event_name}"的相关选项',
                "event_name": event_name,
                "decision_type": decision_type,
            }

        # 根据决策类型筛选选项
        filtered_options = []

        if decision_type == "贤明":
            # 贤明决策：选择标签为"贤明"的选项，如果没有则选择没有特定标签的选项
            filtered_options = [opt for opt in options if opt["label"] == "贤明"]
            if not filtered_options:
                filtered_options = [
                    opt
                    for opt in options
                    if opt["label"] == "" or opt["label"] == "混沌没有发生变化"
                ]

        elif decision_type == "混沌":
            # 混沌决策：选择标签为"混沌"的选项，如果没有则随机选择
            filtered_options = [opt for opt in options if opt["label"] == "混沌"]
            if not filtered_options:
                filtered_options = options

        else:
            # 其他情况：返回所有选项
            filtered_options = options

        if not filtered_options:
            return {
                "success": False,
                "message": f'事件"{event_name}"没有适合"{decision_type}"类型的选项',
                "event_name": event_name,
                "decision_type": decision_type,
                "all_options": options,
            }

        # 选择一个选项（如果有多个，随机选择）
        chosen_option = random.choice(filtered_options)

        return {
            "success": True,
            "event_name": event_name,
            "decision_type": decision_type,
            "chosen_option": chosen_option["option_text"],
            "ocr_text": chosen_option["ocr_text"],
            # "result": chosen_option["result_text"],
            "label": chosen_option["label"],
            "row_index": chosen_option["row_index"],
            "total_options": len(options),
            "filtered_options": len(filtered_options),
        }

    def list_all_events(self) -> List[str]:
        """获取所有事件名称列表"""
        if self.data is None:
            return []

        return self.data["事件名称"].dropna().unique().tolist()

    def get_event_summary(self, event_name: str) -> Dict:
        """
        获取事件的详细信息摘要

        Args:
            event_name: 事件名称

        Returns:
            事件摘要信息
        """
        options = self.find_event_options(event_name)

        if not options:
            return {"success": False, "message": f'未找到事件"{event_name}"'}

        # 统计各种标签的选项数量
        label_counts = {}
        for option in options:
            label = option["label"] or "无标签"
            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "success": True,
            "event_name": event_name,
            "total_options": len(options),
            "label_distribution": label_counts,
            "options": options,
        }


# 使用示例
def main():
    """主函数示例"""
    # 初始化决策代理（请替换为你的Excel文件路径）
    agent = DecisionAgent("agent\monopoly.xlsx")

    # 列出所有事件
    print("=== 所有事件列表 ===")
    events = agent.list_all_events()
    for i, event in enumerate(events, 1):
        print(f"{i}. {event}")

    print("\n" + "=" * 50 + "\n")

    # 示例：获取特定事件的决策
    test_events = ["人参", "村路事故"]

    for event_name in test_events:
        if event_name in events:
            print(f"=== 事件: {event_name} ===")

            # 贤明决策
            result_wise = agent.get_decision(event_name, "贤明")
            print(f"\n【贤明决策】")
            if result_wise["success"]:
                print(f"选择: {result_wise['chosen_option']}")
                print(f"结果: {result_wise['result']}")
                print(f"标签: {result_wise['label']}")
            else:
                print(f"失败: {result_wise['message']}")

            # 混沌决策
            result_chaos = agent.get_decision(event_name, "混沌")
            print(f"\n【混沌决策】")
            if result_chaos["success"]:
                print(f"选择: {result_chaos['chosen_option']}")
                print(f"结果: {result_chaos['result']}")
                print(f"标签: {result_chaos['label']}")
            else:
                print(f"失败: {result_chaos['message']}")

            print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
