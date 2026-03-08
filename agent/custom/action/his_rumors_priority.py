import time
import json
import difflib
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger

@AgentServer.custom_action("HisRumorsPriority")
class HisRumorsPriority(CustomAction):
    def __init__(self):
        super().__init__()
        self.priority_list = [
            "短刀",
            "绣衣楼标记",
            "酒盏",
            "鸢羽",
            "竹叶"
        ]
        # Similarity threshold for text matching (0.0 - 1.0)
        self.similarity_threshold = 0.6

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        # Update priority list from parameters if provided
        raw_param = argv.custom_action_param
        try:
            param_dict = json.loads(raw_param) if isinstance(raw_param, str) else raw_param
        except (json.JSONDecodeError, TypeError):
            param_dict = {}

        # 检查是否启用优先级选择
        priority_enabled = str(param_dict.get("enabled", "true")).lower() != "false" if param_dict else True

        if param_dict and isinstance(param_dict, dict):
            new_list = [None] * 5
            found_param = False
            for k, v in param_dict.items():
                if k.startswith("priority_"):
                    try:
                        idx = int(k.split("_")[1]) - 1
                        if 0 <= idx < 5:
                            new_list[idx] = v
                            found_param = True
                    except Exception:
                        pass

            if found_param:
                valid_new_list = [x for x in new_list if x is not None]
                if valid_new_list:
                    # 将默认列表中未被用户指定的项追加到末尾，保证全部选项都参与优先级比较
                    default_remaining = [item for item in self.priority_list if item not in valid_new_list]
                    self.priority_list = valid_new_list + default_remaining
                    logger.info(f"已根据参数更新优先级列表: {self.priority_list}")

        try:
            # 等待 1.5 秒，确保游戏 UI 和文字已完全渲染
            time.sleep(1.5)

            # 截取屏幕
            img = context.tasker.controller.post_screencap().wait().get()
            if img is None:
                logger.error("HisRumorsPriority: 截图失败。")
                return False

            override_config = {
                "HisRumorsPriority_TempOCR": {
                    "recognition": {
                        "type": "OCR",
                        "param": {
                            "roi": [79, 515, 520, 51]
                        },
                    }
                }
            }
            
            # 运行动态生成的 OCR 任务
            ocr_result = context.run_recognition("HisRumorsPriority_TempOCR", img, override_config)
            
            if ocr_result is None:
                logger.error("HisRumorsPriority: 动态 OCR 运行失败，返回为 None。")
                return False

            # 兼容处理属性名称，防止拼写差异导致报错
            results = getattr(ocr_result, 'filtered_results', getattr(ocr_result, 'filterd_results', None))

            if not results:
                logger.warning("HisRumorsPriority: OCR 运行成功，但屏幕上未检测到任何文字。")
                return False

            # 不启用优先级时，直接点击最左侧选项
            if not priority_enabled:
                leftmost = min(results, key=lambda r: r.box[0])
                x, y, w, h = leftmost.box
                context.tasker.controller.post_click(x + w // 2, y + h // 2).wait()
                logger.info(f"未启用优先级，点击最左侧选项: '{leftmost.text}' at ({x}, {y})")
                return True

            # 4. 筛选并匹配选项
            visible_options = []
            for res in results:
                text = res.text.strip()
                if not text:
                    continue
                
                # 精确或模糊匹配优先级列表
                match_index = self.find_priority_index(text)
                if match_index != -1:
                    visible_options.append({
                        "text": text,
                        "box": res.box,
                        "priority": match_index
                    })
                    logger.info(f"模糊匹配成功: '{text}' -> 优先级 {match_index} ({self.priority_list[match_index]})")

            if not visible_options:
                # 优先级选项未在屏幕上，回退点击最左侧选项
                leftmost = min(results, key=lambda r: r.box[0])
                x, y, w, h = leftmost.box
                context.tasker.controller.post_click(x + w // 2, y + h // 2).wait()
                logger.info(f"优先级选项本轮未出现，回退点击最左侧选项: '{leftmost.text}'")
                return True

            # 5. 选择最优选项 (Index 越小优先级越高)
            visible_options.sort(key=lambda x: x["priority"])
            best_option = visible_options[0]

            logger.info(f"点击选项 '{best_option['text']}' (优先级 {best_option['priority']})")

            # 6. 点击选项
            x, y, w, h = best_option["box"]
            cx = x + w // 2
            cy = y + h // 2
            
            context.tasker.controller.post_click(cx, cy).wait()
            
            return True

        except Exception as e:
            logger.error(f"HisRumorsPriority 发生错误: {e}")
            return False

    def find_priority_index(self, text):
        """
        查找匹配文本的优先级索引。如果未找到则返回 -1。
        """
        # 1. 精确或子串匹配
        for idx, p_text in enumerate(self.priority_list):
            if p_text in text or text in p_text:
                return idx
        
        # 2. 模糊匹配
        for idx, p_text in enumerate(self.priority_list):
            ratio = difflib.SequenceMatcher(None, p_text, text).ratio()
            if ratio >= self.similarity_threshold:
                return idx
                
        return -1