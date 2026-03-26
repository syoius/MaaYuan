import time
import json
import difflib
from zhconv import convert
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger

@AgentServer.custom_action("HisRumorsPriority")
class HisRumorsPriority(CustomAction):
    def __init__(self):
        super().__init__()
        self.priority_list = [
            "袁基",
            "左慈",
            "刘辩",
            "傅融",
            "孙策"
        ]
        # Similarity threshold for text matching (0.0 - 1.0)
        self.similarity_threshold = 0.6

    @staticmethod
    def normalize_text(text) -> str:
        """
        统一文本到简体中文，便于 OCR 繁简混用时匹配。
        """
        return convert(str(text or "").strip(), "zh-cn")

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        # Update priority list from parameters if provided
        raw_param = argv.custom_action_param
        try:
            param_dict = json.loads(raw_param) if isinstance(raw_param, str) else raw_param
        except (json.JSONDecodeError, TypeError):
            param_dict = {}

        # 检查是否启用优先级选择
        priority_enabled = str(param_dict.get("enabled", "true")).lower() != "false" if param_dict else True

        # 直接根据priority_X创建优先级字典 {priority值: 人物名}
        priority_map = {}
        if param_dict and isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k.startswith("priority_"):
                    try:
                        priority_num = int(k.split("_")[1])
                        if 1 <= priority_num <= 5:
                            priority_map[priority_num] = self.normalize_text(v)
                    except Exception:
                        pass
        
        # 把priority_map输出到日志，方便调试
        if priority_map:
            priority_info = ", ".join([f"priority_{p}:{name}" for p, name in sorted(priority_map.items())])
            logger.info(f"当前优先级配置: {priority_info}")

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
                raw_text = str(res.text or "").strip()
                if not raw_text:
                    continue
                text = self.normalize_text(raw_text)
                
                # 逐一检查priority_map中的人物，看OCR识别的文本是否匹配
                for priority_num, person_name in priority_map.items():
                    if self.text_match(text, person_name):
                        visible_options.append({
                            "raw_text": raw_text,
                            "text": text,
                            "box": res.box,
                            "priority": priority_num,  # 保存priority数字，而不是index
                            "name": person_name
                        })
                        logger.info(f"匹配成功: '{raw_text}' -> priority_{priority_num}({person_name})")
                        break  # 找到匹配后跳出循环，不需继续检查其他priority

            if not visible_options:
                # 优先级选项未在屏幕上，回退点击最左侧选项
                leftmost = min(results, key=lambda r: r.box[0])
                x, y, w, h = leftmost.box
                context.tasker.controller.post_click(x + w // 2, y + h // 2).wait()
                logger.info(f"优先级选项本轮未出现，回退点击最左侧选项: '{leftmost.text}'")
                return True

            # 5. 选择最优选项 (priority数字越小优先级越高，越先执行)
            visible_options.sort(key=lambda x: x["priority"])
            best_option = visible_options[0]

            logger.info(f"点击选项 '{best_option['text']}' (priority_{best_option['priority']})")

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
        text = self.normalize_text(text)
        # 1. 精确或子串匹配
        for idx, p_text in enumerate(self.priority_list):
            p_text = self.normalize_text(p_text)
            if p_text in text or text in p_text:
                return idx
        
        # 2. 模糊匹配
        for idx, p_text in enumerate(self.priority_list):
            p_text = self.normalize_text(p_text)
            ratio = difflib.SequenceMatcher(None, p_text, text).ratio()
            if ratio >= self.similarity_threshold:
                return idx
                
        return -1

    def text_match(self, ocr_text, target_name):
        """
        检查OCR识别的文本是否与目标人物名匹配。
        支持精确互为子字符串、模糊匹配。
        """
        target_name = self.normalize_text(target_name)
        # 1. 精确或子串匹配
        if target_name in ocr_text or ocr_text in target_name:
            return True
        
        # 2. 模糊匹配
        ratio = difflib.SequenceMatcher(None, target_name, ocr_text).ratio()
        if ratio >= self.similarity_threshold:
            return True
        
        return False
