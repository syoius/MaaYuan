from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
import difflib
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
        if argv.custom_action_param and isinstance(argv.custom_action_param, dict):
            new_list = [None] * 5
            found_param = False
            for k, v in argv.custom_action_param.items():
                if k.startswith("priority_"):
                    try:
                        idx = int(k.split("_")[1]) - 1
                        if 0 <= idx < 5:
                            new_list[idx] = v
                            found_param = True
                    except:
                        pass
            
            # If we found at least one priority param, update the list
            # Filter None values to keep the list compact, or keep them to maintain strict slots? 
            # Logic: If user sets P1=A, P2=B, we want [A, B].
            if found_param:
                # Fill missing slots with remaining default items or just ignore?
                # Better to just use what's provided.
                valid_new_list = [x for x in new_list if x is not None]
                if valid_new_list:
                    self.priority_list = valid_new_list
                    logger.info(f"Updated priority list from params: {self.priority_list}")

        logger.info("Starting HisRumorsPriority check...")
        
        try:
            # 1. Capture Screenshot
            img = context.tasker.controller.post_screencap().wait().get()
            if not img:
                logger.error("Failed to capture screenshot for HisRumorsPriority.")
                return False

            # 2. Run OCR to find text on screen
            # "OCR" is widely used as a default recognition task name in standard MAA setups
            ocr_result = context.run_recognition("OCR", img)
            
            if not ocr_result or not ocr_result.filterd_results:
                logger.warning("HisRumorsPriority: No text detected on screen.")
                return False

            # 3. Filter and Match Options
            visible_options = []
            for res in ocr_result.filterd_results:
                text = res.text.strip()
                if not text:
                    continue
                
                # Check directly or fuzzy match against priority list
                match_index = self.find_priority_index(text)
                if match_index != -1:
                    visible_options.append({
                        "text": text,
                        "box": res.box,
                        "priority": match_index
                    })
                    logger.info(f"Fuzzy matched option: '{text}' -> Priority {match_index} ({self.priority_list[match_index]})")

            if not visible_options:
                logger.info("HisRumorsPriority: No priority options found on screen.")
                return False

            # 4. Select the Best Option (Lowest index = Higher priority)
            # Sort by priority index
            visible_options.sort(key=lambda x: x["priority"])
            best_option = visible_options[0]

            logger.info(f"HisRumorsPriority: Clicking choice '{best_option['text']}' (Priority {best_option['priority']})")

            # 5. Click the Option
            x, y, w, h = best_option["box"]
            cx = x + w // 2
            cy = y + h // 2
            
            context.tasker.controller.post_click(cx, cy).wait()
            
            return True

        except Exception as e:
            logger.error(f"Error in HisRumorsPriority: {e}")
            return False

    def find_priority_index(self, text):
        """
        Finds the index of the matching priority text.
        Returns -1 if not found.
        """
        # 1. Exact or Substring Match
        for idx, p_text in enumerate(self.priority_list):
            if p_text in text or text in p_text:
                return idx
        
        # 2. Fuzzy Match
        for idx, p_text in enumerate(self.priority_list):
            ratio = difflib.SequenceMatcher(None, p_text, text).ratio()
            if ratio >= self.similarity_threshold:
                return idx
                
        return -1
