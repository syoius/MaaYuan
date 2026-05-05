"""
BBQv2 Custom Action - 基于 Kotlin BbqPlanner 的 token 调度烧烤逻辑。

参照 bbq_kotlin 中 BbqPlanner + BbqRuntimeManager + BbqActionPlanFactory 的设计，
实现 token 状态机调度：扫描气泡 → 决策 → 执行 → 等待 → 循环。

pipeline 中通过 custom_action_param 传入 5 个位置的食材及其烧烤时长，可在 pipeline 中编辑。
"""

import json
import time
import threading
import queue

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

# ============================================================
# 常量
# ============================================================

# 点击/滑动后的调度延迟 (ms)，参照 Kotlin GESTURE_DELAY_MS = 200
GESTURE_DELAY_S = 0.200
# 交付滑动时长 (ms)，参照 Kotlin DELIVERY_SWIPE_DURATION_MS = 260
DELIVERY_SWIPE_DURATION_MS = 260
# 倒水长按时长 (ms)，参照 Kotlin 2000ms
POUR_DURATION_MS = 2000
# 动作间最小间隔 (s)
ACTION_INTERVAL_S = 0.15
# 审查间隔 (s)
REVIEW_INTERVAL_S = 0.5
# 交付后延迟 (s)，等待清理按钮出现
DELIVERY_SETTLE_S = 0.3
# 会话最大时长 (s)
MAX_SESSION_DURATION_S = 600

# 烤架 ROI [x, y, w, h]
GRILL_1_ROI = [9, 838, 286, 179]
GRILL_2_ROI = [303, 836, 286, 179]

# 烤架槽位 ROI — 交付时的滑动起点 (取 ROI 中心)
GRILL_SLOTS_ROI = {
    "1-1": [9, 838, 95, 179],
    "1-2": [104, 838, 95, 179],
    "1-3": [199, 838, 95, 179],
    "2-1": [321, 839, 95, 179],
    "2-2": [398, 836, 95, 179],
    "2-3": [468, 836, 96, 180],
}

# 客人餐盘 ROI [x, y, w, h] (交付/清理)
PLATE_ROI = {1: [137, 760, 6, 6], 2: [359, 763, 6, 6], 3: [585, 775, 8, 7]}

# 客人气泡 ROI [x, y, w, h] (扫描订单)
BUBBLE_ROI = {1: [68, 469, 202, 119], 2: [261, 470, 231, 126], 3: [499, 468, 201, 132]}

# 清理按钮 ROI [x, y, w, h]
CLEAN_ROI = {1: [134, 786, 9, 6], 2: [360, 787, 9, 6], 3: [586, 787, 9, 6]}
# 计时清理点击 ROI [x, y, w, h] — 交付完成后主动点击清理
CLEANUP_CLICK_ROI = {1: [127, 780, 18, 8], 2: [354, 781, 19, 7], 3: [580, 781, 18, 7]}

# 食材栏固定位置 ROI [x, y, w, h] — 位置1~5是固定的，食物可任意分配
POSITION_CLICK_ROI = {
    1: [206, 1139, 8, 15],
    2: [296, 1143, 9, 16],
    3: [388, 1145, 9, 16],
    4: [478, 1146, 9, 16],
    5: [570, 1145, 9, 16],
}

# 饮料名称 → pipeline 节点名称映射（默认值，运行时从参数动态生成）
DRINK_SELECT_NODE = {
    "广陵山泉": "BBQv2_选择广陵山泉",
    "红蓼桃桃": "BBQv2_选择红蓼桃桃",
    "荼蘼芒芒": "BBQv2_选择荼蘼芒芒",
    "碧竹青青": "BBQv2_选择碧竹青青",
    "朱栾幽幽": "BBQv2_选择朱栾幽幽",
    "瑞杏": "BBQv2_选择瑞杏",
    "冷酷果汁": "BBQv2_选择冷酷果汁",
    "白梅芝芝": "BBQv2_选择白梅芝芝",
    "瞒呐": "BBQv2_选择瞒呐",
    "兴霸客": "BBQv2_选择兴霸客",
    "祢雪冰衡": "BBQv2_选择祢雪冰衡",
    "戏雪的茶": "BBQv2_选择戏雪的茶",
    "特调·巫血": "BBQv2_选择特调·巫血",
    "孔夫子特调": "BBQv2_选择孔夫子特调",
    "特调草划": "BBQv2_选择特调草划",
    "西凉酥山": "BBQv2_选择西凉酥山",
}
DRINK_PRODUCT_NODE = {
    "广陵山泉": "BBQv2_成品广陵山泉",
    "红蓼桃桃": "BBQv2_成品红蓼桃桃",
    "荼蘼芒芒": "BBQv2_成品荼蘼芒芒",
    "碧竹青青": "BBQv2_成品碧竹青青",
    "朱栾幽幽": "BBQv2_成品朱栾幽幽",
    "瑞杏": "BBQv2_成品瑞杏",
    "冷酷果汁": "BBQv2_成品冷酷果汁",
    "白梅芝芝": "BBQv2_成品白梅芝芝",
    "瞒呐": "BBQv2_成品瞒呐",
    "兴霸客": "BBQv2_成品兴霸客",
    "祢雪冰衡": "BBQv2_成品祢雪冰衡",
    "戏雪的茶": "BBQv2_成品戏雪的茶",
    "特调·巫血": "BBQv2_成品特调·巫血",
    "孔夫子特调": "BBQv2_成品孔夫子特调",
    "特调草划": "BBQv2_成品特调草划",
    "西凉酥山": "BBQv2_成品西凉酥山",
}

# 有 pipeline 模板节点的食物/饮料列表（默认值，运行时从参数动态生成）
FOOD_TEMPLATES = [
    "肉丸",
    "牛骨髓",
    "糍粑",
    "豆角",
    "海鲜",
    "巫彭",
    "奶豆腐",
    "白薯",
    "菌子",
    "羊肉串",
    "鱼",
    "卷饼",
    "葱",
    "梭子蟹",
    "肉排",
    "广陵山泉",
    "冷酷果汁",
    "红蓼桃桃",
    "白梅芝芝",
    "碧竹青青",
    "朱栾幽幽",
    "荼蘼芒芒",
    "瑞杏",
    "瞒呐",
    "兴霸客",
    "祢雪冰衡",
    "戏雪的茶",
    "特调·巫血",
    "孔夫子特调",
    "特调草划",
    "西凉酥山",
]

# 饮料位置编号到默认名称的映射
DRINK_POSITION_NAMES = {1: "广陵山泉", 2: "红蓼桃桃"}

# 食材位置编号到默认名称的映射 (BBQv2 食材)
POSITION_NAMES = {1: "肉丸", 2: "牛骨髓", 3: "糍粑", 4: "豆角", 5: "海鲜"}

# 状态机
S_WAITING = "WAITING"
S_COOKING = "COOKING"
S_COOKED = "COOKED"
S_DRINK_READY = "DRINK_READY"
S_PENDING_CONFIRM = "PENDING_CONFIRM"
S_DELIVERED = "DELIVERED"

# 交付后等待气泡消失的宽限期 (秒)
DEMAND_RECHECK_GRACE_S = 0.4
# PENDING_CONFIRM 超时 (秒)，超时强制清除
PENDING_CONFIRM_TIMEOUT_S = 5.0

# 默认烧烤时长 (秒) — 可在 pipeline 中覆盖
# 参照 Kotlin: 排骨 6.25, 韭菜 2, 肉丸 4, 年糕 3, 虾 5
# BBQv2 食材用相近时长作为默认值
DEFAULT_COOK_DURATIONS = {
    "肉丸": 4.0,
    "牛骨髓": 6.25,
    "糍粑": 3.0,
    "豆角": 2.0,
    "海鲜": 5.0,
    "巫彭": 5.9,
    "奶豆腐": 3.0,
    "白薯": 3.0,
    "菌子": 3.0,
    "羊肉串": 5.0,
    "鱼": 4.0,
    "卷饼": 5.0,
    "葱": 2.0,
    "梭子蟹": 6.25,
    "肉排": 6.25,
}

# ============================================================
# 工具函数
# ============================================================


def _should_stop(context: Context) -> bool:
    """协作式停止检查。"""
    try:
        if bool(getattr(context, "stop", False)):
            # logger.info("BBQ_should_stop: context.stop=True")
            return True
        tasker = getattr(context, "tasker", None)
        if tasker is not None:
            if bool(getattr(tasker, "stopping", False)):
                # logger.info("BBQ_should_stop: tasker.stopping=True")
                return True
            if not tasker.running:
                # logger.info("BBQ_should_stop: tasker.running=False")
                return True
    except Exception as e:
        logger.warning(f"BBQ_should_stop异常: {e}")
        return False
    return False


def _screencap(context: Context):
    """截取当前屏幕，返回 numpy 数组 (BGR)。"""
    try:
        return context.tasker.controller.post_screencap().wait().get()
    except Exception:
        logger.exception("BBQ截图失败")
        return None


def _roi_center(roi):
    """ROI [x, y, w, h] 的中心坐标。"""
    return (int(roi[0] + roi[2] / 2), int(roi[1] + roi[3] / 2))


def _box_to_tuple(box):
    """将 MaaFW Rect 转换为 (x, y, w, h) tuple。"""
    if box is None:
        return None
    if isinstance(box, (tuple, list)):
        return tuple(box)
    return (box.x, box.y, box.w, box.h)


# ============================================================
# BBQ Token 数据结构 (参照 Kotlin BbqPlanner)
# ============================================================


class BbqToken:
    __slots__ = (
        "id",
        "guest_idx",
        "food_name",
        "state",
        "grill_slot",
        "started_at",
        "ready_at",
        "delivered_at",
    )

    def __init__(self, token_id, guest_idx, food_name):
        self.id = token_id
        self.guest_idx = guest_idx
        self.food_name = food_name
        self.state = S_WAITING
        self.grill_slot = None
        self.started_at = None
        self.ready_at = None
        self.delivered_at = None


# ============================================================
# BBQv2Custom Action
# ============================================================


@AgentServer.custom_action("BBQv2Custom")
class BBQv2Custom(CustomAction):
    """
    基于 token 调度的烧烤自定义动作。

    参数格式 (custom_action_param):
    {
        "position_1": "肉丸",     // 位置1的食材名称 (对应烧烤架第1个位置)
        "position_2": "牛骨髓",     // 位置2
        "position_3": "糍粑",     // 位置3
        "position_4": "豆角",     // 位置4
        "position_5": "海鲜",     // 位置5
        "cook_time_肉丸": 4.0,    // 肉丸烧烤时长(秒)
        "cook_time_牛骨髓": 6.25,   // 牛骨髓烧烤时长(秒)
        "cook_time_糍粑": 3.0,    // 糍粑烧烤时长(秒)
        "cook_time_豆角": 2.0,    // 豆角烧烤时长(秒)
        "cook_time_海鲜": 5.0,    // 海鲜烧烤时长(秒)
        "enable_second_grill": false,  // 是否启用第二个烧烤架
        "max_duration": 600       // 最大会话时长(秒)
    }
    """

    def __init__(self):
        super().__init__()
        self.tokens = []
        self.next_token_id = 1
        self.grill_state = {}
        self.drink_slot = None
        self.drink_machine_busy = False
        self.drink_select_delay = 0.3  # 默认300ms，run中会被pipeline参数覆盖
        self.drink_positions = {}  # drink_name → position_number
        self.customer_orders = {}
        self.customer_cleanup = {}
        self.session_start = 0
        self.food_positions = {}
        self.cook_durations = {}
        self._state_lock = threading.Lock()
        self._delivery_queue = queue.Queue()
        self._cleanup_queue = queue.Queue()
        self._cook_timers = {}
        self._cleanup_timers = {}

    def _reset_runtime_config(self):
        """清空上一轮 run 解析出的运行时配置。"""
        self.food_positions = {}
        self.drink_positions = {}
        self.cook_durations = {}

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        self._reset_runtime_config()

        # 优先从 argv 读取 custom_action_param
        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except json.JSONDecodeError:
                logger.warning(f"【魂生又一串】自助烤串: 无法解析 argv 参数")

        # 从节点数据读取 base defaults 和 attach（interface override）
        base_params = {}
        attach_params = {}
        try:
            node_data = context.get_node_data("BBQv2Custom启动")
            if node_data:
                base_params = (
                    node_data.get("action", {})
                    .get("param", {})
                    .get("custom_action_param", {})
                )
                attach_params = node_data.get("attach", {})
                # if attach_params:
                #     logger.info(f"BBQv2Custom: attach参数={attach_params}")
        except Exception as e:
            logger.warning(f"【魂生又一串】自助烤串: 读取节点数据失败: {e}")

        # 合并优先级: base defaults < argv params < attach (interface override)
        merged = dict(base_params)
        merged.update(params)
        merged.update(attach_params)
        params = merged

        # 解析食材位置配置 — food_name → position_number
        for pos in range(1, 6):
            food_name = params.get(f"position_{pos}")
            if food_name:
                self.food_positions[food_name] = pos

        # 解析烧烤时长
        for food_name in self.food_positions:
            key = f"cook_time_{food_name}"
            raw = params.get(key, DEFAULT_COOK_DURATIONS.get(food_name, 4.0))
            self.cook_durations[food_name] = float(raw)

        # 注册饮料名称 — drink_name → position_number
        for drink_pos in range(1, 3):
            drink_name = params.get(f"drink_position_{drink_pos}")
            if drink_name:
                self.drink_positions[drink_name] = drink_pos

        enable_second_grill = params.get("enable_second_grill", False)
        max_duration = params.get("max_duration", MAX_SESSION_DURATION_S)
        self.drink_select_delay = params.get("drink_select_delay", 300) / 1000.0
        self.delivery_settle = params.get("delivery_settle", 300) / 1000.0
        self.pending_timeout = params.get("pending_timeout", 5000) / 1000.0
        self.cleanup_delay = params.get("cleanup_delay", 1000) / 1000.0

        # logger.info(f"BBQv2Custom 启动: 原始参数={params}")
        # logger.info(f"BBQv2Custom 启动: 食材={self.food_positions}, 饮料={self.drink_positions}, 时长={self.cook_durations}")

        # 初始化会话
        self._init_session(enable_second_grill)
        self.session_start = time.time()

        # 主循环
        try:
            self._main_loop(context, max_duration)
        except Exception as e:
            logger.exception(f"【魂生又一串】自助烤串 异常: {e}")
            return CustomAction.RunResult(success=False)
        finally:
            self._cancel_all_timers()

        logger.info("【魂生又一串】自助烤串结束")
        return CustomAction.RunResult(success=True)

    def _init_session(self, enable_second_grill):
        """初始化会话状态。"""
        self.tokens = []
        self.next_token_id = 1
        self.grill_state = {}
        self.drink_slot = None
        self.drink_machine_busy = False
        self.customer_orders = {1: [], 2: [], 3: []}
        self.customer_cleanup = {1: False, 2: False, 3: False}
        self._cancel_all_timers()

        # 清空交付队列和清理队列
        for q in (self._delivery_queue, self._cleanup_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        slots = ["1-1", "1-2", "1-3"]
        if enable_second_grill:
            slots += ["2-1", "2-2", "2-3"]
        for slot_id in slots:
            self.grill_state[slot_id] = None

    # ----------------------------------------------------------
    # 计时器 + 交付队列
    # ----------------------------------------------------------

    def _schedule_cook_timer(self, slot_id: str, token):
        """为烤架上的食物启动独立计时器，到期后入队等待交付。"""
        cook_time = self.cook_durations.get(token.food_name, 4.0)
        guest_idx = token.guest_idx
        food_name = token.food_name
        token_id = token.id

        def _on_timer():
            # logger.info(f"BBQ: 计时器到期 {slot_id} {food_name}")
            self._delivery_queue.put((slot_id, guest_idx, food_name, token_id))

        timer = threading.Timer(cook_time, _on_timer)
        timer.daemon = True
        timer.start()
        self._cook_timers[slot_id] = timer
        # logger.info(f"BBQ: 启动计时器 {slot_id} {food_name} {cook_time}s")

    def _cancel_all_timers(self):
        """取消所有未到期的计时器。"""
        for _, timer in self._cook_timers.items():
            timer.cancel()
        self._cook_timers.clear()
        for _, timer in self._cleanup_timers.items():
            timer.cancel()
        self._cleanup_timers.clear()

    def _process_delivery_queue(self, context: Context):
        """处理交付队列，逐个 swipe。每完成一个 swipe 后优先处理清理。"""
        checked_cleanup = False
        while not self._delivery_queue.empty():
            # 清理优先：每次 swipe 前先处理待执行的清理
            self._process_cleanup_queue(context)

            try:
                slot_id, guest_idx, food_name, token_id = (
                    self._delivery_queue.get_nowait()
                )
            except queue.Empty:
                break

            # 检查槽位是否仍有效
            slot_data = self.grill_state.get(slot_id)
            if not slot_data or slot_data.get("token_id") != token_id:
                logger.info(f"【魂生又一串】跳过过期交付 {slot_id} {food_name}")
                continue

            # 执行滑动交付
            grill_roi = GRILL_SLOTS_ROI.get(slot_id, GRILL_1_ROI)
            start_x, start_y = _roi_center(grill_roi)
            end_x, end_y = _roi_center(PLATE_ROI[guest_idx])

            logger.info(f"【魂生又一串】交付 {food_name} {slot_id}→客人{guest_idx}")
            context.tasker.controller.post_swipe(
                start_x, start_y, end_x, end_y, DELIVERY_SWIPE_DURATION_MS
            ).wait()
            # 只有队列里还有下一个才等，最后一个不用等
            if not self._delivery_queue.empty():
                time.sleep(self.delivery_settle)

            # 更新状态
            with self._state_lock:
                token = self._get_token(token_id)
                if token:
                    token.state = S_PENDING_CONFIRM
                    token.delivered_at = time.time()
                self.grill_state[slot_id] = None
                self._cook_timers.pop(slot_id, None)

            # 首次交付后截图检测一次清理（只截一次、识别一次）
            if not checked_cleanup:
                checked_cleanup = True
                img = _screencap(context)
                if img is not None:
                    self._check_cleanup(context, img)

            # 该客人全部交付完成 → 启动清理计时器
            if self._is_guest_done(guest_idx):
                self._schedule_cleanup(guest_idx)

        # 队列清空后再处理一轮清理（最后一个 swipe 后可能有新清理入队）
        self._process_cleanup_queue(context)

    def _is_guest_done(self, guest_idx: int) -> bool:
        """客人是否所有订单都已交付（无 WAITING/COOKING/COOKED/DRINK_READY token，且队列中无该客人的待交付项）。"""
        for t in self.tokens:
            if t.guest_idx == guest_idx and t.state in (
                S_WAITING,
                S_COOKING,
                S_COOKED,
                S_DRINK_READY,
            ):
                return False
        # 检查交付队列中是否还有该客人的项目
        for item in list(self._delivery_queue.queue):
            if item[1] == guest_idx:  # (slot_id, guest_idx, food_name, token_id)
                return False
        return True

    def _schedule_cleanup(self, guest_idx: int):
        """客人全部交付完成后，延迟 cleanup_delay 秒入队等待清理。"""

        def _on_cleanup():
            # logger.info(f"BBQ: 清理计时到期 客人{guest_idx}")
            self._cleanup_queue.put(guest_idx)

        timer = threading.Timer(self.cleanup_delay, _on_cleanup)
        timer.daemon = True
        timer.start()
        self._cleanup_timers[guest_idx] = timer
        # logger.info(f"BBQ: 启动清理计时 客人{guest_idx} {self.cleanup_delay}s")

    def _process_cleanup_queue(self, context: Context):
        """处理清理队列：直接点击盘子位置清理。"""
        while not self._cleanup_queue.empty():
            try:
                guest_idx = self._cleanup_queue.get_nowait()
            except queue.Empty:
                break
            cleanup_roi = CLEANUP_CLICK_ROI[guest_idx]
            cx, cy = _roi_center(cleanup_roi)
            # logger.info(f"BBQ: 计时清理 客人{guest_idx} ({cx},{cy})")
            context.tasker.controller.post_click(cx, cy).wait()
            # 清理该客人的 token
            with self._state_lock:
                self.customer_orders[guest_idx] = []
                self.customer_cleanup[guest_idx] = False
                self.tokens = [
                    t
                    for t in self.tokens
                    if not (
                        t.guest_idx == guest_idx
                        and t.state
                        in (S_DELIVERED, S_PENDING_CONFIRM, S_COOKED, S_DRINK_READY)
                    )
                ]
            self._cleanup_timers.pop(guest_idx, None)

    def _check_cleanup(self, context: Context, img):
        """用 MaaFW 识别清理按钮并执行清理。"""
        result = context.run_recognition("BBQv2_清理_模板", img)
        if not result or not getattr(result, "hit", False):
            return

        box = _box_to_tuple(result.box)
        if not box:
            return
        bx = box[0]

        # 根据 x 坐标判断是哪个客人的清理按钮
        if bx < 200:
            guest_idx = 1
        elif bx < 420:
            guest_idx = 2
        else:
            guest_idx = 3

        logger.info(f"【魂生又一串】清理客人{guest_idx}餐盘")
        cx, cy = _roi_center(box)
        context.tasker.controller.post_click(cx, cy).wait()
        time.sleep(GESTURE_DELAY_S)

        with self._state_lock:
            self.customer_orders[guest_idx] = []
            self.customer_cleanup[guest_idx] = False
            self.tokens = [
                t
                for t in self.tokens
                if not (
                    t.guest_idx == guest_idx
                    and t.state
                    in (S_DELIVERED, S_PENDING_CONFIRM, S_COOKED, S_DRINK_READY)
                )
            ]
            # 只清理没有活跃 COOKING token 的烤架槽位
            for slot_id, slot_data in self.grill_state.items():
                if slot_data and slot_data.get("guest_idx") == guest_idx:
                    token_id = slot_data.get("token_id")
                    token = self._get_token(token_id) if token_id else None
                    if token and token.state == S_COOKING:
                        continue  # 保留正在烧烤的槽位
                    self.grill_state[slot_id] = None
                    timer = self._cook_timers.pop(slot_id, None)
                    if timer:
                        timer.cancel()

    def _detect_food_count(self, context, img, node_name, bubble_roi):
        """检测气泡中某种食物的数量。先匹配整个气泡，用匹配到的 box 排除已识别区域，再找下一个。"""
        x, y, w, h = bubble_roi
        current_roi = [x, y, w, h]
        count = 0
        for _ in range(4):  # 最多检测4个（4个相同食物）
            override = {node_name: {"recognition": {"param": {"roi": current_roi}}}}
            result = context.run_recognition(node_name, img, override)
            if not result or not getattr(result, "hit", False):
                break
            count += 1
            box = result.box
            # 排除已匹配区域，从 box 右边缘向左留5px余量，避免截断
            new_x = box.x + box.w - 5
            bubble_right = x + w
            remaining_w = bubble_right - new_x
            if remaining_w < 20:  # 剩余区域太小，不可能再有图标
                break
            # 新 ROI 终止于大气泡边缘，h 保持不变
            current_roi = [new_x, y, remaining_w, h]
        return count

    def _scan_orders(self, context: Context, img):
        """用 MaaFW 框架识别扫描所有客人气泡，支持检测重复食物图标。"""
        for guest in range(1, 4):
            bubble_roi = BUBBLE_ROI[guest]
            demands = []
            for food_name in FOOD_TEMPLATES:
                if (
                    food_name not in self.food_positions
                    and food_name not in self.drink_positions
                ):
                    continue
                node_name = f"BBQv2_{food_name}_模板"
                count = self._detect_food_count(context, img, node_name, bubble_roi)
                for _ in range(count):
                    demands.append(food_name)
            self.customer_orders[guest] = demands
        logger.info(f"【魂生又一串】当前点单情况 {self.customer_orders}")

    def _process_drink_orders(self, context: Context):
        """查找等待中的饮料订单并执行制作+交付。一次只处理一杯。"""
        with self._state_lock:
            drink_token = None
            for token in self.tokens:
                if token.state == S_WAITING and token.food_name in self.drink_positions:
                    drink_token = token
                    break

        if drink_token is None:
            return

        self.drink_machine_busy = True
        try:
            success = self._do_drink_order(
                context, drink_token.food_name, drink_token.guest_idx, drink_token
            )
            if not success:
                # 制作失败，token 保持 WAITING，下轮重试
                logger.info(
                    f"【魂生又一串】饮料 {drink_token.food_name} 制作失败，下轮重试"
                )
        finally:
            self.drink_machine_busy = False

    def _check_stop_conditions(self, context: Context, img) -> bool:
        """检测停止条件：确定按钮 或 时间归零。"""
        # 检测"确定"按钮
        result = context.run_recognition("BBQv2_确定按钮", img)
        if result and getattr(result, "hit", False):
            logger.info("【魂生又一串】检测到确定按钮，点击并停止")
            context.run_action("BBQv2_确定按钮", result.box, "", {})
            return True
        # 检测"00:00"时间归零
        result2 = context.run_recognition("BBQv2_时间归零", img)
        if result2 and getattr(result2, "hit", False):
            logger.info("【魂生又一串】检测到时间归零，停止")
            return True
        return False

    def _main_loop(self, context: Context, max_duration: float):
        """主循环：交单/清理最高优先级，饮料只在无待交食物时处理。"""
        while not _should_stop(context):
            elapsed = time.time() - self.session_start
            if elapsed > max_duration:
                logger.info(f"【魂生又一串】超时 ({max_duration}s)")
                break

            # 检测停止条件
            img = _screencap(context)
            if img is None:
                time.sleep(0.3)
                continue
            if self._check_stop_conditions(context, img):
                break

            # 1. 交单最高优先级：处理计时器到期的食物
            self._process_delivery_queue(context)

            # 1b. 处理清理队列（客人全部交付后计时到期，只在无活跃交付时执行）
            if self._delivery_queue.empty():
                self._process_cleanup_queue(context)

            # 2. 清理（用主循环截图识别一次即可，交付队列内部已处理过首次清理）
            self._check_cleanup(context, img)

            # 3. 烧焦食物
            self._check_burnt_food(context, img)

            # 4. 扫描订单 → 对账 → 放食物上架 → 启动计时器
            self._scan_orders(context, img)
            with self._state_lock:
                self._reconcile_tokens()
                self._cleanup_delivered_tokens()
                token_summary = [
                    (t.id, t.guest_idx, t.food_name, t.state) for t in self.tokens
                ]
                # logger.info(f"BBQ tokens: {token_summary}")
                self._place_waiting_food(context)

            # 5. 饮料：只在交单队列空 + 无待上架食物时才处理
            if (
                not self.drink_machine_busy
                and self._delivery_queue.empty()
                and not self._any_waiting_food()
            ):
                self._process_drink_orders(context)

            time.sleep(ACTION_INTERVAL_S)

    # ----------------------------------------------------------
    # 视觉扫描 (参照 Kotlin BbqTemplateScanner)
    # ----------------------------------------------------------

    def _reconcile_tokens(self):
        """参照 Kotlin reconcileTokensWithVision：计数对账，防止重复创建 token。"""
        # 收集所有 (guest, food_name) 对 — 来自视觉扫描 + 已有 token
        pair_keys = set()
        for guest, orders in self.customer_orders.items():
            for food_name in orders:
                pair_keys.add((guest, food_name))
        for token in self.tokens:
            pair_keys.add((token.guest_idx, token.food_name))

        for guest_idx, food_name in pair_keys:
            observed_count = self.customer_orders.get(guest_idx, []).count(food_name)

            # 不含 DELIVERED 的 token
            tokens_for_pair = [
                t
                for t in self.tokens
                if t.guest_idx == guest_idx
                and t.food_name == food_name
                and t.state != S_DELIVERED
            ]

            # guaranteed: 正在处理中，气泡还在是合理的
            guaranteed_count = sum(
                1
                for t in tokens_for_pair
                if t.state in (S_WAITING, S_COOKING, S_COOKED, S_DRINK_READY)
            )

            # pending: 已交付但气泡可能还没消失
            pending_tokens = sorted(
                [t for t in tokens_for_pair if t.state == S_PENDING_CONFIRM],
                key=lambda t: t.delivered_at or 0,
            )

            # 多余的 pending → 确认为 DELIVERED
            pending_needed = max(observed_count - guaranteed_count, 0)
            for token in pending_tokens[pending_needed:]:
                token.state = S_DELIVERED
                token.grill_slot = None

            # 不足 → 创建新 token
            unresolved = guaranteed_count + min(len(pending_tokens), pending_needed)
            shortfall = observed_count - unresolved
            for _ in range(max(shortfall, 0)):
                token = self._create_token(guest_idx, food_name)
                # logger.info(f"BBQ: 新建 token {token.id} 客人{guest_idx} {food_name}")

    def _cleanup_delivered_tokens(self):
        """移除已确认的 DELIVERED token。PENDING_CONFIRM 超时也强制清除。"""
        now = time.time()
        # PENDING_CONFIRM 超时 → 强制 DELIVERED
        for token in self.tokens:
            if token.state == S_PENDING_CONFIRM and token.delivered_at:
                if now - token.delivered_at >= self.pending_timeout:
                    # logger.info(
                    #     f"BBQ: token {token.id} {token.food_name} PENDING超时，强制清除"
                    # )
                    token.state = S_DELIVERED
                    token.grill_slot = None
        # 移除 DELIVERED token
        self.tokens = [
            t
            for t in self.tokens
            if not (t.state == S_DELIVERED and self._should_drop_delivered(t, now))
        ]

    def _should_drop_delivered(self, token, now):
        """判断 DELIVERED token 是否应该移除。"""
        # 客人清理中 → 移除
        if self.customer_cleanup.get(token.guest_idx, False):
            return True
        # 气泡已消失 → 移除
        if token.food_name not in self.customer_orders.get(token.guest_idx, []):
            return True
        # 超过宽限期 → 移除
        if token.delivered_at and now - token.delivered_at >= DEMAND_RECHECK_GRACE_S:
            return True
        return False

    def _any_waiting_food(self) -> bool:
        """是否有等待上架的食物 token。"""
        return any(
            t.state == S_WAITING and t.food_name in self.cook_durations
            for t in self.tokens
        )

    def _find_token_for(self, guest_idx, food_name):
        """找到客人对应食物的等待中 token。"""
        for token in self.tokens:
            if (
                token.guest_idx == guest_idx
                and token.food_name == food_name
                and token.state == S_WAITING
            ):
                return token
        return None

    def _any_guest_needs(self, food_name, exclude_token_id=None):
        """是否有客人需要指定食物。"""
        for guest, orders in self.customer_orders.items():
            if food_name in orders:
                # 检查是否已有其他 token 在处理 (排除指定 token)
                has_token = any(
                    t.guest_idx == guest
                    and t.food_name == food_name
                    and t.state in (S_WAITING, S_COOKING, S_COOKED, S_DRINK_READY)
                    and t.id != exclude_token_id
                    for t in self.tokens
                )
                if not has_token:
                    return True
        return False

    def _get_token(self, token_id):
        """根据 ID 获取 token。"""
        for t in self.tokens:
            if t.id == token_id:
                return t
        return None

    def _create_token(self, guest_idx, food_name):
        """创建新 token，参照 Kotlin reconcileTokensWithVision()。"""
        token_id = f"token-{self.next_token_id}"
        self.next_token_id += 1
        token = BbqToken(token_id, guest_idx, food_name)
        self.tokens.append(token)
        return token

    # ----------------------------------------------------------
    # 动作执行 (参照 Kotlin BbqActionPlanFactory)
    # ----------------------------------------------------------

    def _cleanup_grill_slot(self, context: Context, slot_id: str):
        """无脑清理烤架槽位 — 从槽位中心滑到垃圾桶。"""
        slot_roi = GRILL_SLOTS_ROI.get(slot_id)
        if not slot_roi:
            return
        sx, sy = _roi_center(slot_roi)
        trash_x, trash_y = 47, 1175
        logger.info(f"【魂生又一串】清理槽位{slot_id}")
        context.tasker.controller.post_swipe(sx, sy, trash_x, trash_y, 150).wait()
        time.sleep(GESTURE_DELAY_S)

    def _place_waiting_food(self, context: Context):
        """找空闲槽位 + 等待中的食物，上架并启动计时器。需在 _state_lock 内调用。"""
        for slot_id, slot_data in self.grill_state.items():
            if slot_data is not None:
                continue
            # 按 token ID 排序，确保先创建的先上架
            for token in sorted(self.tokens, key=lambda t: t.id):
                if (
                    token.state != S_WAITING
                    or token.food_name not in self.cook_durations
                ):
                    continue
                # reconciler 已保证 token 数量 = 需求量，无需再检查是否有客人需要
                # 上架前先清理遗留食物
                self._cleanup_grill_slot(context, slot_id)
                # 上架
                food_name = token.food_name
                pos = self.food_positions.get(food_name)
                if pos is None or pos not in POSITION_CLICK_ROI:
                    continue
                cx, cy = _roi_center(POSITION_CLICK_ROI[pos])
                logger.info(f"【魂生又一串】上架 {food_name} -> 烤架{slot_id}")
                context.tasker.controller.post_click(cx, cy).wait()
                time.sleep(GESTURE_DELAY_S)

                now = time.time()
                cook_time = self.cook_durations.get(food_name, 4.0)
                token.state = S_COOKING
                token.grill_slot = slot_id
                token.started_at = now
                token.ready_at = now + cook_time
                self.grill_state[slot_id] = {
                    "food": food_name,
                    "token_id": token.id,
                    "guest_idx": token.guest_idx,
                    "state": S_COOKING,
                    "started_at": now,
                    "ready_at": now + cook_time,
                }
                self._schedule_cook_timer(slot_id, token)
                break  # 一轮只放一个食物

    def _do_drink_order(
        self, context: Context, drink_name: str, guest_idx: int, token: BbqToken
    ) -> bool:
        """饮料制作+交付。一气呵成，不被其他逻辑打断。

        流程: 检查遗留成品 → 点饮水机 → 点饮料 → 长按倒水2s → 上菜
        """
        logger.info(f"【魂生又一串】开始制作饮料 {drink_name} 给客人{guest_idx}")

        # 步骤 0: 检查遗留成品（一次识别，有就处理，没有就继续）
        img0 = _screencap(context)
        if img0 is not None:
            trash_x, trash_y = 47, 1175
            for d_name, product_node in DRINK_PRODUCT_NODE.items():
                p_result = context.run_recognition(product_node, img0)
                if p_result and getattr(p_result, "hit", False):
                    if d_name == drink_name:
                        px, py = _roi_center(_box_to_tuple(p_result.box))
                        end_x, end_y = _roi_center(PLATE_ROI[guest_idx])
                        logger.info(
                            f"【魂生又一串】发现成品{d_name}，直接交付 客人{guest_idx}"
                        )
                        context.tasker.controller.post_swipe(
                            px, py, end_x, end_y, 150
                        ).wait()
                        time.sleep(self.delivery_settle)
                        token.state = S_PENDING_CONFIRM
                        token.delivered_at = time.time()
                        if self._is_guest_done(guest_idx):
                            self._schedule_cleanup(guest_idx)
                        return True
                    else:
                        px, py = _roi_center(_box_to_tuple(p_result.box))
                        logger.info(f"【魂生又一串】清理遗留成品{d_name}")
                        context.tasker.controller.post_swipe(
                            px, py, trash_x, trash_y, 150
                        ).wait()
                        time.sleep(GESTURE_DELAY_S)
                    break

        # 步骤 1: 点击饮水机
        img = _screencap(context)
        if img is None:
            return False
        result = context.run_recognition("BBQv2_饮水机可用", img)
        if not result or not getattr(result, "hit", False):
            logger.info("【魂生又一串】饮水机不可用，跳过")
            return False
        context.run_action("BBQv2_饮水机可用", result.box, "", {})
        time.sleep(0.3)

        # 步骤 2: 点击饮料种类
        select_node = DRINK_SELECT_NODE.get(drink_name)
        if not select_node:
            logger.warning(f"【魂生又一串】未知饮料 {drink_name}")
            return False
        img2 = _screencap(context)
        if img2 is not None:
            select_result = context.run_recognition(select_node, img2)
            if select_result and getattr(select_result, "hit", False):
                context.run_action(select_node, select_result.box, "", {})
                time.sleep(self.drink_select_delay)
            else:
                logger.info(f"【魂生又一串】未找到饮料 {drink_name} 选项")
                return False

        # 步骤 3: 长按倒水 (2000ms，阻塞到完成)
        img3 = _screencap(context)
        if img3 is not None:
            pour_result = context.run_recognition("BBQv2_长按倒水", img3)
            if pour_result and getattr(pour_result, "hit", False):
                # logger.info("【魂生又一串】长按倒水")
                context.run_action("BBQv2_长按倒水", pour_result.box, "", {})
            else:
                logger.info("【魂生又一串】倒水按钮未找到")

        # 步骤 4: 滑动交付到客人餐盘
        begin_x, begin_y = 636, 1002
        end_x, end_y = _roi_center(PLATE_ROI[guest_idx])

        # logger.info(
        #     f"【魂生又一串】交付饮料 {drink_name} {begin_x},{begin_y}→{end_x},{end_y}"
        # )
        context.tasker.controller.post_swipe(begin_x, begin_y, end_x, end_y, 150).wait()
        time.sleep(self.delivery_settle)

        # 更新 token 状态
        token.state = S_PENDING_CONFIRM
        token.delivered_at = time.time()
        logger.info(f"【魂生又一串】饮料 {drink_name} 已交付客人{guest_idx}")

        # 该客人全部交付完成 → 启动清理计时
        if self._is_guest_done(guest_idx):
            self._schedule_cleanup(guest_idx)

        return True

    def _check_burnt_food(self, context: Context, img):
        """检测并处理烧焦食物 — 从烤架滑动到垃圾桶。"""
        result = context.run_recognition("BBQv2_烧焦检测", img)
        if not result or not getattr(result, "hit", False):
            return

        box = _box_to_tuple(result.box)
        if not box:
            return

        start_x, start_y = _roi_center(box)
        trash_x, trash_y = 47, 1175  # 垃圾桶位置

        logger.info(f"【魂生又一串】检测到烧焦食物 ({start_x},{start_y})，丢弃到垃圾桶")
        context.tasker.controller.post_swipe(
            start_x, start_y, trash_x, trash_y, 150
        ).wait()
        time.sleep(GESTURE_DELAY_S)

        # 清理对应烤架槽位状态
        with self._state_lock:
            for slot_id, slot_data in self.grill_state.items():
                if slot_data:
                    sx, sy = _roi_center(GRILL_SLOTS_ROI.get(slot_id, [0, 0, 0, 0]))
                    if abs(sx - start_x) < 100 and abs(sy - start_y) < 100:
                        logger.info(f"【魂生又一串】清理烧焦槽位 {slot_id}")
                        self.grill_state[slot_id] = None
                        timer = self._cook_timers.pop(slot_id, None)
                        if timer:
                            timer.cancel()
                        break
