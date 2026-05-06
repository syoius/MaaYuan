"""
BBQv3 Custom Action - experimental threaded scheduler.

Compared with BBQv2 this version separates slow visual scanning from real
controller actions. Scanning can keep recognizing old screenshots while the
single executor thread handles urgent deliveries, and scan commits are adjusted
by an action mutation log so an old scan cannot re-add a just-delivered item.
"""

import json
import queue
import threading
import time
from collections import deque

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

from . import bbq_v2_custom as v2

S_PLACING = "PLACING"
S_DELIVERING = "DELIVERING"
S_DRINKING = "DRINKING"

ACTIVE_TOKEN_STATES = (
    v2.S_WAITING,
    v2.S_COOKING,
    v2.S_COOKED,
    v2.S_DRINK_READY,
    S_PLACING,
    S_DELIVERING,
    S_DRINKING,
)

GRILL_BLOCKING_STATES = (
    S_PLACING,
    v2.S_COOKING,
    v2.S_COOKED,
    S_DELIVERING,
)

PRIO_STOP = 0
PRIO_DELIVER = 10
PRIO_BURNT = 15
PRIO_CLEANUP = 20

DEFAULT_SCAN_INTERVAL_S = 0.20
DEFAULT_SCAN_SETTLE_S = 0.18
DEFAULT_SCAN_RECHECK_S = 1.20
DEFAULT_SCAN_BUDGET_S = 1.20
DEFAULT_SCAN_READY_MARGIN_S = 0.25
DEFAULT_PLACE_BURST_LIMIT = 6
MUTATION_HISTORY_LIMIT = 128


def _clamp_int(value, min_value: int, max_value: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _mask_recognition_box(img, box, padding: int = 3):
    rect = v2._box_to_tuple(box)
    if not rect:
        return
    x, y, w, h = rect
    height, width = img.shape[:2]
    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(width, int(x + w) + padding)
    y2 = min(height, int(y + h) + padding)
    if x1 < x2 and y1 < y2:
        img[y1:y2, x1:x2] = 0


@AgentServer.custom_action("BBQv3Custom")
class BBQv3Custom(v2.BBQv2Custom):
    """
    Multi-threaded BBQ scheduler.

    Thread model:
    - Timer callbacks only enqueue high-priority actions.
    - Executor thread is the only thread that performs click/swipe/run_action.
    - Scanner thread performs screenshot/recognition and commits per-guest
      order snapshots after adjusting them with action mutations.
    """

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._action_queue = queue.PriorityQueue()
        self._action_seq = 0
        self._action_seq_lock = threading.Lock()
        self._action_busy = threading.Event()
        # Maa AgentServer reverse APIs are not safe to call concurrently from
        # multiple Python threads. Keep scanner/executor independent at the
        # scheduler level, but serialize each framework/controller call.
        self._maa_lock = threading.RLock()
        self._vision_lock = self._maa_lock
        self._threads = []

        self._mutation_seq = 0
        self._mutation_log = deque(maxlen=MUTATION_HISTORY_LIMIT)
        self._dirty_guests = set()
        self._cleanup_action_pending = set()
        self._cleanup_scheduled_guests = set()
        self._stop_confirm_enqueued = False
        self._next_scan_guest = 1

        self._last_action_at = 0.0
        self._last_scan_at = 0.0
        self.max_customers = 2
        self._last_guest_scan_at = {guest: 0.0 for guest in self._active_guests()}
        self._guest_order_item_counts = {guest: 0 for guest in self._active_guests()}
        self._guest_grill_slots = {guest: set() for guest in self._active_guests()}
        self._last_scan_duration = DEFAULT_SCAN_BUDGET_S
        self._next_drink_attempt_at = 0.0
        self._drink_action_touched = False

        self.scan_interval_s = DEFAULT_SCAN_INTERVAL_S
        self.scan_settle_s = DEFAULT_SCAN_SETTLE_S
        self.scan_recheck_s = DEFAULT_SCAN_RECHECK_S
        self.scan_budget_s = DEFAULT_SCAN_BUDGET_S
        self.scan_ready_margin_s = DEFAULT_SCAN_READY_MARGIN_S
        self.max_demands_per_guest = 4
        self.place_burst_limit = DEFAULT_PLACE_BURST_LIMIT

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        self._reset_runtime_config()

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except json.JSONDecodeError:
                logger.warning("【魂生又一串V3】自助烤串: 无法解析 argv 参数")

        base_params = {}
        attach_params = {}
        for node_name in ("BBQv2Custom启动",):
            try:
                node_data = context.get_node_data(node_name)
                if node_data:
                    base_params = (
                        node_data.get("action", {})
                        .get("param", {})
                        .get("custom_action_param", {})
                    )
                    attach_params = node_data.get("attach", {})
                    break
            except Exception as e:
                logger.warning(f"【魂生又一串V3】读取节点数据失败 {node_name}: {e}")

        merged = dict(base_params)
        merged.update(params)
        merged.update(attach_params)
        params = merged

        for pos in range(1, 6):
            food_name = params.get(f"position_{pos}")
            if food_name:
                self.food_positions[food_name] = pos

        for food_name in self.food_positions:
            key = f"cook_time_{food_name}"
            raw = params.get(key, v2.DEFAULT_COOK_DURATIONS.get(food_name, 4.0))
            self.cook_durations[food_name] = float(raw)

        for drink_pos in range(1, 3):
            drink_name = params.get(f"drink_position_{drink_pos}")
            if drink_name:
                self.drink_positions[drink_name] = drink_pos

        enable_second_grill = params.get("enable_second_grill", False)
        max_duration = params.get("max_duration", v2.MAX_SESSION_DURATION_S)
        self.drink_select_delay = params.get("drink_select_delay", 300) / 1000.0
        self.delivery_settle = params.get("delivery_settle", 300) / 1000.0
        self.pending_timeout = params.get("pending_timeout", 5000) / 1000.0
        self.cleanup_delay = params.get("cleanup_delay", 1000) / 1000.0
        self.max_customers = _clamp_int(params.get("max_customers", 2), 1, 3, 2)

        self.scan_interval_s = params.get("scan_interval", 200) / 1000.0
        self.scan_settle_s = params.get("scan_settle", 180) / 1000.0
        self.scan_recheck_s = params.get("scan_recheck", 1200) / 1000.0
        self.scan_budget_s = params.get("scan_budget", 1200) / 1000.0
        self.scan_ready_margin_s = params.get("scan_ready_margin", 250) / 1000.0
        self.max_demands_per_guest = max(1, int(params.get("max_demands_per_guest", 4)))
        self.place_burst_limit = max(
            1, int(params.get("place_burst_limit", DEFAULT_PLACE_BURST_LIMIT))
        )
        self._last_scan_duration = self.scan_budget_s

        self._init_session(enable_second_grill)
        self.session_start = time.time()

        try:
            self._main_loop(context, max_duration)
        except Exception as e:
            logger.exception(f"【魂生又一串V3】自助烤串异常: {e}")
            return CustomAction.RunResult(success=False)
        finally:
            self._stop_event.set()
            self._cancel_all_timers()
            self._join_threads()

        logger.info("【魂生又一串V3】自助烤串结束")
        return CustomAction.RunResult(success=True)

    def _init_session(self, enable_second_grill):
        super()._init_session(enable_second_grill)
        self.customer_orders = {guest: [] for guest in self._active_guests()}
        self.customer_cleanup = {guest: False for guest in self._active_guests()}
        self._stop_event.clear()
        self._action_busy.clear()
        self._threads = []
        self._mutation_seq = 0
        self._mutation_log.clear()
        self._dirty_guests = set(self._active_guests())
        self._cleanup_action_pending.clear()
        self._cleanup_scheduled_guests.clear()
        self._stop_confirm_enqueued = False
        self._next_scan_guest = 1
        self._last_action_at = 0.0
        self._last_scan_at = 0.0
        self._last_guest_scan_at = {guest: 0.0 for guest in self._active_guests()}
        self._guest_order_item_counts = {guest: 0 for guest in self._active_guests()}
        self._guest_grill_slots = {guest: set() for guest in self._active_guests()}
        self._next_drink_attempt_at = 0.0
        self._drink_action_touched = False

        while not self._action_queue.empty():
            try:
                self._action_queue.get_nowait()
                self._action_queue.task_done()
            except queue.Empty:
                break

    # ----------------------------------------------------------
    # Thread lifecycle
    # ----------------------------------------------------------

    def _main_loop(self, context: Context, max_duration: float):
        self._start_threads(context)
        while not self._stop_event.is_set():
            if self._should_stop_safe(context):
                self._stop_event.set()
                break
            elapsed = time.time() - self.session_start
            if elapsed > max_duration:
                logger.info(f"【魂生又一串V3】超时 ({max_duration}s)")
                self._stop_event.set()
                break
            time.sleep(0.10)

    def _start_threads(self, context: Context):
        scanner = threading.Thread(
            target=self._scanner_loop,
            args=(context,),
            name="BBQv3Scanner",
            daemon=True,
        )
        executor = threading.Thread(
            target=self._executor_loop,
            args=(context,),
            name="BBQv3Executor",
            daemon=True,
        )
        self._threads = [scanner, executor]
        for thread in self._threads:
            thread.start()

    def _join_threads(self):
        for thread in self._threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning(f"【魂生又一串V3】线程未及时退出: {thread.name}")

    def _active_guests(self):
        return range(1, self.max_customers + 1)

    def _should_stop_safe(self, context: Context) -> bool:
        with self._maa_lock:
            return v2._should_stop(context)

    # ----------------------------------------------------------
    # Queue helpers
    # ----------------------------------------------------------

    def _enqueue_action(self, priority: int, action: str, payload=None):
        if self._stop_event.is_set():
            return
        with self._action_seq_lock:
            self._action_seq += 1
            seq = self._action_seq
        self._action_queue.put((priority, seq, action, payload or {}))

    def _has_queued_action(self, max_priority=None) -> bool:
        with self._action_queue.mutex:
            if max_priority is None:
                return bool(self._action_queue.queue)
            return any(item[0] <= max_priority for item in self._action_queue.queue)

    def _run_serial_action(self, func, *args, **kwargs):
        if self._stop_event.is_set():
            return None
        self._action_busy.set()
        did_action = False
        try:
            result = func(*args, **kwargs)
            did_action = bool(result)
            return result
        finally:
            if did_action:
                self._last_action_at = time.time()
            self._action_busy.clear()

    def _record_mutation_locked(
        self, kind: str, guest_idx=None, food_name=None, orders=None
    ):
        self._mutation_seq += 1
        mutation = {
            "seq": self._mutation_seq,
            "kind": kind,
            "guest": guest_idx,
            "food": food_name,
            "at": time.time(),
        }
        if orders is not None:
            mutation["orders"] = list(orders)
        self._mutation_log.append(mutation)
        if guest_idx in self.customer_orders:
            self._dirty_guests.add(guest_idx)
        return mutation

    @staticmethod
    def _remove_one(items, value):
        try:
            items.remove(value)
        except ValueError:
            pass

    @staticmethod
    def _token_order(token):
        try:
            return int(str(token.id).split("-")[-1])
        except (TypeError, ValueError):
            return 0

    # ----------------------------------------------------------
    # Timer callbacks
    # ----------------------------------------------------------

    def _schedule_cook_timer(self, slot_id: str, token):
        cook_time = self.cook_durations.get(token.food_name, 4.0)
        guest_idx = token.guest_idx
        food_name = token.food_name
        token_id = token.id

        def _on_timer():
            if self._stop_event.is_set():
                return
            with self._state_lock:
                slot_data = self.grill_state.get(slot_id)
                live_token = self._get_token(token_id)
                if (
                    not slot_data
                    or not live_token
                    or slot_data.get("token_id") != token_id
                ):
                    return
                live_token.state = v2.S_COOKED
                slot_data["state"] = v2.S_COOKED
            self._enqueue_action(
                PRIO_DELIVER,
                "deliver",
                {
                    "slot_id": slot_id,
                    "guest_idx": guest_idx,
                    "food_name": food_name,
                    "token_id": token_id,
                },
            )

        timer = threading.Timer(cook_time, _on_timer)
        timer.daemon = True
        timer.start()
        self._cook_timers[slot_id] = timer

    def _schedule_cleanup(self, guest_idx: int):
        if guest_idx not in self.customer_orders:
            return
        with self._state_lock:
            if guest_idx in self._cleanup_scheduled_guests:
                return
            self._cleanup_scheduled_guests.add(guest_idx)
            multiplier = self._cleanup_delay_multiplier_locked(guest_idx)
            delay_s = self.cleanup_delay * multiplier

        def _on_cleanup():
            if self._stop_event.is_set():
                return
            self._enqueue_cleanup_action(guest_idx, source="timer")

        # logger.info(
        #     f"【魂生又一串V3】客人{guest_idx}清理延时 "
        #     f"{delay_s:.2f}s ({self.cleanup_delay:.2f}s×{multiplier})"
        # )
        timer = threading.Timer(delay_s, _on_cleanup)
        timer.daemon = True
        timer.start()
        self._cleanup_timers[guest_idx] = timer

    def _cleanup_delay_multiplier_locked(self, guest_idx: int) -> int:
        token_count = sum(
            1
            for token in self.tokens
            if token.guest_idx == guest_idx
            and token.state
            in (
                v2.S_PENDING_CONFIRM,
                v2.S_DELIVERED,
                S_DELIVERING,
                S_DRINKING,
            )
        )
        order_count = len(self.customer_orders.get(guest_idx, []))
        remembered_count = self._guest_order_item_counts.get(guest_idx, 0)
        return max(1, token_count, order_count, remembered_count)

    def _enqueue_cleanup_action(self, guest_idx: int, source: str, box=None):
        if guest_idx not in self.customer_orders:
            return
        with self._state_lock:
            if guest_idx in self._cleanup_action_pending:
                return
            self._cleanup_action_pending.add(guest_idx)
        self._enqueue_action(
            PRIO_CLEANUP,
            "cleanup",
            {"guest_idx": guest_idx, "source": source, "box": box},
        )

    def _cancel_all_timers(self):
        super()._cancel_all_timers()
        with self._state_lock:
            self._cleanup_scheduled_guests.clear()
            self._cleanup_action_pending.clear()

    # ----------------------------------------------------------
    # Scanner thread
    # ----------------------------------------------------------

    def _scanner_loop(self, context: Context):
        while not self._stop_event.is_set():
            try:
                if not self._should_scan_now():
                    self._stop_event.wait(self.scan_interval_s)
                    continue

                started_at = time.time()
                with self._state_lock:
                    scan_seq = self._mutation_seq

                img = self._safe_screencap(context)
                if img is None:
                    self._stop_event.wait(0.30)
                    continue

                if self._check_stop_conditions_from_scan(context, img):
                    break
                self._detect_visual_events(context, img)

                committed = False
                for guest_idx in self._guests_to_scan():
                    if self._scanner_should_yield():
                        break
                    demands, complete = self._scan_guest_orders(context, img, guest_idx)
                    if not complete:
                        break
                    self._commit_guest_scan(guest_idx, demands, scan_seq)
                    self._mark_guest_scanned(guest_idx)
                    committed = True
                    if self._scanner_should_yield():
                        break

                now = time.time()
                self._last_scan_duration = max(now - started_at, 0.05)
                self._last_scan_at = now
                if committed:
                    self._stop_event.wait(self.scan_interval_s)
                else:
                    self._stop_event.wait(0.05)
            except Exception as e:
                logger.exception(f"【魂生又一串V3】扫描线程异常: {e}")
                self._stop_event.wait(0.30)

    def _safe_screencap(self, context: Context):
        with self._maa_lock:
            return v2._screencap(context)

    def _recognize(self, context: Context, node_name: str, img, override=None):
        with self._maa_lock:
            if override is None:
                return context.run_recognition(node_name, img)
            return context.run_recognition(node_name, img, override)

    def _run_action_safe(
        self, context: Context, entry: str, box, detail="", override=None
    ):
        with self._maa_lock:
            return context.run_action(entry, box, detail, override or {})

    def _post_click_wait(self, context: Context, x: int, y: int):
        with self._maa_lock:
            return context.tasker.controller.post_click(x, y).wait()

    def _post_swipe_wait(
        self,
        context: Context,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int,
    ):
        with self._maa_lock:
            return context.tasker.controller.post_swipe(
                start_x, start_y, end_x, end_y, duration
            ).wait()

    def _cleanup_grill_slot(self, context: Context, slot_id: str):
        slot_roi = v2.GRILL_SLOTS_ROI.get(slot_id)
        if not slot_roi:
            return
        sx, sy = v2._roi_center(slot_roi)
        trash_x, trash_y = 47, 1175
        # logger.info(f"【魂生又一串V3】清理槽位{slot_id}")
        self._post_swipe_wait(context, sx, sy, trash_x, trash_y, 150)
        time.sleep(v2.GESTURE_DELAY_S)

    def _track_guest_slot_locked(self, guest_idx: int, slot_id: str):
        self._guest_grill_slots.setdefault(guest_idx, set()).add(slot_id)

    def _untrack_guest_slot_locked(self, guest_idx: int, slot_id: str):
        self._guest_grill_slots.setdefault(guest_idx, set()).discard(slot_id)

    def _slots_for_guest_cleanup_locked(self, guest_idx: int):
        slots = set()
        for token in self.tokens:
            if token.guest_idx == guest_idx and token.grill_slot:
                slots.add(token.grill_slot)
        for slot_id, slot_data in self.grill_state.items():
            if slot_data and slot_data.get("guest_idx") == guest_idx:
                slots.add(slot_id)
        for slot_id in self._guest_grill_slots.get(guest_idx, set()):
            slot_data = self.grill_state.get(slot_id)
            if slot_data is None or slot_data.get("guest_idx") == guest_idx:
                slots.add(slot_id)
        return [slot_id for slot_id in self.grill_state if slot_id in slots]

    def _has_other_guest_grill_work_locked(self, guest_idx: int) -> bool:
        for token in self.tokens:
            if (
                token.guest_idx != guest_idx
                and token.food_name in self.cook_durations
                and token.state in GRILL_BLOCKING_STATES
            ):
                return True
        for slot_data in self.grill_state.values():
            if slot_data and slot_data.get("guest_idx") != guest_idx:
                return True
        return False

    def _should_scan_now(self) -> bool:
        if self._action_busy.is_set():
            return False
        if self._has_queued_action(PRIO_CLEANUP):
            return False
        now = time.time()
        if now - self._last_action_at < self.scan_settle_s:
            return False
        if self._ready_imminent(self._scan_budget_window()):
            return False
        with self._state_lock:
            has_dirty_guest = bool(self._dirty_guests)
            has_stale_guest = any(
                now - self._last_guest_scan_at.get(guest_idx, 0.0)
                >= self.scan_recheck_s
                for guest_idx in self._active_guests()
            )
        return (
            has_dirty_guest
            or has_stale_guest
            or now - self._last_scan_at >= self.scan_recheck_s
        )

    def _scan_budget_window(self):
        return (
            max(self.scan_budget_s, self._last_scan_duration) + self.scan_ready_margin_s
        )

    def _ready_imminent(self, window_s: float) -> bool:
        now = time.time()
        with self._state_lock:
            for token in self.tokens:
                if token.state in (v2.S_COOKING, v2.S_COOKED) and token.ready_at:
                    if token.ready_at - now <= window_s:
                        return True
        return False

    def _scanner_should_yield(self) -> bool:
        if self._stop_event.is_set() or self._action_busy.is_set():
            return True
        if self._has_queued_action(PRIO_CLEANUP):
            return True
        return self._ready_imminent(0.05)

    def _guests_to_scan(self):
        with self._state_lock:
            start = self._next_scan_guest
            guest_count = self.max_customers
        return [
            ((start + offset - 1) % guest_count) + 1 for offset in range(guest_count)
        ]

    def _mark_guest_scanned(self, guest_idx: int):
        with self._state_lock:
            self._last_guest_scan_at[guest_idx] = time.time()
            self._next_scan_guest = (guest_idx % self.max_customers) + 1

    def _check_stop_conditions_from_scan(self, context: Context, img) -> bool:
        result = self._recognize(context, "BBQv2_确定按钮", img)
        if result and getattr(result, "hit", False):
            with self._state_lock:
                if self._stop_confirm_enqueued:
                    return True
                self._stop_confirm_enqueued = True
            self._enqueue_action(PRIO_STOP, "stop_confirm", {"box": result.box})
            return True

        result2 = self._recognize(context, "BBQv2_时间归零", img)
        if result2 and getattr(result2, "hit", False):
            logger.info("【魂生又一串V3】检测到时间归零，停止")
            self._stop_event.set()
            return True
        return False

    def _detect_visual_events(self, context: Context, img):
        cleanup = self._recognize(context, "BBQv2_清理_模板", img)
        if cleanup and getattr(cleanup, "hit", False):
            box = v2._box_to_tuple(cleanup.box)
            if box:
                guest_idx = self._guest_from_x(box[0])
                if guest_idx not in self.customer_orders:
                    return
                self._enqueue_cleanup_action(guest_idx, source="vision", box=box)

        burnt = self._recognize(context, "BBQv2_烧焦检测", img)
        if burnt and getattr(burnt, "hit", False):
            box = v2._box_to_tuple(burnt.box)
            if box:
                self._enqueue_action(PRIO_BURNT, "burnt", {"box": box})

    def _scan_guest_orders(self, context: Context, img, guest_idx: int):
        bubble_roi = v2.BUBBLE_ROI[guest_idx]
        demands = []
        for food_name in v2.FOOD_TEMPLATES:
            if self._scanner_should_yield():
                return demands, False
            if (
                food_name not in self.food_positions
                and food_name not in self.drink_positions
            ):
                continue
            node_name = f"BBQv2_{food_name}_模板"
            count, complete = self._detect_food_count_safe(
                context, img, node_name, bubble_roi
            )
            if not complete:
                return demands, False
            for _ in range(count):
                demands.append(food_name)
                if len(demands) >= self.max_demands_per_guest:
                    return demands, True
        return demands, True

    def _detect_food_count_safe(self, context, img, node_name, bubble_roi):
        working_img = img.copy()
        count = 0
        for _ in range(4):
            if self._scanner_should_yield():
                return count, False
            override = {node_name: {"recognition": {"param": {"roi": bubble_roi}}}}
            result = self._recognize(context, node_name, working_img, override)
            if not result or not getattr(result, "hit", False):
                break
            count += 1
            _mask_recognition_box(working_img, result.box)
        return count, True

    def _commit_guest_scan(self, guest_idx: int, demands, scan_seq: int):
        if guest_idx not in self.customer_orders:
            return
        adjusted = list(demands)
        with self._state_lock:
            for mutation in self._mutation_log:
                if mutation["seq"] <= scan_seq or mutation["guest"] != guest_idx:
                    continue
                if mutation["kind"] == "deliver":
                    self._remove_one(adjusted, mutation["food"])
                elif mutation["kind"] == "cleanup":
                    adjusted = []

            merged_orders = self._merge_scan_demands_locked(guest_idx, adjusted)
            self.customer_orders[guest_idx] = merged_orders
            self._guest_order_item_counts[guest_idx] = max(
                self._guest_order_item_counts.get(guest_idx, 0),
                len(merged_orders),
            )
            self.customer_cleanup[guest_idx] = False
            self._dirty_guests.discard(guest_idx)
            self._reconcile_tokens()
            self._cleanup_delivered_tokens()

        if merged_orders:
            logger.info(f"【魂生又一串V3】客人{guest_idx}点单 {merged_orders}")

    # ----------------------------------------------------------
    # Executor thread
    # ----------------------------------------------------------

    def _executor_loop(self, context: Context):
        while not self._stop_event.is_set():
            try:
                priority, seq, action, payload = self._action_queue.get(timeout=0.05)
            except queue.Empty:
                try:
                    self._try_execute_planned_work(context)
                except Exception as e:
                    logger.exception(f"【魂生又一串V3】规划动作失败: {e}")
                    self._stop_event.wait(0.20)
                continue

            try:
                if action == "stop_confirm":
                    self._run_serial_action(
                        self._execute_stop_confirm, context, payload
                    )
                elif action == "deliver":
                    self._run_serial_action(self._execute_delivery, context, payload)
                elif action == "cleanup":
                    self._run_serial_action(self._execute_cleanup, context, payload)
                elif action == "burnt":
                    self._run_serial_action(self._execute_burnt, context, payload)
            except Exception as e:
                logger.exception(f"【魂生又一串V3】执行动作失败 {action}: {e}")
            finally:
                self._action_queue.task_done()

    def _try_execute_planned_work(self, context: Context):
        if self._stop_event.is_set():
            return
        if self._has_queued_action(PRIO_CLEANUP):
            return
        placed_any = False
        placed_guest = None
        for _ in range(self.place_burst_limit):
            if self._has_queued_action(PRIO_CLEANUP):
                break
            if self._ready_imminent(0.05):
                break
            guest_idx = self._run_serial_action(
                self._execute_place_waiting_food, context, placed_guest
            )
            if not guest_idx:
                break
            if placed_guest is None:
                placed_guest = guest_idx
            placed_any = True
        if self._has_queued_action(PRIO_CLEANUP):
            return
        if placed_guest is not None:
            self._run_serial_action(self._execute_drink_order, context, placed_guest)
            return
        if time.time() < self._next_drink_attempt_at:
            return
        if self._run_serial_action(self._execute_drink_order, context, None):
            return
        if placed_any:
            return
        if self._any_waiting_food():
            return

    def _execute_stop_confirm(self, context: Context, payload):
        box = payload.get("box")
        if box is not None:
            logger.info("【魂生又一串V3】检测到确定按钮，点击并停止")
            self._run_action_safe(context, "BBQv2_确定按钮", box)
        self._stop_event.set()
        return box is not None

    def _execute_delivery(self, context: Context, payload):
        slot_id = payload["slot_id"]
        token_id = payload["token_id"]
        guest_idx = payload["guest_idx"]
        food_name = payload["food_name"]

        with self._state_lock:
            slot_data = self.grill_state.get(slot_id)
            token = self._get_token(token_id)
            if (
                not slot_data
                or not token
                or slot_data.get("token_id") != token_id
                or token.state in (v2.S_DELIVERED, v2.S_PENDING_CONFIRM)
            ):
                logger.info(f"【魂生又一串V3】跳过过期交付 {slot_id} {food_name}")
                return False
            token.state = S_DELIVERING
            slot_data["state"] = S_DELIVERING

        grill_roi = v2.GRILL_SLOTS_ROI.get(slot_id, v2.GRILL_1_ROI)
        start_x, start_y = v2._roi_center(grill_roi)
        end_x, end_y = v2._roi_center(v2.PLATE_ROI[guest_idx])

        logger.info(f"【魂生又一串V3】交付 {food_name} {slot_id}→客人{guest_idx}")
        self._post_swipe_wait(
            context, start_x, start_y, end_x, end_y, v2.DELIVERY_SWIPE_DURATION_MS
        )
        time.sleep(self.delivery_settle)

        schedule_cleanup = False
        with self._state_lock:
            token = self._get_token(token_id)
            if token:
                token.state = v2.S_PENDING_CONFIRM
                token.delivered_at = time.time()
                token.grill_slot = None
            self.grill_state[slot_id] = None
            self._untrack_guest_slot_locked(guest_idx, slot_id)
            self._cook_timers.pop(slot_id, None)
            self._record_mutation_locked("deliver", guest_idx, food_name)
            schedule_cleanup = self._is_guest_done_locked(guest_idx)

        if schedule_cleanup:
            self._schedule_cleanup(guest_idx)
        return True

    def _execute_cleanup(self, context: Context, payload):
        guest_idx = payload["guest_idx"]
        if guest_idx not in self.customer_orders:
            with self._state_lock:
                self._cleanup_action_pending.discard(guest_idx)
                self._cleanup_scheduled_guests.discard(guest_idx)
            return False
        box = payload.get("box")
        if not box:
            box = self._find_cleanup_box(context, guest_idx)
        if box:
            cx, cy = v2._roi_center(box)
        else:
            cx, cy = v2._roi_center(v2.CLEANUP_CLICK_ROI[guest_idx])

        # logger.info(f"【魂生又一串V3】清理客人{guest_idx}餐盘")
        self._post_click_wait(context, cx, cy)
        time.sleep(v2.GESTURE_DELAY_S)

        with self._state_lock:
            self.customer_orders[guest_idx] = []
            self._guest_order_item_counts[guest_idx] = 0
            self.customer_cleanup[guest_idx] = False
            self.tokens = [
                t
                for t in self.tokens
                if not (
                    t.guest_idx == guest_idx
                    and t.state
                    in (
                        v2.S_DELIVERED,
                        v2.S_PENDING_CONFIRM,
                        v2.S_COOKED,
                        v2.S_DRINK_READY,
                        S_DELIVERING,
                        S_DRINKING,
                    )
                )
            ]
            for slot_id, slot_data in self.grill_state.items():
                if slot_data and slot_data.get("guest_idx") == guest_idx:
                    token_id = slot_data.get("token_id")
                    token = self._get_token(token_id) if token_id else None
                    if token and token.state in (v2.S_COOKING, S_PLACING):
                        continue
                    self.grill_state[slot_id] = None
                    self._untrack_guest_slot_locked(guest_idx, slot_id)
                    timer = self._cook_timers.pop(slot_id, None)
                    if timer:
                        timer.cancel()
            timer = self._cleanup_timers.pop(guest_idx, None)
            if timer:
                timer.cancel()
            self._cleanup_scheduled_guests.discard(guest_idx)
            self._cleanup_action_pending.discard(guest_idx)
            self._record_mutation_locked("cleanup", guest_idx, None)
            if not any(
                slot_data and slot_data.get("guest_idx") == guest_idx
                for slot_data in self.grill_state.values()
            ):
                self._guest_grill_slots.setdefault(guest_idx, set()).clear()
        return True

    def _find_cleanup_box(self, context: Context, guest_idx: int):
        img = self._safe_screencap(context)
        if img is None:
            return None
        result = self._recognize(context, "BBQv2_清理_模板", img)
        if not result or not getattr(result, "hit", False):
            return None
        box = v2._box_to_tuple(result.box)
        if not box or self._guest_from_x(box[0]) != guest_idx:
            return None
        return box

    def _execute_burnt(self, context: Context, payload):
        box = payload.get("box")
        if not box:
            return False
        start_x, start_y = v2._roi_center(box)
        trash_x, trash_y = 47, 1175

        logger.info(f"【魂生又一串V3】检测到烧焦食物 ({start_x},{start_y})，丢弃")
        self._post_swipe_wait(context, start_x, start_y, trash_x, trash_y, 150)
        time.sleep(v2.GESTURE_DELAY_S)

        with self._state_lock:
            for slot_id, slot_data in self.grill_state.items():
                if not slot_data:
                    continue
                sx, sy = v2._roi_center(v2.GRILL_SLOTS_ROI.get(slot_id, [0, 0, 0, 0]))
                if abs(sx - start_x) >= 100 or abs(sy - start_y) >= 100:
                    continue
                token_id = slot_data.get("token_id")
                token = self._get_token(token_id) if token_id else None
                owner_guest = slot_data.get("guest_idx")
                if token:
                    owner_guest = token.guest_idx
                    token.state = v2.S_WAITING
                    token.grill_slot = None
                    token.started_at = None
                    token.ready_at = None
                self.grill_state[slot_id] = None
                if owner_guest:
                    self._untrack_guest_slot_locked(owner_guest, slot_id)
                timer = self._cook_timers.pop(slot_id, None)
                if timer:
                    timer.cancel()
                break
        return True

    def _execute_place_waiting_food(self, context: Context, preferred_guest_idx=None):
        reservation = None
        with self._state_lock:
            for slot_id, slot_data in self.grill_state.items():
                if slot_data is not None:
                    continue
                for token in sorted(self.tokens, key=self._token_order):
                    if (
                        token.state != v2.S_WAITING
                        or token.food_name not in self.cook_durations
                    ):
                        continue
                    if (
                        preferred_guest_idx is not None
                        and token.guest_idx != preferred_guest_idx
                    ):
                        continue
                    pos = self.food_positions.get(token.food_name)
                    if pos is None or pos not in v2.POSITION_CLICK_ROI:
                        continue
                    token.state = S_PLACING
                    token.grill_slot = slot_id
                    self.grill_state[slot_id] = {
                        "food": token.food_name,
                        "token_id": token.id,
                        "guest_idx": token.guest_idx,
                        "state": S_PLACING,
                        "started_at": None,
                        "ready_at": None,
                    }
                    reservation = (
                        slot_id,
                        token.id,
                        token.guest_idx,
                        token.food_name,
                        pos,
                    )
                    break
                if reservation:
                    break

        if not reservation:
            return False

        slot_id, token_id, guest_idx, food_name, pos = reservation
        try:
            self._cleanup_grill_slot(context, slot_id)

            cx, cy = v2._roi_center(v2.POSITION_CLICK_ROI[pos])
            # logger.info(f"【魂生又一串V3】上架 {food_name} -> 烤架{slot_id}")
            self._post_click_wait(context, cx, cy)
            time.sleep(v2.GESTURE_DELAY_S)

            now = time.time()
            cook_time = self.cook_durations.get(food_name, 4.0)
            with self._state_lock:
                self._track_guest_slot_locked(guest_idx, slot_id)
                token = self._get_token(token_id)
                if not token or token.state != S_PLACING:
                    self.grill_state[slot_id] = None
                    self._untrack_guest_slot_locked(guest_idx, slot_id)
                    return guest_idx
                token.state = v2.S_COOKING
                token.started_at = now
                token.ready_at = now + cook_time
                self.grill_state[slot_id] = {
                    "food": food_name,
                    "token_id": token_id,
                    "guest_idx": guest_idx,
                    "state": v2.S_COOKING,
                    "started_at": now,
                    "ready_at": now + cook_time,
                }
                self._schedule_cook_timer(slot_id, token)
            return guest_idx
        except Exception:
            with self._state_lock:
                token = self._get_token(token_id)
                if token and token.state == S_PLACING:
                    token.state = v2.S_WAITING
                    token.grill_slot = None
                if self.grill_state.get(slot_id, {}).get("token_id") == token_id:
                    self.grill_state[slot_id] = None
                    self._untrack_guest_slot_locked(guest_idx, slot_id)
            raise

    def _execute_drink_order(self, context: Context, preferred_guest_idx=None) -> bool:
        drink_token = None
        with self._state_lock:
            for token in sorted(self.tokens, key=self._token_order):
                if (
                    token.state == v2.S_WAITING
                    and token.food_name in self.drink_positions
                    and (
                        preferred_guest_idx is None
                        or token.guest_idx == preferred_guest_idx
                    )
                ):
                    if self._has_other_guest_grill_work_locked(token.guest_idx):
                        continue
                    drink_token = token
                    token.state = S_DRINKING
                    break

        if drink_token is None:
            return False

        self.drink_machine_busy = True
        success = False
        try:
            self._drink_action_touched = False
            success = self._do_drink_order_v3(
                context, drink_token.food_name, drink_token.guest_idx, drink_token.id
            )
            if not success:
                self._next_drink_attempt_at = time.time() + 0.60
            else:
                self._next_drink_attempt_at = 0.0
            return success or self._drink_action_touched
        finally:
            with self._state_lock:
                token = self._get_token(drink_token.id)
                if token and not success and token.state == S_DRINKING:
                    token.state = v2.S_WAITING
            self.drink_machine_busy = False

    def _do_drink_order_v3(
        self, context: Context, drink_name: str, guest_idx: int, token_id: str
    ) -> bool:
        logger.info(f"【魂生又一串V3】开始制作饮料 {drink_name} 给客人{guest_idx}")

        img0 = self._safe_screencap(context)
        if img0 is not None:
            trash_x, trash_y = 47, 1175
            for d_name, product_node in self._configured_drink_product_nodes(
                drink_name
            ):
                p_result = self._recognize(context, product_node, img0)
                if not p_result or not getattr(p_result, "hit", False):
                    continue
                if d_name == drink_name:
                    px, py = v2._roi_center(v2._box_to_tuple(p_result.box))
                    end_x, end_y = v2._roi_center(v2.PLATE_ROI[guest_idx])
                    logger.info(f"【魂生又一串V3】发现成品{d_name}，直接交付")
                    self._drink_action_touched = True
                    self._post_swipe_wait(context, px, py, end_x, end_y, 150)
                    time.sleep(self.delivery_settle)
                    self._finish_drink_delivery(token_id, guest_idx, drink_name)
                    return True
                px, py = v2._roi_center(v2._box_to_tuple(p_result.box))
                logger.info(f"【魂生又一串V3】清理遗留成品{d_name}")
                self._drink_action_touched = True
                self._post_swipe_wait(context, px, py, trash_x, trash_y, 150)
                time.sleep(v2.GESTURE_DELAY_S)
                break

        img = self._safe_screencap(context)
        if img is None:
            return False
        result = self._recognize(context, "BBQv2_饮水机可用", img)
        if not result or not getattr(result, "hit", False):
            logger.info("【魂生又一串V3】饮水机不可用，跳过")
            return False
        self._drink_action_touched = True
        self._run_action_safe(context, "BBQv2_饮水机可用", result.box)
        time.sleep(0.3)

        select_node = v2.DRINK_SELECT_NODE.get(drink_name)
        if not select_node:
            logger.warning(f"【魂生又一串V3】未知饮料 {drink_name}")
            return False
        img2 = self._safe_screencap(context)
        if img2 is None:
            return False
        select_result = self._recognize(context, select_node, img2)
        if not select_result or not getattr(select_result, "hit", False):
            logger.info(f"【魂生又一串V3】未找到饮料 {drink_name} 选项")
            return False
        self._drink_action_touched = True
        self._run_action_safe(context, select_node, select_result.box)
        time.sleep(self.drink_select_delay)

        img3 = self._safe_screencap(context)
        if img3 is not None:
            pour_result = self._recognize(context, "BBQv2_长按倒水", img3)
            if pour_result and getattr(pour_result, "hit", False):
                self._drink_action_touched = True
                self._run_action_safe(context, "BBQv2_长按倒水", pour_result.box)
            else:
                logger.info("【魂生又一串V3】倒水按钮未找到")

        begin_x, begin_y = 636, 1002
        end_x, end_y = v2._roi_center(v2.PLATE_ROI[guest_idx])
        self._drink_action_touched = True
        self._post_swipe_wait(context, begin_x, begin_y, end_x, end_y, 150)
        time.sleep(self.delivery_settle)
        self._finish_drink_delivery(token_id, guest_idx, drink_name)
        logger.info(f"【魂生又一串V3】饮料 {drink_name} 已交付客人{guest_idx}")
        return True

    def _configured_drink_product_nodes(self, drink_name: str):
        names = []
        if drink_name in self.drink_positions:
            names.append(drink_name)
        for configured_name in self.drink_positions:
            if configured_name != drink_name:
                names.append(configured_name)
        if not names:
            names.append(drink_name)

        return [
            (name, v2.DRINK_PRODUCT_NODE[name])
            for name in names
            if name in v2.DRINK_PRODUCT_NODE
        ]

    def _finish_drink_delivery(self, token_id: str, guest_idx: int, drink_name: str):
        schedule_cleanup = False
        with self._state_lock:
            token = self._get_token(token_id)
            if token:
                token.state = v2.S_PENDING_CONFIRM
                token.delivered_at = time.time()
            self._record_mutation_locked("deliver", guest_idx, drink_name)
            schedule_cleanup = self._is_guest_done_locked(guest_idx)
        if schedule_cleanup:
            self._schedule_cleanup(guest_idx)

    # ----------------------------------------------------------
    # State logic overrides
    # ----------------------------------------------------------

    def _undelivered_active_states(self):
        return (
            v2.S_WAITING,
            v2.S_COOKING,
            v2.S_COOKED,
            v2.S_DRINK_READY,
            S_PLACING,
            S_DRINKING,
        )

    def _merge_scan_demands_locked(self, guest_idx: int, raw_demands):
        previous = self.customer_orders.get(guest_idx, [])
        if not previous:
            return list(raw_demands)

        merged_counts = {}
        names = set(previous) | set(raw_demands)
        for food_name in names:
            previous_count = previous.count(food_name)
            raw_count = raw_demands.count(food_name)
            if raw_count >= previous_count:
                merged_counts[food_name] = raw_count
                continue

            delivered_allowance = self._delivered_reduction_allowance_locked(
                guest_idx, food_name
            )
            merged_counts[food_name] = max(
                raw_count, previous_count - delivered_allowance
            )

        merged = []
        for food_name in v2.FOOD_TEMPLATES:
            merged.extend([food_name] * merged_counts.get(food_name, 0))
        return merged

    def _delivered_reduction_allowance_locked(self, guest_idx: int, food_name: str):
        return sum(
            1
            for token in self.tokens
            if token.guest_idx == guest_idx
            and token.food_name == food_name
            and token.state in (v2.S_PENDING_CONFIRM, v2.S_DELIVERED)
        )

    def _reconcile_tokens(self):
        pair_keys = set()
        for guest, orders in self.customer_orders.items():
            for food_name in orders:
                pair_keys.add((guest, food_name))
        for token in self.tokens:
            pair_keys.add((token.guest_idx, token.food_name))

        for guest_idx, food_name in pair_keys:
            observed_count = self.customer_orders.get(guest_idx, []).count(food_name)
            tokens_for_pair = [
                t
                for t in self.tokens
                if t.guest_idx == guest_idx
                and t.food_name == food_name
                and t.state != v2.S_DELIVERED
            ]
            guaranteed_count = sum(
                1 for t in tokens_for_pair if t.state in ACTIVE_TOKEN_STATES
            )
            pending_tokens = sorted(
                [t for t in tokens_for_pair if t.state == v2.S_PENDING_CONFIRM],
                key=lambda t: t.delivered_at or 0,
            )

            pending_needed = max(observed_count - guaranteed_count, 0)
            for token in pending_tokens[pending_needed:]:
                token.state = v2.S_DELIVERED
                token.grill_slot = None

            unresolved = guaranteed_count + min(len(pending_tokens), pending_needed)
            shortfall = observed_count - unresolved
            for _ in range(max(shortfall, 0)):
                self._create_token(guest_idx, food_name)

    def _any_waiting_food(self) -> bool:
        with self._state_lock:
            return any(
                t.state == v2.S_WAITING and t.food_name in self.cook_durations
                for t in self.tokens
            )

    def _is_guest_done(self, guest_idx: int) -> bool:
        with self._state_lock:
            return self._is_guest_done_locked(guest_idx)

    def _is_guest_done_locked(self, guest_idx: int) -> bool:
        for token in self.tokens:
            if token.guest_idx == guest_idx and token.state in ACTIVE_TOKEN_STATES:
                return False
        with self._action_queue.mutex:
            for item in self._action_queue.queue:
                payload = item[3]
                if payload.get("guest_idx") == guest_idx:
                    return False
        return True

    @staticmethod
    def _guest_from_x(x: int) -> int:
        if x < 200:
            return 1
        if x < 420:
            return 2
        return 3
