import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from maa.agent.agent_server import AgentServer

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
spec = importlib.util.spec_from_file_location("jianmu", AGENT / "custom/action/jianmu.py")
jianmu = importlib.util.module_from_spec(spec)
with patch.object(AgentServer, "custom_action", return_value=lambda action: action):
    spec.loader.exec_module(jianmu)


class JianmuTests(unittest.TestCase):
    def test_tree_detection_survives_dispatch_button_sparkle(self):
        vision = jianmu.Vision()
        frame = np.full((1280, 720, 3), 75, dtype=np.uint8)
        for name, x, y in [
            ("tree_title", 50, 154), ("outside_button", 610, 1141),
            ("dispatch", 501, 1016), ("collect", 606, 1016),
        ]:
            image = vision.templates[name]
            h, w = image.shape[:2]
            frame[y:y+h, x:x+w] = image
        frame[1015:1040, 513:535] = 255
        self.assertIsNone(vision.hit(frame, "dispatch", (487, 1004, 90, 84)))
        self.assertTrue(vision.tree(frame))
        frame[1118:1224, 590:688] = 75
        self.assertFalse(vision.tree(frame))

    def test_tree_scroll_rechecks_the_settled_activity_screen(self):
        patrol = jianmu.Patrol(Mock(), Mock())
        patrol.swipe = Mock()
        patrol.pause = Mock()
        patrol.wait = Mock(side_effect=["moving", "settled"])
        with patch.object(jianmu, "vertical_delta", side_effect=[373, 0]):
            self.assertEqual(patrol.tree_step("before", 1), ("settled", True))
        patrol.swipe.assert_called_once_with(350, 480, 350, 740, duration=730)
        self.assertEqual(patrol.wait.call_args_list, [
            call(patrol.vision.tree, "树内画面"), call(patrol.vision.tree, "滑动后的树内主界面")
        ])

    def test_map_recognizes_each_marker_once_and_ignores_bright_background(self):
        vision = jianmu.Vision()
        frame = np.full((1280, 720, 3), 221, dtype=np.uint8)
        expected = []
        for kind, (x, y) in zip(("gold", "purple", "blue", "bang"),
                                ((120, 400), (250, 540), (410, 700), (530, 870))):
            template = vision.templates[f"mark_{kind}"]
            h, w = template.shape[:2]
            frame[y:y+h, x:x+w] = template
            expected.append((kind, (x + w // 2, y + h // 2)))
        self.assertEqual(vision.markers(frame), expected)

    def test_overlapping_tree_views_find_direction_and_stationary_boundary(self):
        rng = np.random.default_rng(17)
        scene = rng.integers(0, 255, (1653, 720, 3), dtype=np.uint8)
        before, after = scene[:1280].copy(), scene[373:].copy()
        after[35:240, :220] = 255
        self.assertEqual(jianmu.vertical_delta(before, after), -373)
        self.assertEqual(jianmu.vertical_delta(after, before), 373)
        self.assertEqual(jianmu.vertical_delta(after, after.copy()), 0)

    def test_orders_wait_for_upgrades_and_recheck_markers_after_delivery(self):
        context = Mock()
        context.run_task.return_value = SimpleNamespace(status=SimpleNamespace(succeeded=True))
        patrol = jianmu.Patrol(context, Mock())
        patrol.shot = Mock(return_value="tree")
        patrol.is_upgrade_result = Mock(return_value=None)
        patrol.wait = Mock(side_effect=["tree", "map"])
        patrol.traverse = Mock(side_effect=["top", "bottom", "top-again"])
        patrol.inspect_view = Mock(side_effect=["top", "bottom"])
        patrol.click = Mock()
        patrol.scan_markers = Mock(side_effect=[("map", ["first"]), ("updated", [])])
        patrol.deliver_marker = Mock(side_effect=[("updated", True), ("updated", False)])
        patrol.accepted_orders = Mock(return_value=("updated", False))
        patrol.cycle()
        self.assertEqual(patrol.traverse.call_args_list, [
            call("tree", 1), call("top", -1, "upgrade"), call("bottom", 1, "plan")
        ])
        self.assertEqual(patrol.scan_markers.call_args_list, [call("map"), call("updated")])
        patrol.traverse.side_effect = ["top", jianmu.ScanError("missing floor")]
        patrol.wait = Mock(return_value="tree")
        patrol.inspect_view = Mock(return_value="top")
        patrol.scan_markers.reset_mock()
        with self.assertRaises(jianmu.ScanError):
            patrol.cycle()
        patrol.scan_markers.assert_not_called()

    def test_upgrade_footer_ocr_requires_the_upgrade_result_title(self):
        patrol = jianmu.Patrol(Mock(), Mock())
        patrol.vision.hit.side_effect = lambda frame, key, *args, **kwargs: (
            (360, 510) if key == "upgrade_result" and frame == "result" else None
        )
        patrol.ocr = Mock(return_value=[SimpleNamespace(text="点击 空白处关闭", box=(293, 1193, 134, 24))])
        self.assertIsNone(patrol.upgrade_result_footer("other"))
        patrol.ocr.assert_not_called()
        self.assertEqual(patrol.upgrade_result_footer("result"), (360, 1205))

    def test_closing_map_details_handles_optional_reward(self):
        for frames, clicks in [
            (["map"], [call((676, 201))]),
            (["reward", "map"], [call((676, 201)), call((360, 1211))]),
        ]:
            with self.subTest(frames=frames):
                patrol = jianmu.Patrol(Mock(), Mock())
                hits = {("detail", "panel_close"): (676, 201), ("reward", "reward_title"): (360, 244)}
                patrol.vision.hit.side_effect = lambda frame, key, *args, **kwargs: hits.get((frame, key))
                patrol.vision.outside.side_effect = lambda frame: frame == "map"
                patrol.result_footer = Mock(return_value=(360, 1211))
                patrol.shot = Mock(side_effect=frames)
                patrol.click = Mock()
                self.assertEqual(patrol.close_map_panel("detail"), "map")
                self.assertEqual(patrol.click.call_args_list, clicks)

    def test_map_marker_retry_uses_fresh_position_and_stays_on_map(self):
        marker = ("bang", (616, 972))
        for fresh_markers, expected, clicks in [
            ([("bang", (615, 967))], ("detail", True), [call((616, 972)), call((615, 967))]),
            ([("blue", (616, 972))], ("fresh map", False), [call((616, 972))]),
        ]:
            with self.subTest(fresh_markers=fresh_markers):
                patrol = jianmu.Patrol(Mock(), Mock())
                patrol.vision.outside.side_effect = lambda frame: frame in ("map", "fresh map")
                patrol.vision.markers.side_effect = [[marker], fresh_markers]
                patrol.wait = Mock(side_effect=[jianmu.PatrolError("未确认标记详情"), "detail"])
                patrol.shot = Mock(return_value="fresh map")
                patrol.click = Mock()
                self.assertEqual(patrol.panel("map", marker), expected)
                self.assertEqual(patrol.click.call_args_list, clicks)

    def test_map_scan_continues_when_one_marker_does_not_open(self):
        blocked = ("bang", (616, 972))
        next_marker = ("blue", (579, 460))
        patrol = jianmu.Patrol(Mock(), Mock())
        patrol.vision.markers.return_value = [blocked, next_marker]
        patrol.vision.hit.return_value = None
        patrol.panel = Mock(side_effect=[("map", False), ("detail", True)])
        patrol.ocr = Mock(return_value=[])
        patrol.order_button = Mock(return_value=None)
        patrol.close_map_panel = Mock(return_value="map")
        self.assertEqual(patrol.scan_markers("map"), ("map", []))
        self.assertEqual(patrol.panel.call_args_list, [call("map", blocked), call("map", next_marker)])
        patrol.close_map_panel.assert_called_once_with("detail")

    def test_unaccepted_gold_order_is_inspected_without_accepting(self):
        marker = ("gold", (475, 780))
        patrol = jianmu.Patrol(Mock(), Mock())
        patrol.vision.markers.return_value = [marker]
        patrol.vision.hit.return_value = None
        patrol.panel = Mock(return_value=("unaccepted", True))
        patrol.ocr = Mock(return_value=[])
        patrol.order_button = Mock(side_effect=lambda frame, key: (
            (360, 1050) if key == "accept_order" else None
        ))
        patrol.close_map_panel = Mock(return_value="map")
        patrol.click = Mock()

        self.assertEqual(patrol.scan_markers("map"), ("map", []))
        self.assertEqual(patrol.deliver_marker("map", [marker]), ("map", False))
        patrol.click.assert_not_called()
        self.assertTrue(all(args[1] == "deliver_order" for args, _ in patrol.order_button.call_args_list))

    def test_upgrade_retries_only_the_still_visible_second_confirmation(self):
        patrol = jianmu.Patrol(Mock(), Mock())
        hits = {
            ("panel", "upgrade_confirm"): (574, 1190),
            ("interrupt", "upgrade_interrupt"): (350, 570),
            ("interrupt", "upgrade_interrupt_ok"): (505, 785),
        }
        patrol.vision.hit.side_effect = lambda frame, key, *args, **kwargs: hits.get((frame, key))
        patrol.tabs = Mock(return_value=[None, None, SimpleNamespace(box=(400, 822, 100, 32))])
        patrol.ocr = Mock(return_value=[])
        patrol.wait = Mock(side_effect=["panel", "interrupt", "interrupt", "result"])
        patrol.upgrade_result_footer = Mock(side_effect=lambda frame: (360, 1205) if frame == "result" else None)
        patrol.close_upgrade_result = Mock(return_value="tree")
        patrol.click = Mock()
        patrol.pause = Mock()

        self.assertEqual(patrol.upgrade_room("panel"), ("tree", True))
        self.assertEqual(patrol.click.call_args_list, [
            call((450, 838)), call((574, 1190)), call((505, 785)), call((505, 785)),
        ])
        patrol.close_upgrade_result.assert_called_once_with("result")

    def test_upgrade_closes_material_refund_before_closing_result(self):
        patrol = jianmu.Patrol(Mock(), Mock())
        hits = {
            ("panel", "upgrade_confirm"): (574, 1190),
            ("interrupt", "upgrade_interrupt"): (350, 570),
            ("interrupt", "upgrade_interrupt_ok"): (505, 785),
            ("refund", "upgrade_refund_title"): (360, 244),
            ("result", "upgrade_result"): (360, 510),
        }
        patrol.vision.hit.side_effect = lambda frame, key, *args, **kwargs: hits.get((frame, key))
        patrol.vision.tree.side_effect = lambda frame: frame == "tree"
        patrol.tabs = Mock(return_value=[(671, 861), None, SimpleNamespace(box=(400, 822, 100, 32))])
        patrol.ocr = Mock(return_value=[SimpleNamespace(text="资源产出效率")])
        patrol.shot = Mock(side_effect=["panel", "interrupt", "refund", "result", "panel", "tree"])
        patrol.result_footer = Mock(side_effect=lambda frame: (360, 1210) if frame == "refund" else (360, 1205))
        patrol.click = Mock()
        patrol.pause = Mock()

        self.assertEqual(patrol.upgrade_room("panel"), ("tree", True))
        self.assertEqual(patrol.click.call_args_list, [
            call((450, 838)), call((574, 1190)), call((505, 785)),
            call((360, 1210)), call((360, 1205)), call((671, 861)),
        ])


if __name__ == "__main__":
    unittest.main()
