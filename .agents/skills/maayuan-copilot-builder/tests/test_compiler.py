# -*- coding: utf-8 -*-
"""
Unit tests for MaaYuan Copilot Compiler
Covers:
- Order vs Slot separation
- Multi-action cell token splitting
- Multi-segment rounds (e.g. 牢狱 / 再动 mechanics)
- Order scoping per segment (restart allowed across segments, duplicates within segment rejected)
- Execution chain continuity across segments
- MaaFramework post_pipeline loading with 10+ actions per round
- Real strategy fixture with 12回合->牢狱->13回合 and 18回合->牢狱->19回合
"""

import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import compiler
from compiler import (
    ValidationError,
    parse_cell_tokens,
    classify_note,
    validate_ir,
    compile_ir_to_copilot,
    assign_missing_orders,
    NoteTier
)


class TestMaaYuanCompiler(unittest.TestCase):

    def test_single_round_five_actions(self):
        """1. Single round with 5 actions maintains standard behavior."""
        ir = {
            "title": "单回合5动测试",
            "stage_name": "test_r1",
            "rounds": [
                {
                    "round": 1,
                    "cells": {
                        "1": "1↑",
                        "2": "5↑",
                        "3": "4↑",
                        "4": "3A",
                        "5": "2A"
                    }
                }
            ]
        }
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]

        # Action 1: slot 1
        self.assertIn("1号位", actions["回合1行动1"]["focus"])
        self.assertEqual(actions["回合1行动1"]["next"], ["史子眇sp", "回合1行动2"])

        # Action 2: slot 5 (order 2)
        self.assertIn("5号位", actions["回合1行动2"]["focus"])
        self.assertEqual(actions["回合1行动2"]["next"], ["史子眇sp", "回合1行动3"])

        # Action 5: slot 2 (order 5, last action of single round)
        self.assertIn("2号位", actions["回合1行动5"]["focus"])
        self.assertEqual(actions["回合1行动5"]["next"], ["史子眇sp", "抄作业战斗胜利"])

    def test_two_segments_ten_actions(self):
        """2. Single game round with two segments (5 actions each) compiles to 回合1行动1 ~ 回合1行动10."""
        ir = {
            "title": "两段行动测试",
            "stage_name": "test_two_seg",
            "rounds": [
                {
                    "round": 12,
                    "segments": [
                        {
                            "label": "12回合",
                            "cells": {"1": "2↓", "2": "4↑", "3": "1↓", "4": "5↑", "5": "3↑"}
                        },
                        {
                            "label": "牢狱",
                            "cells": {"1": "2A", "2": "5↓", "3": "1A", "4": "4↓", "5": "3↓"}
                        }
                    ]
                }
            ]
        }
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]

        # Verify all 10 action nodes exist
        for i in range(1, 11):
            self.assertIn(f"回合12行动{i}", actions)

        # There must be only ONE turn recognition node for Round 12
        self.assertIn("检测回合12", actions)
        self.assertNotIn("检测牢狱", actions)
        self.assertNotIn("检测回合13", actions)

    def test_order_restart_across_segments_allowed(self):
        """3. Both segments start order numbering from 1: perfectly valid, no duplicate error."""
        ir = {
            "title": "跨段Order复用测试",
            "stage_name": "test_order_restart",
            "rounds": [
                {
                    "round": 12,
                    "segments": [
                        {
                            "label": "第1轮",
                            "cells": {"1": "1A", "2": "2A", "3": "3A", "4": "4A", "5": "5A"}
                        },
                        {
                            "label": "牢狱再动",
                            "cells": {"1": "1↑", "2": "2↑", "3": "3↑", "4": "4↑", "5": "5↑"}
                        }
                    ]
                }
            ]
        }
        # Must not raise duplicate order error
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]
        self.assertEqual(len([k for k in actions if k.startswith("回合12行动")]), 10)

    def test_duplicate_order_within_same_segment_raises(self):
        """4. Duplicate orders within the SAME segment must raise ValidationError."""
        ir = {
            "title": "同段重复Order报错测试",
            "stage_name": "test_dup_in_seg",
            "rounds": [
                {
                    "round": 12,
                    "segments": [
                        {
                            "label": "正常段",
                            "cells": {"1": "1A", "2": "2A"}
                        },
                        {
                            "label": "错误段",
                            "cells": {"1": "1↑", "2": "1↓"}  # Duplicate order 1!
                        }
                    ]
                }
            ]
        }
        with self.assertRaises(ValidationError) as ctx:
            compile_ir_to_copilot(ir)
        self.assertIn("Duplicate action order 1", str(ctx.exception))

    def test_three_segments_fifteen_actions(self):
        """5. Three segments compile to 15 actions seamlessly (not hardcoded to 10)."""
        ir = {
            "title": "三段15动测试",
            "stage_name": "test_three_segs",
            "rounds": [
                {
                    "round": 5,
                    "segments": [
                        {"label": "段1", "cells": {"1": "1A", "2": "2A", "3": "3A", "4": "4A", "5": "5A"}},
                        {"label": "段2", "cells": {"1": "1A", "2": "2A", "3": "3A", "4": "4A", "5": "5A"}},
                        {"label": "段3", "cells": {"1": "1A", "2": "2A", "3": "3A", "4": "4A", "5": "5A"}}
                    ]
                }
            ]
        }
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]
        for i in range(1, 16):
            self.assertIn(f"回合5行动{i}", actions)

        # Transition checks
        self.assertEqual(actions["回合5行动5"]["next"], ["史子眇sp", "回合5行动6"])
        self.assertEqual(actions["回合5行动10"]["next"], ["史子眇sp", "回合5行动11"])
        self.assertEqual(actions["回合5行动15"]["next"], ["史子眇sp", "抄作业战斗胜利"])

    def test_segment1_last_action_links_to_segment2(self):
        """6. Segment 1's last action must link directly to Segment 2's first action, NOT next round."""
        ir = {
            "title": "段间衔接测试",
            "stage_name": "test_seg_link",
            "rounds": [
                {
                    "round": 12,
                    "segments": [
                        {"label": "12回合", "cells": {"1": "2↓", "2": "4↑", "3": "1↓", "4": "5↑", "5": "3↑"}},
                        {"label": "牢狱", "cells": {"1": "2A", "2": "5↓", "3": "1A", "4": "4↓", "5": "3↓"}}
                    ]
                },
                {
                    "round": 13,
                    "cells": {"1": "3↓", "2": "2A", "3": "1↓", "4": "4A", "5": "5A"}
                }
            ]
        }
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]

        # Action 5 is the last action of segment 1
        act5_next = actions["回合12行动5"]["next"]
        self.assertEqual(act5_next, ["史子眇sp", "回合12行动6"])
        self.assertNotIn("检测回合13", act5_next)
        self.assertNotIn("抄作业战斗胜利-check", act5_next)

    def test_last_segment_last_action_links_to_next_round(self):
        """7. Only the final segment's last action links to victory check and next game round."""
        ir = {
            "title": "末段转入下回合测试",
            "stage_name": "test_last_seg_next",
            "rounds": [
                {
                    "round": 12,
                    "segments": [
                        {"label": "12回合", "cells": {"1": "2↓", "2": "4↑", "3": "1↓", "4": "5↑", "5": "3↑"}},
                        {"label": "牢狱", "cells": {"1": "2A", "2": "5↓", "3": "1A", "4": "4↓", "5": "3↓"}}
                    ]
                },
                {
                    "round": 13,
                    "cells": {"1": "3↓", "2": "2A", "3": "1↓", "4": "4A", "5": "5A"}
                }
            ]
        }
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]

        # Action 10 is the final action of Round 12
        act10_next = actions["回合12行动10"]["next"]
        self.assertEqual(act10_next, ["史子眇sp", "抄作业战斗胜利-check", "检测回合13"])

    def test_maaframework_load_multi_segment_job(self):
        """8. MaaFramework Resource loads a pipeline with 10+ actions in a single round without error."""
        ir = {
            "title": "MaaFW十动载入测试",
            "stage_name": "test_maafw_multi",
            "rounds": [
                {
                    "round": 12,
                    "segments": [
                        {"label": "12回合", "cells": {"1": "2↓", "2": "4↑", "3": "1↓", "4": "5↑", "5": "3↑"}},
                        {"label": "牢狱", "cells": {"1": "2A", "2": "5↓", "3": "1A", "4": "4↓", "5": "3↓"}}
                    ]
                },
                {
                    "round": 13,
                    "cells": {"1": "3↓", "2": "2A", "3": "1↓", "4": "4A", "5": "5A"}
                }
            ]
        }
        copilot_doc = compile_ir_to_copilot(ir)

        import tempfile

        import maa.resource
        res = maa.resource.Resource()
        # 必须用系统临时目录。早先这里写的是字面量 "resource/base/pipeline/..."，
        # 会在 skill 目录下留下一串空目录，还会让 verify_resource_load.py
        # 把那个空壳误认成 MaaYuan 安装根目录。
        with tempfile.TemporaryDirectory() as tmp:
            temp_path = os.path.join(tmp, "test_multi_seg_pipeline.json")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(copilot_doc["actions"], f, indent=2, ensure_ascii=False)

            job = res.post_pipeline(temp_path)
            job.wait()
            self.assertTrue(job.done)
            self.assertIn("检测回合12", res.node_list)
            self.assertIn("回合12行动1", res.node_list)
            self.assertIn("回合12行动5", res.node_list)
            self.assertIn("回合12行动6", res.node_list)
            self.assertIn("回合12行动10", res.node_list)
            self.assertIn("检测回合13", res.node_list)
            self.assertIn("回合13行动5", res.node_list)

    def test_real_strategy_fixture_with_jail_mechanic(self):
        """9. Full real fixture: Round 12 (牢狱) -> Round 13 -> Round 18 (牢狱) -> Round 19."""
        fixture_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'real_strategy_fixture.json')
        with open(fixture_path, 'r', encoding='utf-8') as f:
            fixture = json.load(f)

        copilot = compile_ir_to_copilot(fixture)
        self.assertEqual(copilot["version"], 3)
        self.assertEqual(copilot["stage_name"], "dg_50")

        actions = copilot["actions"]

        # Exactly 4 game round detection nodes: 12, 13, 18, 19
        self.assertIn("检测回合12", actions)
        self.assertIn("检测回合13", actions)
        self.assertIn("检测回合18", actions)
        self.assertIn("检测回合19", actions)
        self.assertNotIn("检测牢狱", actions)

        # Round 12 has 10 actions (segment 1 + 牢狱)
        for i in range(1, 11):
            self.assertIn(f"回合12行动{i}", actions)
        self.assertEqual(actions["回合12行动5"]["next"], ["史子眇sp", "回合12行动6"])
        self.assertEqual(actions["回合12行动10"]["next"], ["史子眇sp", "抄作业战斗胜利-check", "检测回合13"])

        # Round 13 has 5 actions
        for i in range(1, 6):
            self.assertIn(f"回合13行动{i}", actions)
        self.assertEqual(actions["回合13行动5"]["next"], ["史子眇sp", "抄作业战斗胜利-check", "检测回合18"])

        # Round 18 has 10 actions (segment 1 + 牢狱)
        for i in range(1, 11):
            self.assertIn(f"回合18行动{i}", actions)
        self.assertEqual(actions["回合18行动5"]["next"], ["史子眇sp", "回合18行动6"])
        self.assertEqual(actions["回合18行动10"]["next"], ["史子眇sp", "抄作业战斗胜利-check", "检测回合19"])

        # Round 19 is final round: action 5 points to victory
        self.assertEqual(actions["回合19行动5"]["next"], ["史子眇sp", "抄作业战斗胜利"])

    def test_chain_after_same_order_keeps_following_skill(self):
        """3圈后↑ is one order: switch form, then the up-arrow skill."""
        tokens = parse_cell_tokens(slot=1, cell_text="3圈后↑", round_num=9)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["order"], 3)
        self.assertEqual([step["kind"] for step in tokens[0]["actions"]], ["switch_form", "ultimate"])

    def test_round9_chain_flattens_to_seven_actions(self):
        ir = {
            "title": "金乌枝1第9回合",
            "stage_name": "金乌枝1",
            "rounds": [{
                "round": 9,
                "cells": {"1": "3圈后↑", "2": "1↓2↑", "3": "4↑", "4": "5↑", "5": "6A"},
            }, {
                "round": 10,
                "cells": {"1": "1A"},
            }],
        }
        actions = compile_ir_to_copilot(ir)["actions"]
        # 排除「N行动M兜底」节点——它们是切形态识别失败时的坐标兜底分支，
        # 不属于攻略描述的物理动作序列（2026-10-06 新增）。
        names = [
            key for key in actions
            if key.startswith("回合9行动") and not key.endswith("兜底")
        ]
        self.assertEqual(len(names), 7)
        meta = compile_ir_to_copilot(ir)["_observer_metadata"]
        self.assertEqual(meta["回合9行动3"]["source_order"], 3)
        self.assertEqual(meta["回合9行动3"]["subaction_index"], 0)
        self.assertEqual(meta["回合9行动3"]["raw_cell"], "3圈后↑")
        self.assertEqual(meta["回合9行动3"]["action_type"], "SwitchForm")
        self.assertEqual(meta["回合9行动4"]["source_order"], 3)
        self.assertEqual(meta["回合9行动4"]["subaction_index"], 1)
        self.assertEqual(meta["回合9行动4"]["action_type"], "Ultimate")
        self.assertEqual(meta["回合9行动7"]["source_order"], 6)
        self.assertEqual(meta["回合9行动7"]["action_type"], "Attack")
        self.assertEqual(meta["回合9行动4"]["action_spec"]["begin"][:2], [77, 1060])
        self.assertEqual(meta["回合9行动3"]["action_spec"]["action"], "SwitchForm")
        self.assertIn("检测回合10", actions["回合9行动7"]["next"])

    def test_jinwuzhi_ir_round9_is_seven_actions(self):
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "jinwuzhi1_lvbu_ir.json")
        with open(path, encoding="utf-8") as handle:
            ir = json.load(handle)
        copilot = compile_ir_to_copilot(ir)
        names = [
            key for key in copilot["actions"]
            if key.startswith("回合9行动") and not key.endswith("兜底")
        ]
        self.assertEqual(names, [f"回合9行动{i}" for i in range(1, 8)])
        meta = copilot["_observer_metadata"]["回合9行动4"]
        self.assertEqual(meta["raw_cell"], "3圈后↑")
        self.assertEqual(meta["action_type"], "Ultimate")

    def test_misread_ge_is_not_a_skill(self):
        """个 is not the up arrow. Compiling it must fail instead of casting a skill."""
        with self.assertRaises(ValidationError):
            parse_cell_tokens(slot=1, cell_text="3圈后个", round_num=9)

    def test_single_actions_unchanged(self):
        for text, order, kind in (("4↑", 4, "ultimate"), ("5↑", 5, "ultimate"), ("6A", 6, "normal")):
            tokens = parse_cell_tokens(slot=1, cell_text=text, round_num=1)
            self.assertEqual(len(tokens), 1)
            self.assertEqual(tokens[0]["order"], order)
            self.assertEqual(tokens[0]["kind"], kind)

    def test_multi_action_cells(self):
        """Token parser test: 4A9A and 2↑8↓."""
        tokens_3 = parse_cell_tokens(slot=3, cell_text="4A9A", round_num=2)
        self.assertEqual(len(tokens_3), 2)
        self.assertEqual(tokens_3[0]["order"], 4)
        self.assertEqual(tokens_3[1]["order"], 9)

        tokens_2 = parse_cell_tokens(slot=2, cell_text="2↑8↓", round_num=3)
        self.assertEqual(len(tokens_2), 2)
        self.assertEqual(tokens_2[0]["order"], 2)
        self.assertEqual(tokens_2[1]["order"], 8)

    def test_missing_order_validation_error(self):
        """Tokens missing numeric order prefix must raise ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            parse_cell_tokens(slot=1, cell_text="↑", round_num=1)
        self.assertIn("no valid action token", str(ctx.exception))

    def test_invalid_slot_validation_error(self):
        """Slot outside [1, 5] must raise ValidationError."""
        ir = {
            "title": "测试非法Slot",
            "stage_name": "test_slot",
            "rounds": [{"round": 1, "cells": {"6": "1↑"}}]
        }
        with self.assertRaises(ValidationError) as ctx:
            compile_ir_to_copilot(ir)
        self.assertIn("out of range", str(ctx.exception))

    def test_notes_three_tier_classification(self):
        """Tests Tier A, Tier B, Tier C classification logic."""
        note_a = classify_note("开局集火左侧目标")
        self.assertEqual(note_a["tier"], NoteTier.TIER_A)

        note_b = classify_note("阿婵必须出暴击，否则建议重开")
        self.assertEqual(note_b["tier"], NoteTier.TIER_B)
        self.assertFalse(note_b["supported"])

        note_c = classify_note("练度要求：王粲冷却-1，颜良35000血")
        self.assertEqual(note_c["tier"], NoteTier.TIER_C)

    def test_verification_mode_shadow(self):
        """Tests that verification_mode='shadow' compiles combat nodes to ShadowObserver CustomAction."""
        ir = {
            "title": "Shadow模式测试",
            "stage_name": "test_shadow",
            "rounds": [
                {
                    "round": 1,
                    "cells": {
                        "1": "1A",
                        "2": "2↑",
                        "3": "3↓"
                    }
                }
            ]
        }
        copilot = compile_ir_to_copilot(ir, verification_mode="shadow")
        actions = copilot["actions"]

        # 1. Action 1 (Attack): slot 1
        act1 = actions["回合1行动1"]
        self.assertEqual(act1["action"], "Custom")
        self.assertEqual(act1["custom_action"], "ShadowObserver")
        self.assertEqual(act1["post_delay"], 0)
        self.assertEqual(act1["recognition"], "DirectHit")
        param1 = json.loads(act1["custom_action_param"])
        self.assertEqual(param1["slot"], 1)
        self.assertEqual(param1["action_type"], "Attack")
        self.assertEqual(param1["action_def"]["action"], "Click")
        self.assertEqual(param1["original_post_delay"], 6000)

        # 2. Action 2 (Ultimate): slot 2
        act2 = actions["回合1行动2"]
        self.assertEqual(act2["action"], "Custom")
        self.assertEqual(act2["custom_action"], "ShadowObserver")
        self.assertEqual(act2["post_delay"], 0)
        param2 = json.loads(act2["custom_action_param"])
        self.assertEqual(param2["slot"], 2)
        self.assertEqual(param2["action_type"], "Ultimate")
        self.assertEqual(param2["action_def"]["action"], "Swipe")
        self.assertEqual(param2["original_post_delay"], 10000)

        # 3. Action 3 (Defense): slot 3
        act3 = actions["回合1行动3"]
        self.assertEqual(act3["action"], "Custom")
        self.assertEqual(act3["custom_action"], "ShadowObserver")
        self.assertEqual(act3["post_delay"], 0)
        param3 = json.loads(act3["custom_action_param"])
        self.assertEqual(param3["slot"], 3)
        self.assertEqual(param3["action_type"], "Defense")
        self.assertEqual(param3["action_def"]["action"], "Swipe")
        self.assertEqual(param3["original_post_delay"], 6000)

    def test_switch_form_has_template_plus_coordinate_fallback(self):
        """
        切形态节点必须同时具备模板识别与坐标兜底两条路径。

        背景（2026-10-06）：模板 lb_l2h/lb_h2l 是在吕布站1 号位时截的。
        改站 4 号位后相似度只有 0.66~0.74，稳定够不到0.8 阈值，重试 12 次后
        整个任务被判失败。所以现在必须有 fallback_target 兜底节点。
        """
        ir = {
            "stage_name": "切形态兜底验证",
            "title": "切形态兜底验证",
            "details": "",
            "rounds": [
                {"round": 1, "cells": {"1": "1圈"}},
            ],
        }
        actions = compile_ir_to_copilot(ir)["actions"]

        node = actions.get("回合1行动1")
        self.assertIsNotNone(node, "切形态应生成主节点")
        rec = node["recognition"]
        self.assertEqual(rec["type"], "TemplateMatch")

        # 站位专属模板应在模板列表首位
        templates = rec["param"]["template"]
        self.assertTrue(
            templates[0].startswith("copilot/lb_form_s1.png"),
            f"应优先使用按站位截的模板，实际首位是 {templates[0]}",
        )
        # 旧模板保留兜底（两种形态状态都能认）
        self.assertIn("copilot/lb_l2h.png", templates)
        self.assertIn("copilot/lb_h2l.png", templates)

        # ROI 只框本卡牌下方，不应横跨整排
        roi = rec["param"]["roi"]
        self.assertLess(roi[2], 200, "ROI 应按站位收窄，避免误匹配相邻卡牌")

        # 阈值不应高于历史可达分数(~0.74)
        self.assertLessEqual(rec["param"]["threshold"], 0.7)

        # 兜底分支：主节点 next 首位应是兜底节点
        fb_name = f"回合1行动1兜底"
        self.assertIn(fb_name, node["next"], "识别节点 next 首位应挂坐标兜底分支")

        fb = actions.get(fb_name)
        self.assertIsNotNone(fb)
        self.assertEqual(fb["recognition"], "DirectHit")
        self.assertEqual(fb["action"], "Click")
        # 兜底坐标应落在该卡牌下方（y 大于卡牌基线 1060）
        self.assertGreater(fb["target"][1], 1060)

        # 兜底节点必须能继续往下走，不能是死路。
        # 本例只有 1 个动作，切形态即末动，所以应走向战斗胜利判定。
        self.assertTrue(fb["next"], "兜底节点不能没有后继")
        self.assertIn("抄作业战斗胜利", fb["next"])

    def test_multi_symbol_cell_expands_to_repeated_actions(self):
        """
        「数字+多符号」应展开成同一order 下的多个连续物理动作。

        背景（2026-10-06）：攻略里3 号位（夏侯渊）在 3、4 回合能连打8~9 次，
        原表写作「五锤大其余A」这种条件型描述，无法在战中判断，
        因此按用户实测固化：第3回合 大AAA大AAA、第4回合 A大AAA大AAA。
        """
        # ---- 解析层 ----
        r3 = parse_cell_tokens(3, "5↑AAA↑AAA", 3)
        self.assertEqual(len(r3), 1, "应只有一个 order")
        self.assertEqual(r3[0]["order"], 5)
        kinds = [s["kind"] for s in r3[0]["actions"]]
        self.assertEqual(
            kinds,
            ["ultimate", "normal", "normal", "normal",
             "ultimate", "normal", "normal", "normal"],
            "第3回合应展开为大AAA大AAA 共 8 个动作",
        )

        r4 = parse_cell_tokens(3, "4A↑AAA↑AAA", 4)
        self.assertEqual(len(r4), 1)
        self.assertEqual(r4[0]["order"], 4)
        kinds4 = [s["kind"] for s in r4[0]["actions"]]
        self.assertEqual(
            kinds4,
            ["normal", "ultimate", "normal", "normal", "normal",
             "ultimate", "normal", "normal", "normal"],
            "第4回合应展开为A大AAA大AAA 共 9 个动作",
        )

        # ---- 编译层：order 决定流水位置，跨 order 不会被拆乱 ----
        ir = {
            "stage_name": "多符号展开验证",
            "title": "多符号展开验证",
            "details": "",
            "rounds": [
                {"round": 1, "cells": {"1": "1↑", "3": "2↑AAA↑AAA"}},
            ],
        }
        actions = compile_ir_to_copilot(ir)["actions"]
        # order1(slot1) 先，order2(slot3) 的 8 步紧随其后
        self.assertEqual(actions["回合1行动1"]["text_doc"], "1大")
        seq = [actions[f"回合1行动{i}"]["text_doc"] for i in range(2, 10)]
        self.assertEqual(seq, ["3大", "3普", "3普", "3普", "3大", "3普", "3普", "3普"])
        # 总共 1 + 8 = 9 个动作节点
        real = [k for k in actions if k.startswith("回合1行动") and not k.endswith("兜底")]
        self.assertEqual(len(real), 9)


class TestAssignMissingOrders(unittest.TestCase):
    """
    攻略表格常常只标部分行动序号，未标序号的格子要自动补齐（用户 2026-10-06 确认）。

    规则：
    - 已标序号的格子永远保持原样，包括「某人必须最后一个行动」；
    - 未标序号的格子按站位从左到右，依次填入剩余的空序号（从小到大）；
    - 空格（退场/待实测）不占用任何序号。
    """

    def test_labelled_orders_are_never_renumbered(self):
        """全部标了序号的回合，补序后必须原封不动。"""
        cells = {"1": "1↑", "2": "4↓", "3": "5↓", "4": "3↓", "5": "2↓"}
        self.assertEqual(assign_missing_orders(cells, 4), cells)

    def test_unlabelled_fill_left_to_right(self):
        """第 1 回合：2号位=1、3号位=2 已标；1/4/5 号位补成 3/4/5。"""
        got = assign_missing_orders(
            {"1": "↓", "2": "1↓", "3": "2A", "4": "↓", "5": "↓"}, 1
        )
        self.assertEqual(got, {"1": "3↓", "2": "1↓", "3": "2A", "4": "4↓", "5": "5↓"})

    def test_late_labelled_order_survives(self):
        """「某人必须最后行动」：4 号位已占 order 1，其余从 2 起补。"""
        got = assign_missing_orders({"1": "A", "2": "B", "3": "C", "4": "1↑", "5": "D"}, 2)
        self.assertEqual(got["4"], "1↑")
        self.assertEqual(got["1"], "2A")
        self.assertEqual(got["2"], "3B")
        self.assertEqual(got["3"], "4C")
        self.assertEqual(got["5"], "5D")

    def test_empty_cell_consumes_no_order(self):
        """空格不占序号；后面的已标序号不该被挪动。"""
        got = assign_missing_orders({"1": "1A", "3": "2B"}, 6)
        self.assertEqual(got, {"1": "1A", "3": "2B"})

    def test_duplicate_labelled_order_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            assign_missing_orders({"1": "2A", "2": "2B"}, 1)
        self.assertIn("Duplicate action order 2", str(ctx.exception))

    def test_order_beyond_actor_count_raises(self):
        """已标序号超出本段实际 order 数 => 表格本身漏了格子，必须报错。"""
        with self.assertRaises(ValidationError) as ctx:
            assign_missing_orders({"1": "1A", "2": "5B", "3": "C"}, 1)
        self.assertIn("exceeds the number of", str(ctx.exception))

    def test_multi_order_cell_is_not_counted_as_one(self):
        """
        一个格子可以含多个 order（1↓2↑ = 两个动作）。
        所以「本段 order 总数」不能按格子数算，否则金乌枝第9回合（5 格但有 6 个
        order）会被误判为序号越界。回归锁死这一点。
        """
        cells = {"1": "3圈后↑", "2": "1↓2↑", "3": "4↑", "4": "5↑", "5": "6A"}
        self.assertEqual(assign_missing_orders(cells, 9), cells)

    def test_order_exceeding_actor_count_raises_in_compile(self):
        """IR 整体校验也要拦住「序号 6 但本段只有 2 个 order」这种表错。"""
        ir = {
            "title": "序号越界测试",
            "stage_name": "test_order_overflow",
            "rounds": [{"round": 1, "cells": {"1": "1A", "2": "6B"}}],
        }
        with self.assertRaises(ValidationError):
            compile_ir_to_copilot(ir)

    def test_kaiju_rounds_match_expected_order(self):
        """
        锁定「六连速刷」补序 + 分段后的实际执行顺序，防止回归。

        ⚠️ 表格有 6 行，但游戏内只有 5 个回合：第 5、6 行是同一回合内的
        「回溯」两段（用户 2026-10-06 澄清）。所以第 5 回合是 1 段12 动+
        2 段 12 动 = 24 动，且不存在第 6 回合。
        """
        expected = {
            1: [2, 3, 1, 4, 5],
            2: [1, 2, 3, 4, 5],
            3: [2, 3, 1, 4, 5],
            4: [1, 5, 4, 2, 3],
            #第一段 12 动 + 第二段（回溯）12 动
            5: [2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 1,
                1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5],
        }
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "ruyuan_kaiju_ir.json")
        with open(path, encoding="utf-8") as handle:
            ir = json.load(handle)
        copilot = compile_ir_to_copilot(ir)
        meta = copilot["_observer_metadata"]
        for rnd, slots in expected.items():
            names = [k for k in copilot["actions"] if k.startswith(f"回合{rnd}行动")]
            names.sort(key=lambda k: meta[k]["action_index"])
            self.assertEqual([meta[k]["slot"] for k in names], slots, f"第{rnd}回合顺序不符")

    def test_kaiju_table_row6_is_segment_not_round6(self):
        """
        回归：表格第 5、6 行是同回合内的两段，绝不能编成两个独立回合。

        编错会多生成一个 `检测回合6`，而游戏回合数全程停在 5，
        该节点 20 秒后必然 Task timeout（2026-10-06 实测事故）。
        """
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "ruyuan_kaiju_ir.json")
        with open(path, encoding="utf-8") as handle:
            ir = json.load(handle)
        copilot = compile_ir_to_copilot(ir)
        actions = copilot["actions"]

        # 只应有5 个回合检测节点
        checks = sorted(int(k[len("检测回合"):]) for k in actions if k.startswith("检测回合"))
        self.assertEqual(checks, [1, 2, 3, 4, 5])
        self.assertNotIn("检测回合6", actions)
        self.assertNotIn("回合6行动1", actions)

        # 第 5 回合下挂两个 segment，24 个动作序号连续（段间不插检测）
        meta = copilot["_observer_metadata"]
        r5 = sorted(
            [k for k in actions if k.startswith("回合5行动") and not k.endswith("兜底")],
            key=lambda k: meta[k]["action_index"],
        )
        self.assertEqual(len(r5), 24)
        self.assertEqual([meta[k]["action_index"] for k in r5], list(range(1, 25)))
        segments = {meta[k]["segment"] for k in r5}
        self.assertEqual(len(segments), 2, "第5回合应有两个行动段")

        # 段与段之间是直接衔接，不经过任何回合检测
        self.assertEqual(actions["回合5行动12"]["next"], ["史子眇sp", "回合5行动13"])

        # 第 5 回合是最后一回合：最后一个动作直接接战斗胜利
        self.assertEqual(actions["回合5行动24"]["next"], ["史子眇sp", "抄作业战斗胜利"])


class TestRecordOnlyNotes(unittest.TestCase):
    """Tier D：像命令但作业做不到的旁注，只记录、不生成动作、不报不支持。"""

    def test_enemy_ai_and_retry_and_loadout_notes(self):
        for text in ("默念别打诸葛亮", "回溯回合", "带星石"):
            self.assertEqual(classify_note(text)["tier"], NoteTier.TIER_D, text)
            self.assertFalse(classify_note(text)["generates_action"])

    def test_stat_requirements_stay_tier_c(self):
        """练度/冷却数值是普通策略建议，不能被 Tier D 吃掉。"""
        self.assertEqual(classify_note("练度要求：王粲冷却-1").get("tier"), NoteTier.TIER_C)

    def test_tier_d_note_produces_no_action_and_no_warning(self):
        ir = {
            "title": "仅记录旁注测试",
            "stage_name": "test_tier_d",
            "details": "基础说明",
            "rounds": [{
                "round": 1,
                "cells": {"1": "1A", "2": "2A"},
                "notes": ["默念别打诸葛亮", "回溯回合", "带星石"],
            }],
        }
        copilot = compile_ir_to_copilot(ir)
        details = copilot["doc"]["details"]
        self.assertIn("仅记录，不影响本作业执行", details)
        self.assertIn("默念别打诸葛亮", details)
        # 不能出现「需人工留意」这类条件分支警告
        self.assertNotIn("需人工留意", details)
        # 旁注不得凭空多出动作节点：仍然是 2 个
        real = [k for k in copilot["actions"]
                if k.startswith("回合1行动") and not k.endswith("兜底")]
        self.assertEqual(len(real), 2)

    def test_kaiju_round5_expands_to_eight_actions(self):
        """第 5 回合 3 号位两段各8 连招，节奏不同：第一段大AAA大大AA，第二段大AAA大AAA。"""
        # 第一段「大AAA / 大大AA」
        first = parse_cell_tokens(3, "4↑AAA↑↑AA", 5)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["order"], 4)
        self.assertEqual(
            [s["kind"] for s in first[0]["actions"]],
            ["ultimate", "normal", "normal", "normal",
             "ultimate", "ultimate", "normal", "normal"],
            "应展开为大AAA大大AA 共 8 个动作",
        )
        # 第二段（回溯）「大AAA / 大AAA」
        second = parse_cell_tokens(3, "3↑AAA↑AAA", 5)
        self.assertEqual(len(second), 1)
        self.assertEqual(
            [s["kind"] for s in second[0]["actions"]],
            ["ultimate", "normal", "normal", "normal",
             "ultimate", "normal", "normal", "normal"],
            "应展开为大AAA大AAA 共 8 个动作",
        )

    def test_kaiju_second_segment_slot3_fills_the_gap(self):
        """
        第二段 3 号位补上后，顺序号 1~5 每个站位各一次，无重号无缺号。

        表格该格原本留空（当时写作「待实测」），用户2026-10-06 补充为
        「大AAA大AAA」后，1~5 号位每人都恰好一个 order。
        """
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "ruyuan_kaiju_ir.json")
        with open(path, encoding="utf-8") as handle:
            ir = json.load(handle)
        copilot = compile_ir_to_copilot(ir)
        meta = copilot["_observer_metadata"]
        r5 = [k for k in copilot["actions"] if k.startswith("回合5行动")]
        r5.sort(key=lambda k: meta[k]["action_index"])

        retrace = [k for k in r5 if meta[k]["segment"] == "回溯"]
        # 12 个动作：1号位A / 2号位A / 3号位8连招 / 4号位大 / 5号位大
        self.assertEqual(len(retrace), 12)
        self.assertEqual([meta[k]["slot"] for k in retrace],
                         [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5])
        # 三个非夏侯渊动作的 order 分别是 1 / 2 / 4 / 5，夏侯渊独占 order 3
        singles = [meta[k]["source_order"] for k in retrace if meta[k]["slot"] != 3]
        self.assertEqual(singles, [1, 2, 4, 5])
        self.assertEqual({meta[k]["source_order"] for k in retrace if meta[k]["slot"] == 3}, {3})


class TestCacheExport(unittest.TestCase):
    def test_export_does_not_create_active_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'resource'
            output = root / 'copilot-cache/new.json'
            doc = {'version': 3, 'actions': {'test': {'action': 'DoNothing'}}, '_observer_metadata': {'test': {'slot': 1}}}
            compiler.export_copilot_json(doc, str(output))
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), doc)
            self.assertFalse((root / 'base').exists())

    def test_export_preserves_active_job_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'resource'
            active = root / 'base/pipeline/copilot/active.json'
            sidecar = root / 'base/pipeline_metadata/copilot/new.observer_metadata.json'
            for path in (active, sidecar):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('existing', encoding='utf-8')
            compiler.export_copilot_json({'actions': {}, '_observer_metadata': {'test': {}}}, str(root / 'copilot-cache/new.json'))
            self.assertEqual(active.read_text(encoding='utf-8'), 'existing')
            self.assertEqual(sidecar.read_text(encoding='utf-8'), 'existing')
            self.assertEqual(list(active.parent.iterdir()), [active])


if __name__ == '__main__':
    unittest.main()
