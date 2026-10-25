"""Bounded live scenarios for operator_scan_debug; no upload or roster assumptions."""
import json
import time


def run_scenario(reader, scenario, steps, state, save, normalize):
    """Keep restoration evidence separate from whether the scenario succeeded."""
    if scenario == "smoke":
        state.update(checks={}, restored=True, phase="smoke")
        for check in ("current", "navigation", "disc-search"):
            child = {"restored": None}
            state["checks"][check] = child

            def save_child():
                state["restored"] = child.get("restored")
                state["phase"] = f"{check}:{child.get('phase', 'initialize')}"
                save()

            run_scenario(reader, check, steps, child, save_child, normalize)
            if "initial_id" not in state:
                state["initial_id"] = child["initial_id"]
            elif child["initial_id"] != state["initial_id"]:
                state["restored"] = False
                raise RuntimeError("短程测试各场景的起点身份不一致")
        skipped = [name for name, child in state["checks"].items() if child.get("skipped")]
        state.update(success=True, phase="complete", accuracy_passed=None,
                     scenario_passed=None if skipped else True, skipped_checks=skipped)
        save()
        return

    def read_main():
        reader._page_image = None
        main = reader._read_main(reader._require_page("main"))
        if not main.get("operator_id"):
            raise RuntimeError("当前密探身份未确认，停止定点测试")
        return main

    def label():
        value = normalize(reader.ocr_text(reader._require_page("disc"), (180, 142, 101, 42)))
        if not value.startswith("命盘"):
            raise RuntimeError(f"命盘标签未确认: {value!r}")
        return value

    main = read_main()
    initial = main["operator_id"]
    state.update(initial_id=initial, observed_ids=[initial], restored=True, phase="ready")
    save()
    if scenario == "current":
        state.update(restored=None, phase="collect_current")
        save()
        record = reader.collect_current_from_main(main)
        state["record"] = record
        final = read_main()
        state["restored"] = final["operator_id"] == initial
        save()
        if not record or record.get("operator_id") != initial or not state["restored"]:
            raise RuntimeError("采集前后身份不一致或返回主界面未确认")
    elif scenario in ("navigation", "disc-search"):
        direction = "next" if reader._operator_arrow_visible(reader._require_page("main"), "next") else "previous"
        path = state["observed_ids"]
        known_position = True
        try:
            for offset in range(steps + 1):
                if scenario == "disc-search":
                    attempt = {"restored": None}
                    state.setdefault("attempts", []).append(attempt)
                    known_position = False
                    state.update(restored=None, phase="probe_disc")
                    save()
                    try:
                        run_scenario(reader, "disc", steps, attempt, save, normalize)
                    finally:
                        known_position = attempt.get("restored") is True
                    if attempt.get("switched"):
                        state["tested_operator_id"] = main["operator_id"]
                        break
                if offset == steps:
                    break
                state.update(restored=None, phase=f"move_{direction}")
                save()
                known_position = False
                if not reader._move_operator(direction, main):
                    known_position = True
                    state["boundary_reached"] = True
                    break
                main = read_main()
                if main["operator_id"] in path:
                    raise RuntimeError("翻页后身份重复，不能确认当前位置")
                path.append(main["operator_id"])
                known_position = True
                save()
        finally:
            # An unconfirmed movement cannot be undone safely by guessing a number of clicks.
            if known_position:
                state["phase"] = "restore_navigation"
                save()
                reverse = "previous" if direction == "next" else "next"
                for expected in reversed(path[:-1]):
                    if not reader._move_operator(reverse, main):
                        raise RuntimeError("返回路径意外到达边界，恢复失败")
                    main = read_main()
                    if main["operator_id"] != expected:
                        raise RuntimeError("返回路径身份不符，恢复失败")
                state["restored"] = main["operator_id"] == initial
                save()
        state["transitions"] = 2 * (len(path) - 1)
        if scenario == "disc-search" and not state.get("tested_operator_id"):
            state["skipped"] = "有界搜索中未找到双套命盘；未宣称双盘切换通过"
        elif scenario == "navigation" and len(path) == 1:
            state["skipped"] = "当前位置没有相邻密探，未验证翻页"
    elif scenario == "disc":
        state.update(restored=None, phase="enter_disc")
        save()
        reader.click(reader._roi_config["clicks"]["disc_entry"], reader._require_page("main"), settle_ms=0)
        original = label()
        state.update(original_label=original, phase="switch_disc")
        save()
        try:
            switched = reader._toggle_disc(original)
            current = label()
            state.update(switched=switched, switched_label=current)
            if switched and current == original:
                raise RuntimeError("命盘切换后标签未变化")
            if not switched:
                state["skipped"] = "仅一套可用命盘；已验证提示关闭，未验证双盘切换"
        finally:
            state["phase"] = "restore_disc"
            save()
            current = label()
            if current != original:
                if not reader._toggle_disc(current):
                    raise RuntimeError("原命盘恢复切换未执行")
            state["restored_label"] = label()
            if state["restored_label"] != original:
                raise RuntimeError("原命盘标签恢复失败")
            reader.click(reader._roi_config["clicks"]["disc_back"], reader._require_page("disc"), settle_ms=0)
            state["restored"] = read_main()["operator_id"] == initial
            save()
            if not state["restored"]:
                raise RuntimeError("命盘返回后密探身份不符")
    else:
        raise ValueError(f"未知实机场景: {scenario}")
    state.update(success=True, phase="complete", accuracy_passed=None,
                 scenario_passed=None if state.get("skipped") else True)
    save()


def run_live(context, collector, config, run):
    """Record incremental evidence; a hard timeout leaves restored=null on disk."""
    import cv2
    from unittest.mock import patch

    state = {"scenario": config["live"], "mode": config["modes"][0],
             "success": False, "restored": None, "phase": "initialize"}

    def save():
        path = run / "live-result.json"
        temporary = run / "live-result.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    save()
    started = time.monotonic()
    deadline = started + max(1, config["timeout"] - 10)
    replay_cases = {}
    observations = []
    with (run / "events.jsonl").open("w", encoding="utf-8") as events:
        def event(kind, **data):
            events.write(json.dumps({"elapsed": time.monotonic() - started, "kind": kind, **data}, ensure_ascii=False) + "\n")
            events.flush()

        class TracedContext:
            tasker = context.tasker

            def run_recognition(self, node, image, *args):
                result = context.run_recognition(node, image, *args)
                event("recognition", node=node, text=collector._joined_text(result))
                return result

            def run_task(self, node, *args):
                event("task_start", node=node)
                result = context.run_task(node, *args)
                event("task_end", node=node, succeeded=bool(result and result.status.succeeded))
                return result

        catalog = json.loads((collector.REPO_ROOT / "agent/operators.json").read_text(encoding="utf-8"))
        try:
            with patch.object(collector, "refresh_operator_catalog", return_value=catalog):
                reader = collector._AgentInfoReader(TracedContext(), {
                    "game": config["game"], "resource": config["resource"],
                    "scan_mode": config["modes"][0], "scan_single": True,
                    "upload": False, "commit": False, "resume": False,
                })
            ensure_running = reader._ensure_running
            screenshot = reader.screenshot
            click = reader.click
            read_main = reader._read_main
            count = 0
            last_capture = [None, None]

            def check_time():
                ensure_running()
                if time.monotonic() >= deadline:
                    raise TimeoutError("定点测试超时，停止进一步点击；请检查恢复状态")

            def capture():
                nonlocal count
                image = screenshot()
                if image is None:
                    raise RuntimeError("实机截图失败")
                if image.shape[:2] != (1280, 720):
                    raise RuntimeError(f"需要 720×1280 截图，实际为 {image.shape[:2]}")
                count += 1
                name = f"frame-{count:04d}.png"
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise RuntimeError("证据截图编码失败")
                (run / name).write_bytes(encoded.tobytes())
                last_capture[:] = [image, name]
                state["last_frame"] = name
                event("screenshot", frame=name)
                return image

            def observed_main(image):
                # Save the exact input frame, never promote its OCR output to ground truth.
                name = last_capture[1] if image is last_capture[0] else f"main-{len(observations):04d}.png"
                if image is not last_capture[0]:
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise RuntimeError("主界面回放截图编码失败")
                    (run / name).write_bytes(encoded.tobytes())
                replay_cases[name] = {"image": name}
                replay = {"game": config["game"], "resource": config["resource"],
                          "cases": list(replay_cases.values())}
                temporary = run / "replay.tmp"
                temporary.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(run / "replay.json")
                result = read_main(image)
                observations.append({"image": name, "observed": {
                    key: result.get(key) for key in ("operator_id", "name_raw", "stats", "_name_ocr_fallback")}})
                (run / "observations.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
                return result

            def traced_click(point, image=None, settle_ms=None):
                event("click", point=point, frame=state.get("last_frame"))
                return click(point, image, settle_ms)

            reader._ensure_running = check_time
            reader.screenshot = capture
            reader.click = traced_click
            reader._read_main = observed_main
            image = reader.screenshot()
            ready = reader.context.run_recognition(collector.PAGE_NODES["main"], image)
            if not ready or not ready.hit:
                raise RuntimeError("请先打开密探主界面；测试未执行点击")
            reader.current_page = "main"
            reader._page_image = image
            run_scenario(reader, config["live"], config["steps"], state, save, collector._normalise)
        except Exception as exc:
            state.update(success=False, error=str(exc), scenario_passed=False)
            event("failure", error=str(exc))
            raise
        finally:
            state["seconds"] = time.monotonic() - started
            save()
    return state
