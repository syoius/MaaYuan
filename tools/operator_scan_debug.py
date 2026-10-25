"""Operator OCR replay and bounded live checks. Run --help for usage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("life", "attack", "level", "cultivation", "operator_id")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path, scope):
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest 必须包含非空 cases 数组")
    result, seen = [], set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("image"), str):
            raise ValueError("每个样本必须提供 image 路径")
        image = (path.parent / case["image"]).resolve()
        if not image.is_file() or image in seen:
            raise ValueError(f"截图不存在或重复: {image}")
        seen.add(image)
        expected = case.get("expected", {})
        if not isinstance(expected, dict) or expected.keys() - set(FIELDS):
            raise ValueError(f"未知标注字段: {expected!r}")
        if scope == "stats" and "operator_id" in expected:
            raise ValueError("operator_id 标注需要 --scope main")
        for key, value in expected.items():
            if key == "operator_id":
                valid = isinstance(value, str) and bool(value.strip())
            else:
                valid = type(value) is int and value >= (0 if key == "cultivation" else 1)
            if not valid:
                raise ValueError(f"标注必须是已确认真值，不能用 null/布尔值: {key}={value!r}")
        result.append({"image": str(image), "sha256": digest(image), "expected": expected})
    return result


def assess(actual, expected):
    return {key: ("missing" if actual.get(key) is None else
                  "correct" if actual[key] == value else "wrong")
            for key, value in expected.items()}


def summarize(rows, modes):
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        counts = {key: 0 for key in ("correct", "missing", "wrong")}
        for row in selected:
            for verdict in row["verdicts"].values():
                counts[verdict] += 1
        checked = sum(counts.values())
        summary[mode] = {
            "frames": len(selected), "labeled_fields": checked, **counts,
            "accuracy_passed": (counts["missing"] == counts["wrong"] == 0) if checked else None,
            "seconds": sum(row["seconds"] for row in selected),
            "ocr_calls": sum(len(row["ocr"]) for row in selected),
        }
    return summary


def mode_differences(rows):
    """Disagreement is a review hint, never evidence that either mode is correct."""
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image"], {})[row["mode"]] = row["actual"]
    differences = []
    for image, readings in by_image.items():
        if not {"normal", "enhanced"} <= readings.keys():
            continue
        fields = {key: {mode: values.get(key) for mode, values in readings.items()}
                  for key in FIELDS if readings["normal"].get(key) != readings["enhanced"].get(key)}
        if fields:
            differences.append({"image": image, "differences": fields})
    return differences


def fingerprints(resource):
    paths = [Path(__file__), ROOT / "tools/operator_scan_live.py", ROOT / "agent/custom/action/agent_info_collector.py",
             ROOT / "agent/custom/action/operator_scan_diagnostics.py",
             ROOT / "agent/custom/action/operator_growth_exchange.py", ROOT / "agent/operators.json"]
    paths += sorted((resource / "pipeline").rglob("*.json"))
    paths += sorted(p for p in (resource / "model/ocr").rglob("*") if p.is_file())
    return {str(path): digest(path) for path in paths}


def run_agent(run, identifier):
    import cv2
    import numpy as np
    from maa.agent.agent_server import AgentServer
    from maa.custom_action import CustomAction
    from maa.toolkit import Toolkit
    from unittest.mock import patch

    sys.path.insert(0, str(ROOT / "agent"))
    from custom.action import agent_info_collector as collector
    cv2.setNumThreads(1)
    config = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    Toolkit.init_option(str(run / "agent"))
    rows = []
    callback = {"success": False, "error": "回调未完成"}

    class Replay(CustomAction):
        def run(self, context, argv):
            if config.get("live"):
                from operator_scan_live import run_live
                try:
                    state = run_live(context, collector, config, run)
                    callback.update(success=True, error=None, summary={config["live"]: state})
                except Exception:
                    state = json.loads((run / "live-result.json").read_text(encoding="utf-8")) if (run / "live-result.json").is_file() else {}
                    callback.update(success=False, error=traceback.format_exc(), summary={config["live"]: state})
                write_json(run / "callback.json", callback)
                return CustomAction.RunResult(success=callback["success"])
            try:
                catalog = json.loads((ROOT / "agent/operators.json").read_text(encoding="utf-8"))
                calls = []

                class ReadOnlyContext:
                    tasker = context.tasker

                    def run_recognition(self, node, image, override):
                        start = time.perf_counter()
                        result = context.run_recognition(node, image, override)
                        calls.append({"node": node, "text": collector._joined_text(result),
                                      "seconds": time.perf_counter() - start})
                        return result

                # Prevent the production reader's normal remote catalogue refresh.
                with patch.object(collector, "refresh_operator_catalog", return_value=catalog):
                    readers = {mode: collector._AgentInfoReader(ReadOnlyContext(), {
                        "game": config["game"], "resource": config["resource"], "scan_mode": mode,
                    }) for mode in config["modes"]}
                with (run / "rows.jsonl").open("w", encoding="utf-8") as output:
                    for index, case in enumerate(config["cases"]):
                        path = Path(case["image"])
                        if digest(path) != case["sha256"]:
                            raise RuntimeError(f"截图在回放前发生变化: {path}")
                        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                        if image is None or image.shape[:2] != (1280, 720):
                            raise ValueError(f"需要 720×1280 竖屏截图: {path}")
                        for mode, reader in readers.items():
                            calls.clear()
                            start = time.perf_counter()
                            if config["scope"] == "main":
                                main = reader._read_main(image)
                                actual = {**main["stats"], "operator_id": main["operator_id"]}
                            else:
                                actual = {key: reader._read_stat(image, key, roi)
                                          for key, roi in reader._roi_config["main_stats"].items()}
                            row = {"image": str(path), "mode": mode, "actual": actual,
                                   "expected": case["expected"], "verdicts": assess(actual, case["expected"]),
                                   "seconds": time.perf_counter() - start, "ocr": list(calls)}
                            if any(value != "correct" for value in row["verdicts"].values()):
                                evidence = run / f"failure-{index:04d}{path.suffix}"
                                if not evidence.exists():
                                    shutil.copyfile(path, evidence)
                                row["evidence"] = evidence.name
                            rows.append(row)
                            output.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output.flush()
                        if (index + 1) % 10 == 0:
                            print(f"回放 {index + 1}/{len(config['cases'])}", flush=True)
                callback.update(success=True, error=None)
            except Exception:
                callback.update(success=False, error=traceback.format_exc())
            differences = mode_differences(rows)
            write_json(run / "mode-differences.json", differences)
            callback["mode_disagreement_frames"] = len(differences)
            callback["summary"] = summarize(rows, config["modes"])
            write_json(run / "callback.json", callback)
            return CustomAction.RunResult(success=callback["success"])

    AgentServer.register_custom_action("OperatorOfflineReplay", Replay())
    AgentServer.start_up(identifier)
    AgentServer.join()
    AgentServer.shut_down()
    return 0


def run_worker(run):
    import numpy as np
    import maa
    from maa.library import Library
    Library.open(Path(maa.__file__).parent / "bin", agent_server=False)
    from maa.resource import Resource
    from maa.tasker import Tasker
    from maa.toolkit import Toolkit
    from maa.controller import CustomController
    from maa.agent_client import AgentClient
    from maa.define import LoggingLevelEnum

    config = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    Toolkit.init_option(str(run / "framework"))
    Tasker.set_stdout_level(LoggingLevelEnum.Error)
    resource = Resource()
    resource.use_cpu()
    if not resource.post_bundle(config["resource"]).wait().status.succeeded:
        raise RuntimeError("资源加载失败；请先运行 configure.py 准备 OCR 模型")

    def forbidden(*args):
        raise RuntimeError("离线回放禁止设备操作")

    methods = {name: forbidden for name in dir(CustomController)
               if getattr(getattr(CustomController, name), "__isabstractmethod__", False)}
    methods.update(connect=lambda self: True, request_uuid=lambda self: "operator-offline",
                   screencap=lambda self: np.zeros((1280, 720, 3), np.uint8))
    if config.get("live"):
        from maa.controller import AdbController
        from maa.define import MaaAdbScreencapMethodEnum
        controller = AdbController(config["adb_path"], config["serial"],
                                   config["screencap_method"] or MaaAdbScreencapMethodEnum.Default,
                                   config["input_method"], {})
    else:
        controller = type("OfflineController", (CustomController,), methods)()
    if not controller.post_connection().wait().status.succeeded:
        raise RuntimeError("控制器初始化失败")
    client = AgentClient()
    if not client.bind(resource):
        raise RuntimeError("Agent 资源绑定失败")
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(run),
                              "--identifier", client.identifier])
    try:
        if not client.connect():
            raise RuntimeError("Agent 连接失败")
        tasker = Tasker()
        if not tasker.bind(resource, controller):
            raise RuntimeError("离线 Tasker 绑定失败")
        job = tasker.post_task("OperatorOfflineReplay", {"OperatorOfflineReplay": {
            "action": {"type": "Custom", "param": {"custom_action": "OperatorOfflineReplay"}}
        }})
        job.wait()
        callback = json.loads((run / "callback.json").read_text(encoding="utf-8"))
        success = job.status.succeeded and callback["success"]
        write_json(run / "worker-result.json", {"execution_success": success,
                   "summary": callback["summary"], "error": callback["error"],
                   "mode_disagreement_frames": callback.get("mode_disagreement_frames")})
        return 0 if success else 2
    finally:
        client.disconnect()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def supervise(command, log, timeout, env):
    # A separate process group lets timeout/interrupt stop the native worker and its Agent.
    import signal
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=os.name != "nt")
    try:
        return child.wait(timeout=timeout)
    finally:
        if os.name != "nt":
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif child.poll() is None:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                           stdout=log, stderr=subprocess.STDOUT, check=False)
        child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="密探离线回放与定点实机测试；不上传、不更新名册。",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog='''
清单格式（image 相对清单目录，expected 可省略或只标注部分字段）：
  {"cases": [{"image": "frame.png", "expected": {"life": 100, "level": 2}}]}
支持标注 life / attack / level / cultivation；main 额外支持 operator_id。
仅支持 720×1280 主界面截图；main 仅做姓名识别，不模拟命盘身份证据。
示例：.venv/bin/python tools/operator_scan_debug.py --manifest cases.json --mode both
输出：manifest.json、fingerprints.json、rows.jsonl、summary.json、失败截图和日志。
未标注字段不计准确率；缺失/误值使退出码为 1，执行失败/超时为 2，其他为 0。
耗时仅覆盖离线识别，不代表全量实机耗时。所有产物默认保存在 debug/ 下。
实机示例：--live navigation --adb-path /path/to/adb --serial DEVICE --steps 2
实机需先打开密探主界面；current 采集当前密探，navigation 相邻往返，disc 切换并恢复命盘。
smoke 自动依次测试当前采集、翻页往返，并在当前位置及最多 steps 位相邻密探中找双盘后返回。
每次实机会生成无标注 replay.json：--manifest RUN/replay.json --scope main --mode both
observations.json 是本次 OCR 观察值，不是人工真值；离线不会据此计算准确率。
双模式回放会生成 mode-differences.json，列出读数不同的截图和字段，供人工核对。
实机默认 enhanced，不支持 both；不查找固定人物或返回名单首位。超时可能无法恢复，
须检查 live-result.json 的 restored（true 已确认，null 未确认，false 恢复后身份不符）。
实机仅验证动作与身份一致性，不宣称无标注数值正确；单人名单/单套命盘会标记跳过。
''')
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("normal", "enhanced", "both"))
    parser.add_argument("--live", choices=("current", "navigation", "disc", "smoke"))
    parser.add_argument("--adb-path", type=Path, help="实机测试使用的 adb 可执行文件")
    parser.add_argument("--serial", help="实机设备地址/序列号，不自动选择设备")
    parser.add_argument("--steps", type=int, default=2, help="向前翻页步数 1–5，随后按实际路径返回")
    parser.add_argument("--input-method", type=int, default=4, help="MaaFramework 输入方法位掩码，默认 4（Maatouch）")
    parser.add_argument("--screencap-method", type=int, help="截图方法位掩码，默认 MaaFramework Default")
    parser.add_argument("--scope", choices=("stats", "main"), default="stats")
    parser.add_argument("--game", choices=("代号鸢", "如鸢"), help="默认采用回放清单设置，否则代号鸢")
    parser.add_argument("--resource", type=Path, help="默认采用回放清单设置，否则 assets/resource/base")
    parser.add_argument("--output", type=Path, help="新建输出目录，拒绝覆盖已有目录")
    parser.add_argument("--timeout", type=int, default=180, help="整个回放子进程的秒数上限，默认 180")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--identifier", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return run_agent(args.worker, args.identifier) if args.identifier else run_worker(args.worker)
    if bool(args.manifest) == bool(args.live) or args.timeout <= 0:
        parser.error("--manifest 与 --live 必须且只能指定一个；--timeout 必须大于 0")
    if args.live and (not args.adb_path or not args.serial or args.mode == "both"
                     or not 1 <= args.steps <= 5 or not 15 <= args.timeout <= 300
                     or args.input_method <= 0 or (args.screencap_method is not None and args.screencap_method <= 0)):
        parser.error("实机需要 --adb-path 和 --serial；单模式，steps 1–5，timeout 15–300 秒，方法位掩码为正数")
    run = None
    try:
        cases = load_cases(args.manifest.resolve(), args.scope) if args.manifest else []
        if args.live and not args.adb_path.is_file():
            raise ValueError("adb 可执行文件不存在")
        metadata = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
        game = args.game or metadata.get("game", "代号鸢")
        if game not in ("代号鸢", "如鸢"):
            raise ValueError("清单包含不支持的 game")
        if "resource" in metadata and not isinstance(metadata["resource"], str):
            raise ValueError("清单 resource 必须是资源目录路径字符串")
        resource = args.resource or Path(metadata.get("resource", ROOT / "assets/resource/base"))
        if args.resource is None and "resource" in metadata and not resource.is_absolute():
            resource = args.manifest.resolve().parent / resource
        resource = resource.resolve()
        if not (resource / "pipeline/agent_info_collector.json").is_file():
            raise ValueError("资源目录缺少 agent_info_collector.json")
        mode = args.mode or ("enhanced" if args.live else "both")
        modes = ["normal", "enhanced"] if mode == "both" else [mode]
        target = args.output or ROOT / "debug/operator-scan-debug" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=False)
        run = target
        config = {"cases": cases, "resource": str(resource), "modes": modes,
                  "game": game, "scope": args.scope}
        if args.live:
            config.update(live=args.live, adb_path=str(args.adb_path.resolve()), serial=args.serial,
                          input_method=args.input_method, screencap_method=args.screencap_method,
                          steps=args.steps, timeout=args.timeout)
        write_json(run / "manifest.json", config)
        before = fingerprints(resource)
        write_json(run / "fingerprints.json", before)
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        with (run / "worker.log").open("w", encoding="utf-8") as log:
            returncode = supervise([sys.executable, str(Path(__file__).resolve()), "--worker", str(run)],
                                   log, args.timeout, env)
        if not (run / "worker-result.json").is_file():
            raise RuntimeError(f"回放子进程未生成结果（退出码 {returncode}），见 worker.log")
        result = json.loads((run / "worker-result.json").read_text(encoding="utf-8"))
        if returncode != 0 or not result["execution_success"]:
            raise RuntimeError(result.get("error") or "原生回放失败，见 worker.log")
        if before != fingerprints(resource) or any(digest(Path(c["image"])) != c["sha256"] for c in cases):
            raise RuntimeError("源码、资源或截图在回放期间发生变化，结果无效")
        write_json(run / "summary.json", result)
        if args.live:
            state = result["summary"][args.live]
            print(f"实机 {args.live}: 场景通过={state['scenario_passed']}, 恢复={state['restored']}, 耗时={state['seconds']:.1f}s")
            for name, check in state.get("checks", {}).items():
                print(f"  {name}: 通过={check.get('scenario_passed')}, 恢复={check.get('restored')}, 跳过={check.get('skipped', '无')}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"结果目录: {run}")
        return 1 if any(s["accuracy_passed"] is False for s in result["summary"].values()) else 0
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        if run:
            failure = {"execution_success": False, "error": str(exc)}
            if args.live:
                state_path = run / "live-result.json"
                failure["live"] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"restored": None}
            write_json(run / "summary.json", failure)
        print(f"回放失败: {exc}\n结果目录: {run}", file=sys.stderr)
        return 2
    finally:
        if run and (run / "replay.json").is_file():
            command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(run / "replay.json"),
                       "--scope", "main", "--mode", "both"]
            print("离线复现（不连接设备）: " + (subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)))


if __name__ == "__main__":
    raise SystemExit(main())
