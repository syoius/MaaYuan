"""Manual, single-trial native Swipe experiment. `plan`/`analyse` never connect devices."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import uuid
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True


def load_core(repo: Path):
    path = repo / "agent/utils/star_swipe_calibration.py"
    spec = importlib.util.spec_from_file_location("star_swipe_calibration_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_motion_algorithms(repo: Path):
    # The AgentServer import switches the native API; standalone tools must NOT import it.
    # Execute the existing pure functions unchanged, as in the previous offline audit.
    path = repo / "agent/custom/action/star_backpack_capture_probe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StarBackpackCaptureProbe":
            break
        if isinstance(node, ast.ImportFrom) and (node.module == "utils" or node.module.startswith("maa")):
            continue
        body.append(node)
    module = types.ModuleType("star_swipe_existing_motion")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def geometry(repo, directory, modules):
    import cv2
    import base64
    manifest = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
    records = [manifest["before"], *manifest["samples"]]
    inputs = []
    core = load_core(repo)
    for record in records:
        image = core.read_image(directory / record["fileName"])
        inputs.append({"fileName": record["fileName"], "width": image.shape[1], "height": image.shape[0],
                       "rgba": base64.b64encode(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)).decode("ascii")})
    run = subprocess.run(["node", str(repo / "tools/star_swipe_geometry.mjs"), str(modules)],
                         input=json.dumps(inputs), text=True, encoding="utf-8", capture_output=True, check=True)
    value = json.loads(run.stdout)
    core.write_json(directory / "geometry.json", value)
    return value


def plan(core):
    return {"schemaVersion": 1, "scope": "not_run", "section_policy": "main and support must be compared separately",
        "reset_policy": "one command per gesture; manually return to identical top before every trial; no automatic reset",
        "runs": [{"group": group, "trial": trial, "section": section,
            "swipe_begin": [360, 930], "swipe_end": [360, 540],
            "swipe_duration_ms": duration, "swipe_end_hold_ms": hold,
            "post_release_settle_ms": 2500, "tab_settle_ms": 2500,
            "sample_offsets_ms": list(core.SAMPLE_OFFSETS_MS),
            "manual_start_authorized": False, "manual_reset_confirmed": False,
            "backend_touch_hold_confirmed": False}
            for section in ("main", "support") for trial in (1, 2, 3)
            for group, (duration, hold) in core.GROUPS.items()]}


def run_trial(args, core):
    # Reject before importing Maa, discovering or connecting any device.
    raw = {"group": args.group, "trial": args.trial, "section": args.section,
           "swipe_begin": [360, 930], "swipe_end": {"E": [360, 480], "F": [360, 435]}.get(args.group, [360, 540]),
           "extended_observation": args.extended_observation,
           "manual_start_authorized": args.manual_start_authorized,
           "manual_reset_confirmed": args.manual_reset_confirmed,
           "backend_touch_hold_confirmed": args.backend_touch_hold_confirmed}
    params = core.parse_calibration_params(raw)
    if args.group == "F":
        # Observation metadata only. Never use the target to select a motion estimate.
        params["target_content_shift_px"] = 499.5
    if not args.adb_path or not args.address:
        raise ValueError("run requires an explicit --adb-path and --address; no device auto-selection")
    config = json.loads(args.controller_config.read_text(encoding="utf-8")) if args.controller_config else {}
    if not isinstance(config, dict):
        raise ValueError("controller config must be a JSON object")
    if args.input_method == "EmulatorExtras" and not config.get("extras", {}).get("mumu", {}).get("enable"):
        raise ValueError("EmulatorExtras requires the existing enabled MuMu controller config")
    from maa.controller import AdbController
    from maa.define import MaaAdbInputMethodEnum, MaaAdbScreencapMethodEnum
    from maa.resource import Resource
    from maa.tasker import Tasker
    from maa.toolkit import Toolkit
    from maa.library import Library
    import importlib.metadata
    directory = args.output / f"{datetime.now():%Y%m%d-%H%M%S}-{args.section}-{args.group}-{args.trial}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    Toolkit.init_option(directory, {"logging": True, "save_on_error": False})
    # Pin ONE down/up-capable backend; never silently fall back to AdbShell.
    controller = AdbController(args.adb_path, args.address,
        screencap_methods=args.screencap_method or MaaAdbScreencapMethodEnum.Default,
        input_methods=getattr(MaaAdbInputMethodEnum, args.input_method), config=config)
    tasker = Tasker()
    try:
        if not controller.post_connection().wait().succeeded:
            raise RuntimeError("explicit controller connection failed")
        if not controller.set_screenshot_target_long_side(1280):
            raise RuntimeError("cannot set the controlled 720x1280 screenshot coordinate system")
        resource = Resource()  # Empty action-only resource. No OCR model is loaded.
        pipeline = directory / "native-swipe-pipeline.json"
        core.write_json(pipeline, core.swipe_pipeline(params))
        if not resource.post_pipeline(pipeline).wait().succeeded or not tasker.bind(resource, controller) or not tasker.inited:
            raise RuntimeError("native Pipeline resource/tasker initialization failed")
        class Context:
            def __init__(self): self.tasker = tasker
            def run_task(self, entry, override):
                return tasker.post_task(entry, override).wait().get()
            def abort_and_wait(self): tasker.post_stop().wait()
        manifest = core.collect_trial(Context(), params, directory)
        manifest["backend"] = {"address": args.address, "input_method": args.input_method,
                               "screencap_method": args.screencap_method or "Default",
                               "controller_uuid": controller.uuid,
                               "controller_config_sha256": hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                               "maafw_python_version": importlib.metadata.version("maafw"),
                               "native_version": Library.version()}
        core.write_json(directory / "trial.json", manifest)
        print(json.dumps({"status": manifest["status"], "directory": str(directory)}, ensure_ascii=False))
        return 0 if manifest["status"] == "captured" else 1
    except KeyboardInterrupt:
        # Framework owns its finite native gesture. Do not implement manual touch events.
        tasker.post_stop().wait()
        raise
    except Exception as exc:
        core.write_json(directory / "setup-failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run", help="ONE real gesture; game and reset must be manually prepared")
    run.add_argument("--group", choices=list("ABCDEF"), required=True)
    run.add_argument("--extended-observation", action="store_true", help="diagnostic only: add passive 3000/4000ms samples; required for E1")
    run.add_argument("--trial", type=int, choices=(1, 2, 3), required=True)
    run.add_argument("--section", choices=("main", "support"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--adb-path", type=Path, required=True)
    run.add_argument("--address", required=True)
    run.add_argument("--input-method", choices=("Maatouch", "MinitouchAndAdbKey", "EmulatorExtras"), required=True)
    run.add_argument("--controller-config", type=Path, help="existing MuMu extras config JSON; read-only")
    run.add_argument("--screencap-method", type=int)
    for flag in ("manual-start-authorized", "manual-reset-confirmed", "backend-touch-hold-confirmed"):
        run.add_argument("--"+flag, action="store_true")
    analyse = sub.add_parser("analyse")
    analyse.add_argument("--runs", nargs="+", type=Path, required=True)
    analyse.add_argument("--output", type=Path, required=True)
    analyse.add_argument("--geometry-modules", type=Path,
        help="frozen pure JS modules from previous OCR audit; never starts a browser")
    args = parser.parse_args()
    core = load_core(args.repo)
    if args.command == "plan":
        print(json.dumps(plan(core), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        return run_trial(args, core)
    probe = load_motion_algorithms(args.repo)
    result = core.summarize_trials(args.runs, probe)
    if args.geometry_modules:
        result["geometry"] = {str(directory): geometry(args.repo, directory, args.geometry_modules) for directory in args.runs
                              if json.loads((directory/"trial.json").read_text(encoding="utf-8"))["status"] == "captured"}
    else:
        result["geometry"] = {"status": "not_run", "reason": "supply --geometry-modules for existing four-column/completeness checks"}
    core.write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
