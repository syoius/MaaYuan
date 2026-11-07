# -*- coding: utf-8 -*-
"""
复现 MaaYuan 客户端的资源加载流程，验证作业不会拖垮整个资源包。

客户端实测行为（2026-10-06 逐条验证，勿再凭猜测改动）：
1. 客户端启动只加载 resource/base 这一个 bundle（日志："Resource: ...\\resource\\base"）。
2. 它会把当前选中的作业**摊平成裸节点 map**写进 base/pipeline/copilot/，
   格式参照resource/copilot-hold/*.json —— 只有节点，没有 version/difficulty 等顶层字段。
3. 完整作业文档留在 copilot-cache/ 供 UI 列出，metadata 留在 base/pipeline_metadata/copilot/。
4. 于是 base/pipeline/copilot/ 下**同一时刻只能有一个作业文件**：
   两个作业会撞约 19 个同名节点（检测回合N / 回合N行动M / 作业信息 / 抄作业*）。
5. 作业节点也不能覆盖 copilot_config.json 里已有的节点，
   引擎遇到重复 key 直接拒绝加载（"key already exists"），即便写{"enabled": false} 也不行。

所以判断作业能否交付的标准只有一个：post_bundle(base) 之后 loaded == True。

用法（必须用 MaaYuan 自带的 python，普通环境没有 maa 模块）：
    "D:\\MaaYuan-win-x86_64-v2.0.0-alpha.1\\python\\python.exe" scripts/verify_resource_load.py

退出码 0 = 可交付；非 0 = 客户端会显示「资源加载失败」。
"""
import contextlib
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path

try:
    import maa.resource as res
except ImportError:
    print("[FAIL] 当前 python 没有 maa 模块，请改用 MaaYuan 自带的 python/python.exe")
    sys.exit(2)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = re.compile(r"\[[A-Za-z0-9_]+\.(?:cpp|h)\]\[L\d+\]")
JSON_BLOB = re.compile(r"\[json=.*")


def _find_maa_root() -> Path:
    """从脚本位置向上找 MaaYuan 安装目录。

    必须校验 resource/base/pipeline/copilot/copilot_config.json 存在——只看
    resource/base 会被测试残留的空目录骗到（2026-10-06 踩过）。
    """
    if os.environ.get('MAAYUAN_CLIENT'):
        return Path(os.environ['MAAYUAN_CLIENT']).resolve()
    here = Path(__file__).resolve()
    marker = Path("resource", "base", "pipeline", "copilot", "copilot_config.json")
    for candidate in here.parents:
        if (candidate / marker).is_file():
            return candidate
        if (candidate / 'assets' / marker).is_file():
            return candidate
    raise FileNotFoundError(
        "未找到 MaaYuan 安装目录（需包含 resource/base/pipeline/copilot/copilot_config.json）"
    )


MAAROOT = _find_maa_root()
BASE = MAAROOT / "resource" / "base"
if not BASE.is_dir() and (MAAROOT / 'assets/resource/base').is_dir():
    BASE = MAAROOT / 'assets/resource/base'
CACHE = MAAROOT / "resource" / "copilot-cache"
PIPE_DIR = BASE / "pipeline" / "copilot"
CONFIG = PIPE_DIR / "copilot_config.json"


def scan_duplicate_keys() -> list:
    """静态扫描 base/pipeline 下的重名 key——引擎会因此拒绝加载。"""
    problems = []
    seen = {}
    for path in sorted((BASE / "pipeline").rglob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"[JSON] {path.name} 解析失败: {exc}")
            continue
        if not isinstance(data, dict):
            problems.append(f"[结构] {path.name} 顶层不是对象（作业本体不该放进 base/pipeline）")
            continue
        for key in data:
            if key in seen:
                problems.append(f"[重名] key={key} 同时存在于 {seen[key]} 与 {path.name}")
            else:
                seen[key] = path.name
    return problems


def check_cache_hygiene() -> list:
    """copilot-cache 里不该有 metadata 旁挂文件，否则会被同步进 base/pipeline。"""
    return sorted(CACHE.glob("*.observer_metadata.json")) if CACHE.is_dir() else []


def load_base() -> tuple:
    """按客户端方式加载资源包，返回 (loaded, 节点数, 错误行列表)。"""
    buf = io.StringIO()
    handle = res.Resource()
    with contextlib.redirect_stdout(buf):
        job = handle.post_bundle(BASE)
        job.wait()
        loaded = bool(handle.loaded)
        node_count = len(handle.node_list) if loaded else 0

    errors = []
    for line in ANSI.sub("", buf.getvalue()).splitlines():
        if "[ERR]" not in line:
            continue
        clean = JSON_BLOB.sub("", NOISE.sub("", line)).strip()
        if clean:
            errors.append(clean[:200])
    return loaded, node_count, errors


def main() -> int:
    print(f"资源根目录: {MAAROOT}")

    print("\n[1] 扫描 base/pipeline 重名 key ...")
    dups = scan_duplicate_keys()
    for d in dups:
        print("    " + d)
    if dups:
        print(f"\n[FAIL] {len(dups)} 处结构问题，引擎必然拒绝加载。")
        print("       修法：base/pipeline/copilot/ 下只保留 copilot_config.json + 一个作业。")
        return 1
    print("    无重名 key")

    print("\n[2] 检查 copilot-cache 卫生 ...")
    leftovers = check_cache_hygiene()
    if leftovers:
        for p in leftovers:
            print(f"    [污染] {p.name} —— 不是合法作业格式，别放在这里")
        return 1
    print("    干净")

    print("\n[3] 实测 post_bundle(resource/base) ...")
    loaded, node_count, errors = load_base()
    if not loaded:
        print("    [FAIL] 资源包加载失败，客户端会显示「资源加载失败」。引擎报错：")
        for e in errors[:8]:
            print("      " + e)
        return 1
    print(f"    加载成功，共 {node_count} 个节点")

    print("\n[4] 作业状态 ...")
    job_files = [p.name for p in PIPE_DIR.glob("*.json") if p.name != "copilot_config.json"]
    if job_files:
        print(f"    base 中的作业节点: {job_files}")
    else:
        print("    base 中暂无作业节点（客户端选中作业时会自行写入）")
    if CACHE.is_dir():
        cached = sorted(p.name for p in CACHE.glob("*.json"))
        print(f"    copilot-cache 作业列表: {cached}")

    print("\n[PASS] 资源可加载，重启客户端即可使用。")
    return 0


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--base', type=Path, help='Resource bundle to inspect (read only)')
        args = parser.parse_args()
        if args.base:
            BASE = args.base.resolve()
            PIPE_DIR = BASE / 'pipeline/copilot'
            CONFIG = PIPE_DIR / 'copilot_config.json'
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
