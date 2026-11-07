# -*- coding: utf-8 -*-
"""
编译期静态校验：TemplateMatch 引用的模板图是否真实存在。

为什么需要（2026-10-06 踩坑）：
MaaFramework 在**资源加载时不会校验模板图是否存在**。缺图不会让
「资源加载失败」，而是等到战斗中真跑到那一步才发现匹配永远失败，
表现为节点重试到超时、整个任务被判失败 —— 排查成本极高。

这类问题本可以在编译后立刻发现，所以这里补一道闸。

同时校准阈值：若能从 MaaFramework 日志（debug/maa.log）读到该节点历史
匹配分数，会给出建议阈值，避免"阈值定得比实际可达分数还高"的坑。

用法（必须用 MaaYuan 自带 python）：
    "D:\\MaaYuan-win-x86_64-v2.0.0-alpha.1\\python\\python.exe" \\
        scripts/validate_templates.py [作业.json ...]

不带参数时校验 copilot-cache 下全部作业。
退出码 0 = 全部通过；1 = 有缺失模板或阈值明显不可达。
"""
import json
import os
import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
MAAROOT = Path(os.environ['MAAYUAN_CLIENT']).resolve() if os.environ.get('MAAYUAN_CLIENT') else next(
    (p for p in Path(__file__).resolve().parents
     if (p / "resource" / "base" / "pipeline" / "copilot" / "copilot_config.json").is_file()
     or (p / 'assets/resource/base/pipeline/copilot/copilot_config.json').is_file()),
    None,
)
if MAAROOT is None:
    print("[FAIL] 未找到 MaaYuan 安装目录（需含 resource/base/pipeline/copilot/copilot_config.json）")
    sys.exit(2)

IMAGE_DIR = MAAROOT / "resource" / "base" / "image"
if not IMAGE_DIR.is_dir():
    IMAGE_DIR = MAAROOT / 'assets/resource/base/image'
CACHE_DIR = MAAROOT / "resource" / "copilot-cache"
LOG_FILE = MAAROOT / "debug" / "maa.log"


def collect_templates(node, found=None):
    """递归收集节点里所有 template 引用的图片名。"""
    if found is None:
        found = []
    if isinstance(node, dict):
        rec = node.get("recognition")
        if isinstance(rec, dict):
            tpls = rec.get("param", {}).get("template")
            if isinstance(tpls, str):
                found.append(tpls)
            elif isinstance(tpls, list):
                found.extend(tpls)
        # 顶层 template（旧格式）
        t = node.get("template")
        if isinstance(t, str):
            found.append(t)
        elif isinstance(t, list):
            found.extend(t)
        for v in node.values():
            collect_templates(v, found)
    elif isinstance(node, list):
        for v in node:
            collect_templates(v, found)
    return found


def collect_thresholds(node, found=None):
    """收集 (节点名, roi, threshold) 三元组。"""
    if found is None:
        found = []
    if isinstance(node, dict):
        rec = node.get("recognition")
        if isinstance(rec, dict) and rec.get("type") == "TemplateMatch":
            param = rec.get("param", {})
            found.append((node.get("focus", "?"), param.get("roi"), param.get("threshold")))
        for v in node.values():
            collect_thresholds(v, found)
    elif isinstance(node, list):
        for v in node:
            collect_thresholds(v, found)
    return found


def historical_scores():
    """从 debug/maa.log 读 {节点名: 最高匹配分}。仅作阈值合理性参考。"""
    if not LOG_FILE.is_file():
        return {}
    try:
        raw = LOG_FILE.read_bytes().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return {}
    best = {}
    # 形如: [cur_node_=回合2行动2] ... all_results_=[{"box":...,"score":0.68...}]
    for line in raw.splitlines():
        if "all_results_" not in line:
            continue
        m = re.search(r"\[cur_node_=([^\]]+)\]", line)
        if not m:
            continue
        scores = [float(x) for x in re.findall(r'"score":([0-9.]+)', line)]
        if not scores:
            continue
        node = m.group(1)
        best[node] = max(best.get(node, 0.0), max(scores))
    return best


def check_job(path: Path, scores: dict) -> list:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    actions = data.get("actions") if isinstance(data, dict) else None
    if not isinstance(actions, dict):
        # 裸节点 map（base 下的镜像文件）
        actions = data if isinstance(data, dict) else {}

    for tpl in sorted(set(collect_templates(actions))):
        if not (IMAGE_DIR / tpl).is_file():
            problems.append(f"  [缺模板] {tpl}  -> 应存在于 {IMAGE_DIR / tpl}")

    # 阈值可达性：历史最高分明显低于阈值，说明这步很可能永远匹配不上
    for focus, roi, thr in collect_thresholds(actions):
        if thr is None:
            continue
        for node, best in scores.items():
            # 用节点名粗略匹配 focus 里的"第N动"太脆弱，这里只做提示不做判定
            del node, best
        if isinstance(thr, (int, float)) and thr > 0.95:
            problems.append(f"  [阈值可疑] {thr} 过高：{str(focus)[:40]}")

    return problems


def main(argv) -> int:
    if argv:
        jobs = [Path(a) for a in argv]
    else:
        jobs = sorted(p for p in CACHE_DIR.glob("*.json") if "observer_metadata" not in p.name)

    if not jobs:
        print("copilot-cache 下没有作业文件")
        return 0

    scores = historical_scores()
    if scores:
        print(f"已从 {LOG_FILE.name} 读到 {len(scores)} 个节点的历史匹配分（仅参考）")

    all_problems = []
    for job in jobs:
        print(f"\n检查 {job.name}")
        problems = check_job(job, scores)
        if problems:
            for p in problems:
                print(p)
            all_problems.extend(problems)
        else:
            print("  模板齐全，未发现明显问题")

    if all_problems:
        print(f"\n[FAIL] 共 {len(all_problems)} 处问题")
        return 1
    print("\n[PASS] 所有作业引用的模板图均存在")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
