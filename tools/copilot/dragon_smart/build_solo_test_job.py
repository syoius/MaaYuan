"""Derive a slot-one-only recording job from the validated observation job."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKILL = ROOT / '.agents/skills/maayuan-copilot-builder'
FILENAME = '诸葛亮单人智能圈-10回合录屏.json'


def build():
    doc = json.loads((HERE / '诸葛亮全程智能圈-10回合测试-v2.4.json').read_text(encoding='utf-8'))
    actions, meta = doc['actions'], doc['_observer_metadata']
    for name in list(actions):
        if re.fullmatch(r'回合\d+行动[2-5]', name):
            del actions[name]
            del meta[name]
    for rnd in range(1, 11):
        key = f'回合{rnd}行动1'
        actions[f'检测回合{rnd}']['next'] = [key]
        actions[key]['focus'] = f'第{rnd}回合：诸葛亮单人智能圈'
        following = f'检测回合{rnd+1}' if rnd < 10 else '智能圈测试10回合结束'
        for target in actions[key]['custom_action_param']['branches'].values():
            actions[target]['next'] = ['智能圈测试提前胜利', following]
    title = '诸葛亮单人智能圈（10回合录屏）'
    doc['stage_name'] = 'zhugeliang_solo_circle_recording'
    doc['level_meta'].update(stage_id=doc['stage_name'], level_id='custom/' + doc['stage_name'], name=title, cat_two=title)
    doc['doc'] = {
        'title': title,
        'details': '10行=10个游戏回合。只在1号位放诸葛亮，其他四位留空；每回合仅1号位执行一次智能圈，然后等待下一回合。'
                   '龙脉外：能量足够放技能，不足则普攻；龙脉内：0龙气下拉、1龙气普攻、至少2龙气下拉。'
                   '按实际状态动态选动作，不是吕布切换形态，也不预写技能/防御序列。'
                   '请手动进入战斗第1回合，用自动抄作业V3，开启战斗中开始抄作业，从第1回合开始。'
                   '执行至第10回合后停止，提前胜利也停止。不自动重开、不处理牢狱或额外再动；识别未确认时停止。'
                   '单人站位已用真实游戏截图离线校对，未代替用户运行战斗。需要智能圈v2.4.1或兼容版本。',
    }
    actions['作业信息']['focus'] = '已加载诸葛亮单人智能圈：每回合仅1号位行动，其他四位留空，共10回合。'
    actions['智能圈测试10回合结束']['focus'] = '诸葛亮单人智能圈10回合录屏作业结束。'
    assert len(meta) == 10 and all(row['slot'] == 1 for row in meta.values())
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', type=Path, help='Optional client root: write complete job to its cache only')
    parser.add_argument('--output', type=Path, default=HERE / FILENAME)
    args = parser.parse_args()
    base_source = args.client / 'resource/base' if args.client else ROOT / 'assets/resource/base'
    if not (base_source / 'pipeline/copilot/copilot_config.json').is_file():
        parser.error('指定资源目录缺少 copilot_config.json')
    doc = build()
    encoded = json.dumps(doc, ensure_ascii=False, indent=2).encode('utf-8')
    with tempfile.TemporaryDirectory(prefix='solo-job-', dir=HERE) as folder:
        base = Path(folder) / 'base'
        def ignore(directory, names):
            p = Path(directory)
            return [n for n in names if p.name == 'copilot' and p.parent.name == 'pipeline' and n.endswith('.json') and n != 'copilot_config.json']
        shutil.copytree(base_source, base, copy_function=shutil.copy2, ignore=ignore)
        (base / 'pipeline/copilot' / FILENAME).write_text(json.dumps(doc['actions'], ensure_ascii=False), encoding='utf-8')
        result = subprocess.run([sys.executable, str(HERE / 'verify_solo_test_job.py'), str(base), FILENAME, str(SKILL)], capture_output=True)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode('utf-8', errors='replace')[-3000:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    if args.client is None:
        print(f'已生成并通过离线检查：{args.output}')
        return
    cache = args.client / 'resource/copilot-cache' / FILENAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent, suffix='.tmp', delete=False) as f:
        f.write(encoded)
        temporary = Path(f.name)
    os.replace(temporary, cache)
    print(f'已生成并通过离线检查：{cache}；当前激活作业未改变。')


if __name__ == '__main__':
    main()
