"""Build a manual-observation job in cache, never change the active pipeline."""
from copy import deepcopy
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKILL = ROOT / '.agents/skills/maayuan-copilot-builder'
FILENAME = '诸葛亮全程智能圈-10回合测试-v2.4.json'


def build(resource_base=None):
    resource_base = Path(resource_base) if resource_base else ROOT / 'assets/resource/base'
    spec = importlib.util.spec_from_file_location('smart_test_compiler', SKILL / 'scripts/compiler.py')
    compiler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compiler
    spec.loader.exec_module(compiler)
    profile = json.loads((SKILL / 'references/runtime_profiles/maayuan-v2.0.0-alpha.1.json').read_text(encoding='utf-8'))
    ir = {
        'title': '诸葛亮全程智能圈（10回合观察测试·v2.4）',
        'stage_name': 'zhugeliang_circle_observation',
        'details': '10行=10个游戏回合，无额外再动段。每回合按1、2、3、4、5号位依次行动：1号位诸葛亮执行圈，其他四人只普攻。'
                   '圈使用DOT维护默认策略：龙脉外能量足够放技能、不足普攻；龙脉内0龙气下拉、1龙气普攻、至少2龙气下拉。'
                   '不是曹丕新表的0龙气普攻策略，也不是吕布切换形态。动作按实时状态选择，不预写第一回合技能或第二回合下拉。'
                   '开局在龙脉外且能量满时应放技能；第二回合在龙脉内且龙气为0或至少2时应下拉，为1时应普攻。'
                   '手动进入测试关卡第1回合，使用抄作业任务，开启战斗中开始抄作业，从第1回合开始。'
                   '仅测一轮，10回合结束或提前胜利则停止；不自动换队或指定关卡，未设计牢狱/全队再动。识别不能确认则停止。',
        # The raw circle token means Lv Bu in the generic compiler. Replace a
        # temporary ordinary attack with our custom dispatcher after compilation.
        'rounds': [{'round': rnd, 'cells': {str(slot): f'{slot}A' for slot in range(1, 6)}} for rnd in range(1, 11)],
    }
    doc = compiler.compile_ir_to_copilot(ir, profile=profile)
    actions, metadata = doc['actions'], doc['_observer_metadata']
    stop, failure, victory = '智能圈测试10回合结束', '智能圈测试未确认停止', '智能圈测试提前胜利'
    actions[stop] = {'recognition': 'DirectHit', 'action': 'DoNothing', 'focus': '智能圈10回合观察测试结束，请核对诸葛亮行动并反馈。'}
    actions[failure] = {'recognition': 'DirectHit', 'action': 'DoNothing', 'focus': '智能圈测试已停止，请查看前面的具体原因。'}
    config = json.loads((resource_base / 'pipeline/copilot/copilot_config.json').read_text(encoding='utf-8'))
    actions[victory] = deepcopy(config['抄作业战斗胜利-check'])
    actions[victory].update(next=[], focus='战斗提前胜利，本次观察测试结束。')
    for rnd in range(1, 11):
        actions[f'检测回合{rnd}']['on_error'] = [failure]
        for slot in range(1, 6):
            key = f'回合{rnd}行动{slot}'
            following = f'回合{rnd}行动{slot+1}' if slot < 5 else f'检测回合{rnd+1}' if rnd < 10 else stop
            actions[key]['next'] = [victory, following]
            if slot > 1:
                actions[key]['focus'] = f'第{rnd}回合：{slot}号位普攻'
        key = f'回合{rnd}行动1'
        continuation = actions[key]['next']
        branches = {}
        for kind, label in (('normal', '普攻'), ('ultimate', '放技能'), ('defense', '下拉防御')):
            name = f'{key}-诸葛亮{label}'
            branches[kind] = name
            actions[name] = deepcopy(profile['slot_coordinates']['1'][kind])
            actions[name].update(recognition='DirectHit', next=list(continuation), focus=f'第{rnd}回合：诸葛亮{label}')
        actions[key] = {
            'recognition': 'DirectHit', 'action': 'Custom', 'custom_action': 'DragonSmartAction',
            'custom_action_param': {'node': key, 'position': 1, 'branches': branches},
            'next': [victory], 'on_error': [failure], 'post_delay': 0,
            'focus': f'第{rnd}回合：观察诸葛亮智能圈', 'text_doc': '1圈',
        }
        metadata[key].update(action_type='DragonSmartConditional', raw_cell='1圈', action_spec={'action': 'Custom'})
    actions['作业信息']['focus'] = '已加载智能圈10回合观察测试v2.4：1号位全程圈，其余四人全程普攻。'
    actions['抄作业胜利后继续'] = {'recognition': 'DirectHit', 'action': 'DoNothing', 'next': [], 'focus': '观察测试结束，不继续下一场。'}
    assert len(metadata) == 50
    assert '检测回合11' not in actions
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', type=Path, help='Optional client root: write complete job to its cache only')
    parser.add_argument('--output', type=Path, default=HERE / FILENAME)
    args = parser.parse_args()
    base_source = args.client / 'resource/base' if args.client else ROOT / 'assets/resource/base'
    if not (base_source / 'pipeline/copilot/copilot_config.json').is_file():
        parser.error('指定资源目录缺少 copilot_config.json')
    doc = build(base_source)
    data = json.dumps(doc, ensure_ascii=False, indent=2)
    with tempfile.TemporaryDirectory(prefix='test-job-', dir=HERE) as folder:
        stage = Path(folder)
        base = stage / 'base'
        def ignore(directory, names):
            p = Path(directory)
            return [n for n in names if p.name == 'copilot' and p.parent.name == 'pipeline' and n.endswith('.json') and n != 'copilot_config.json']
        shutil.copytree(base_source, base, copy_function=shutil.copy2, ignore=ignore)
        (base / 'pipeline/copilot' / FILENAME).write_text(json.dumps(doc['actions'], ensure_ascii=False), encoding='utf-8')
        result = subprocess.run([sys.executable, str(HERE / 'verify_test_job.py'), str(base), FILENAME, str(SKILL)], capture_output=True)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode('utf-8', errors='replace')[-4000:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data, encoding='utf-8')
    if args.client is None:
        print(f'已生成并通过离线检查：{args.output}')
        return
    # Cache only: the user chooses when to activate the job.
    cache = args.client / 'resource/copilot-cache' / FILENAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent, suffix='.tmp', delete=False) as f:
        f.write(data.encode('utf-8'))
        temporary = Path(f.name)
    os.replace(temporary, cache)
    print(f'已生成并通过离线检查：{cache}')


if __name__ == '__main__':
    main()
