"""Offline resource and action-order validation; creates no controller."""
import importlib.util
import json
from pathlib import Path
import sys
from maa.resource import Resource

base, filename, skill = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('test_job_verifier', skill / 'scripts/verify_resource_load.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
verifier.MAAROOT, verifier.BASE = base.parent, base
verifier.CACHE, verifier.PIPE_DIR = base.parent / 'cache', base / 'pipeline/copilot'
verifier.CONFIG = verifier.PIPE_DIR / 'copilot_config.json'
assert verifier.main() == 0
resource = Resource()
resource.post_bundle(base).wait()
assert resource.loaded
actions = json.loads((base / 'pipeline/copilot' / filename).read_text(encoding='utf-8'))
assert len([n for n in actions.values() if n.get('custom_action') == 'DragonSmartAction']) == 10
for rnd in range(1, 11):
    key = f'回合{rnd}行动1'
    params = resource.get_node_data(key)['action']['param']['custom_action_param']
    assert params['position'] == 1 and params['node'] == key
    assert len(set(params['branches'].values())) == 3
    for kind, target in params['branches'].items():
        n = resource.get_node_data(target)
        assert n['action']['type'] == ('Click' if kind == 'normal' else 'Swipe')
        assert actions[target]['next'][-1] == f'回合{rnd}行动2'
    for slot in range(2, 6):
        key = f'回合{rnd}行动{slot}'
        assert resource.get_node_data(key)['action']['type'] == 'Click'
        expected = f'回合{rnd}行动{slot+1}' if slot < 5 else f'检测回合{rnd+1}' if rnd < 10 else '智能圈测试10回合结束'
        assert actions[key]['next'][-1] == expected
assert '检测回合11' not in actions
assert actions['智能圈测试提前胜利']['next'] == []
print('PASS: 10 smart circles + 40 ordinary attacks; native bundle loaded; no device input')
