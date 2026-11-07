"""Verify solo graph and resource loading without creating a game controller."""
import importlib.util
import json
from pathlib import Path
import re
import sys
from maa.resource import Resource

base, filename, skill = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('solo_verifier', skill / 'scripts/verify_resource_load.py')
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
assert not any(re.fullmatch(r'回合\d+行动[2-5]', name) for name in actions)
for rnd in range(1, 11):
    key = f'回合{rnd}行动1'
    assert actions[f'检测回合{rnd}']['next'] == [key]
    params = resource.get_node_data(key)['action']['param']['custom_action_param']
    assert params['position'] == 1 and params['node'] == key
    assert len(set(params['branches'].values())) == 3
    following = f'检测回合{rnd+1}' if rnd < 10 else '智能圈测试10回合结束'
    for kind, target in params['branches'].items():
        node = resource.get_node_data(target)
        assert node['action']['type'] == ('Click' if kind == 'normal' else 'Swipe')
        assert actions[target]['next'] == ['智能圈测试提前胜利', following]
        action = node['action']['param']
        # The only physical battle branches target the leftmost card.
        coord = action['target'] if kind == 'normal' else action['begin']
        assert isinstance(coord, list) and 0 <= coord[0] < 140, action
assert '检测回合11' not in actions
assert actions['智能圈测试提前胜利']['next'] == []
print('PASS: 10 solo circles, all physical branches target slot 1, no other slots; native resource loaded')
