"""Load motion analysis without starting the Maa Agent or a device."""
import ast
import sys
import types


def load_motion_algorithms(repo):
    path = repo / 'agent/custom/action/star_backpack_capture_probe.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'StarBackpackCaptureProbe':
            break
        if isinstance(node, ast.ImportFrom) and (node.module == 'utils' or node.module.startswith('maa')):
            continue
        body.append(node)
    module = types.ModuleType('star_test_motion')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module
