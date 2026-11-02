"""Validate a generated job in an isolated bundle, without a game controller."""
from pathlib import Path
import subprocess
import sys
import tempfile


def run_smoke_test():
    root = next(p for p in Path(__file__).resolve().parents
                if (p / 'tools/copilot/dragon_smart/build_test_job.py').is_file())
    with tempfile.TemporaryDirectory(prefix='copilot-smoke-') as folder:
        result = subprocess.run([sys.executable, str(root / 'tools/copilot/dragon_smart/build_test_job.py'),
                                 '--output', str(Path(folder) / 'test-job.json')])
        return result.returncode == 0


if __name__ == '__main__':
    sys.exit(0 if run_smoke_test() else 1)
