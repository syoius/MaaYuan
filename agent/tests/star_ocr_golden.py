"""Read immutable expected values captured from the last JS/Python comparison."""
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

FIXTURES = Path(__file__).with_name('fixtures') / 'ocr_golden'


@lru_cache(maxsize=None)
def _load(group):
    return json.loads((FIXTURES / f'{group}.json').read_text(encoding='utf-8'))


def golden(group, op, image=None, **inputs):
    if image is not None:
        inputs['image'] = [list(image.shape), sha256(image.tobytes()).hexdigest()]
    key = op + ':' + sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    try:
        return deepcopy(_load(group)[key])
    except KeyError as exc:
        raise AssertionError(f'No fixed OCR expectation for {group}/{key}; verify the sample and inputs') from exc


def field_resolution(kind, variants, section='main'):
    candidates = [dict(text=t['text'], confidence=t['score'], variant=v['variant'])
                  for v in variants for t in v['all_results']]
    return golden('session', 'resolve', kind=kind, section=section, candidates=candidates)
