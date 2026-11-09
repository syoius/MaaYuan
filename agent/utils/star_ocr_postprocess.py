"""Star field postprocessing, equivalent to the retained star-postprocess.js.

Reuse the existing runtime JSON and Context OCR candidates. Do not apply the
operator normalizer (NFKC/zhconv) or the item counter's Unicode digit extraction:
both would broaden the accepted star names/levels.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path


def _round_confidence(value):
    # Math.round(value * 100), including binary floating-point ties. Python's
    # round uses ties-to-even; adding .5 can itself round a value just below a tie.
    scaled = value * 100
    lower = math.floor(scaled)
    return (lower + (scaled - lower >= .5)) / 100


def clean_name(value):
    # JS removes whitespace/underscores/level suffixes, then keeps only L and N.
    # isalnum/\w are not exact substitutes for these Unicode general categories.
    return ''.join(char for char in value
                   if char not in '级級' and unicodedata.category(char)[0] in 'LN')


def resolve_catalog_name(candidates, allowed_names, aliases):
    cleaned = [{**item, 'text': clean_name(item['text'])} for item in candidates]
    cleaned = [item for item in cleaned if item['text']]
    if not cleaned:
        return dict(raw='', normalized=None, confidence=0, reasons=['name_ocr_empty'])
    # max and sorted are stable on ties, as are the JS candidate sorts.
    best = max(cleaned, key=lambda item: item['confidence'])
    normalized = aliases.get(best['text'], best['text'])
    if normalized in allowed_names:
        agreement = sum(aliases.get(item['text'], item['text']) == normalized
                        for item in cleaned)
        confidence = min(.99, best['confidence'] * .85 + min(agreement, 3) * .05)
        return dict(raw=best['text'], normalized=normalized,
                    confidence=_round_confidence(confidence), reasons=[])
    # Never fall back from the highest-scoring unknown to a lower-scoring name.
    return dict(raw=best['text'], normalized=None,
                confidence=_round_confidence(best['confidence'] * .5),
                reasons=['name_unknown'])


def parse_level(value):
    if '-' in value or '负' in value or '負' in value:
        return None
    # ECMAScript \d is ASCII-only. Keep the single-number rule and O/o mapping.
    matches = re.findall(r'[0-9]+', value.replace('O', '0').replace('o', '0'))
    if len(matches) != 1:
        return None
    digits = matches[0].lstrip('0')
    # Bound before int(): JS Number rejects overflow, while arbitrarily many
    # leading zeroes can still denote a valid level.
    if not digits or len(digits) > 2:
        return None
    level = int(digits)
    return level if 1 <= level <= 60 else None


def resolve_level(candidates):
    raw_best = max(candidates, key=lambda item: item['confidence'], default=None)
    groups = {}
    for item in candidates:
        level = parse_level(item['text'])
        if level is not None:
            groups.setdefault(level, []).append(item)
    if not groups:
        return dict(raw=raw_best['text'] if raw_best else '', level=None,
                    confidence=0, reasons=['level_unknown'])
    ranked = []
    for level, items in groups.items():
        weight = 0
        for item in items:
            # Preserve the JS reducer's left-associative addition order.
            weight = weight + item['confidence'] + (.22 if _has_suffix(item['text']) else 0)
        ranked.append(dict(level=level, items=items, weight=weight))
    ranked.sort(key=lambda group: group['weight'], reverse=True)
    winner = ranked[0]
    best = max(winner['items'], key=lambda item: item['confidence'])
    complete_high = _has_suffix(best['text']) and best['confidence'] >= .85
    levels = '/'.join(str(group['level']) for group in ranked)
    if len(ranked) > 1 and not complete_high and winner['weight'] - ranked[1]['weight'] < .15:
        return dict(raw=best['text'], level=None, confidence=0,
                    reasons=['level_strategy_conflict', 'level_candidates:' + levels])
    confidence = min(.99, best['confidence'] * .82 + min(len(winner['items']), 3) * .06)
    return dict(raw=best['text'], level=winner['level'],
                confidence=_round_confidence(confidence),
                reasons=[] if len(ranked) == 1 else ['level_weighted_consensus:' + levels])


def _has_suffix(value):
    return '级' in value or '級' in value


class StarFieldResolver:
    """Load the same runtime data once, including explicitly selected diagnostics."""

    def __init__(self, runtime):
        runtime = Path(runtime)
        stars = json.loads((runtime / 'star_catalog.json').read_text(encoding='utf-8'))['stars']
        self.aliases = json.loads((runtime / 'ocr_aliases.json').read_text(encoding='utf-8'))
        self.allowed = {kind: [star['name'] for star in stars if star['kind'] == kind]
                        for kind in ('主星', '辅星')}

    def resolve(self, kind, candidates, section='main'):
        if kind == 'name':
            allowed = self.allowed['主星' if section == 'main' else '辅星']
            return resolve_catalog_name(candidates, allowed, self.aliases)
        return resolve_level(candidates)
