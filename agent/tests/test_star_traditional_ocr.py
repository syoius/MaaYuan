"""Packaged resolver/Recognizer/transfer checks; OCR text here is synthetic, not image evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
from utils.star_local_ocr_shadow import Bridge, PACKAGED_RUNTIME, Recognizer
from custom.action.star_inventory_sync import compact_inventory


# The complete traditional-name table used by YuanHub's embedded YuanStar OCR.
TRADITIONAL_NAMES = {
    '天機': '天机', '天鉞': '天钺', '天馬': '天马', '天貴': '天贵',
    '陰煞': '阴煞', '太陽': '太阳', '太陰': '太阴', '巨門': '巨门',
    '祿存': '禄存', '左輔': '左辅', '廉貞': '廉贞', '紅鸞': '红鸾',
    '貪狼': '贪狼', '破軍': '破军', '七殺': '七杀', '鈴星': '铃星',
}


class TraditionalOcrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.bridge = Bridge(PACKAGED_RUNTIME, Path(cls.directory.name) / 'bridge.log')
        cls.addClassCleanup(cls.bridge.close)
        cls.stars = json.loads((PACKAGED_RUNTIME / 'star_catalog.json').read_text(encoding='utf-8'))['stars']
        cls.sections = {s['name']: 'main' if s['kind'] == '主星' else 'support'
                        for s in cls.stars if s['kind'] in ('主星', '辅星')}

    def resolve(self, kind, candidates, section='main'):
        variants = [{'variant': 'test', 'all_results': [dict(text=t, score=c) for t, c in candidates]}]
        return self.bridge.resolve(kind, variants, section)

    def test_traditional_resource_contains_shared_star_pipelines(self):
        for name in ('star_local_ocr_navigation.json', 'star_local_ocr_task.json'):
            base = ROOT / 'assets/resource/base/pipeline' / name
            traditional = ROOT / 'assets/resource/zh_tw/pipeline' / name
            self.assertEqual(json.loads(traditional.read_text(encoding='utf-8')),
                             json.loads(base.read_text(encoding='utf-8')))

    def test_all_web_aliases_and_canonical_names_remain_exact_and_section_scoped(self):
        aliases = json.loads((PACKAGED_RUNTIME / 'ocr_aliases.json').read_text(encoding='utf-8'))
        self.assertEqual(aliases, TRADITIONAL_NAMES)
        for raw, canonical in {**{n: n for n in self.sections}, **TRADITIONAL_NAMES}.items():
            section = self.sections[canonical]
            with self.subTest(raw=raw):
                result = self.resolve('name', [(raw, .99)], section)
                self.assertEqual(result['normalized'], canonical)
                self.assertEqual(result['reasons'], [])
                wrong = 'support' if section == 'main' else 'main'
                self.assertIsNone(self.resolve('name', [(raw, .99)], wrong)['normalized'])

    def test_name_cleanup_and_agreement_are_equivalent_without_guessing(self):
        for traditional, canonical in TRADITIONAL_NAMES.items():
            section = self.sections[canonical]
            with self.subTest(traditional=traditional):
                for suffix in ('级', '級'):
                    result = self.resolve('name', [(f' _{traditional}{suffix}！ ', .99)], section)
                    self.assertEqual(result['normalized'], canonical)
                both = self.resolve('name', [(traditional, .99), (canonical, .99)], section)
                same = self.resolve('name', [(canonical, .99), (canonical, .99)], section)
                self.assertEqual(both['confidence'], same['confidence'])
        # No substring matching, inferred spellings, or per-character conversion.
        for text in ('天', '天幾', '天機天鉞', '未知天機', '天機60級', '陀羅', '左弼'):
            with self.subTest(text=text):
                result = self.resolve('name', [(text, .99)], 'support' if text in ('陀羅', '左弼') else 'main')
                self.assertIsNone(result['normalized'])
                self.assertEqual(result['reasons'], ['name_unknown'])
        self.assertEqual(self.resolve('name', [(' _！ ', .99)])['reasons'], ['name_ocr_empty'])
        # The best unknown candidate must not silently fall back to a known name.
        self.assertIsNone(self.resolve('name', [('天幾', .99), ('天機', .98)])['normalized'])
        for raw, canonical, section in [('天機', '天机', 'main'), ('天鉞', '天钺', 'support'),
                                         ('天魁', '天魁', 'support'), ('文曲', '文曲', 'support'),
                                         ('武曲', '武曲', 'main'), ('陀螺', '陀螺', 'support')]:
            self.assertEqual(self.resolve('name', [(raw, .99)], section)['normalized'], canonical)

    def test_level_suffix_has_identical_consensus_conflict_and_bounds(self):
        for candidates, expected in [([(60, .90), ('50', .99)], 60),
                                     ([(60, .80), ('50', .99)], None),
                                     ([(1, .99)], 1), ([(60, .99)], 60),
                                     ([(0, .99)], None), ([(61, .99)], None)]:
            results = []
            for suffix in ('级', '級'):
                values = [(f'{t}{suffix}' if isinstance(t, int) else t, c) for t, c in candidates]
                result = self.resolve('level', values)
                self.assertEqual(result['level'], expected)
                results.append({k: v for k, v in result.items() if k != 'raw'})
            self.assertEqual(*results)
        for text in ('-1级', '-1級', '负1级', '負1級', '1級60', '未知'):
            with self.subTest(text=text):
                self.assertIsNone(self.resolve('level', [(text, .99)])['level'])
        self.assertEqual(self.resolve('level', [('60級', .99)])['raw'], '60級')

    def test_high_confidence_traditional_level_keeps_single_ocr_call(self):
        image = np.full((12, 12, 3), 100, np.uint8)
        rect = dict(x=1, y=1, width=3, height=3)
        card = dict(rowIndex=0, columnIndex=0)
        for text in ('60级', '60級'):
            def provider(image):
                return dict(ocr_ms=1., all_results=[dict(text=text, score=.99, box=[0, 0, 9, 9])])
            recognizer = Recognizer(provider, self.bridge, self.directory.name, save_debug_crops=False)
            result = recognizer.field(image, rect, 'level', 'test', card)
            self.assertEqual(result['resolved'], 60)
            self.assertFalse(result['fallback'])
            self.assertEqual(recognizer.calls, 1)

    def test_traditional_and_simplified_fields_produce_identical_inventory_json(self):
        def payload(traditional):
            sections = {}
            for section, raw, level, quality in [('main', '天機', 60, '橙'), ('support', '天鉞', 20, '紫')]:
                name = raw if traditional else TRADITIONAL_NAMES[raw]
                resolved_name = self.resolve('name', [(name, .99)], section)['normalized']
                resolved_level = self.resolve('level', [(str(level) + ('級' if traditional else '级'), .99)])['level']
                # Two physical instances with identical fields must both survive.
                items = [dict(section=section, name=resolved_name, quality=quality, level=resolved_level)
                         for _ in range(2)]
                sections[section] = dict(complete=True, count=len(items), items=items)
            local = dict(schemaVersion=2, complete=True, sections=sections,
                         experience=dict(complete=True, structured=True, orangeCount=0, purpleCount=123, whiteCount=45),
                         bag=dict(complete=True, currentCount=4, capacity=500))
            return compact_inventory(local, '11111111-1111-4111-8111-111111111111')
        traditional, simplified = payload(True), payload(False)
        self.assertEqual(traditional, simplified)
        self.assertEqual(set(traditional), {'format', 'version', 'scan_id', 'complete', 'counts', 'inventory', 'experience', 'bag'})
        self.assertEqual((traditional['format'], traditional['version']), ('maayuan-star-inventory', 1))
        self.assertEqual(traditional['counts'], dict(main=2, support=2))
        self.assertEqual(traditional['inventory'], [dict(kind='main', name='天机', quality='orange', level=60)] * 2
                         + [dict(kind='support', name='天钺', quality='purple', level=20)] * 2)
        self.assertEqual(traditional['experience'], dict(orange=0, purple=123, white=45))
        self.assertEqual(traditional['bag'], dict(current_count=4, capacity=500))


if __name__ == '__main__':
    unittest.main()
