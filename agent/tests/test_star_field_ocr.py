from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from star_ocr_golden import field_resolution
from utils.star_bag_ocr import BAG_ROI
from utils.star_bag_ocr import parse_bag_text
from utils.star_bag_ocr import recognize_bag
from utils.star_local_ocr_shadow import NativeStarOcr
from utils.star_local_ocr_shadow import PACKAGED_RUNTIME
from utils.star_local_ocr_shadow import Recognizer


class BagOcrTests(unittest.TestCase):
    def test_pair_format_and_real_zero(self):
        self.assertEqual(parse_bag_text(' 203 / 270 '),(203,270))
        self.assertEqual(parse_bag_text('0／500'),(0,500))
    def test_invalid_or_missing_never_becomes_zero(self):
        for text in ('','270','203','203/','/270','203/200','203/0','-1/270','1.5/270','2O3/270','203/270 extra','2147483648/2147483649',None):
            with self.subTest(text=text):self.assertIsNone(parse_bag_text(text))
    def image(self):return np.zeros((1280,720,3),np.uint8)
    def provider(self,values):
        values=iter(values)
        def call(image):
            text,score=next(values)
            self.assertEqual(image.shape,(BAG_ROI[3]*3,BAG_ROI[2]*3,3))
            return {'all_results':[{'text':text,'score':score}],'ocr_ms':1}
        return call
    def test_confident_reading_uses_one_shared_provider_call(self):
        image=self.image();result=recognize_bag(image,self.provider([('203/270',.99)]))
        self.assertTrue(result['complete']);self.assertEqual((result['currentCount'],result['capacity']),(203,270));self.assertEqual(len(result['variants']),1)
        self.assertFalse(image.any())
    def test_low_confidence_and_missing_stop_after_two_variants(self):
        for values in ([('203/270',.2),('203/270',.3)],[('bad',.99),('',.99)]):
            result=recognize_bag(self.image(),self.provider(values));self.assertFalse(result['complete']);self.assertIsNone(result['capacity']);self.assertIsNone(result['currentCount']);self.assertEqual(len(result['variants']),2)
    def test_conflicting_confident_variants_are_unknown(self):
        result=recognize_bag(self.image(),self.provider([('203/270',.92),('208/270',.99)]))
        self.assertFalse(result['complete']);self.assertIsNone(result['capacity']);self.assertIn('bag_numbers_conflict',result['warnings'])
    def test_wrong_frame_never_invokes_ocr(self):
        def forbidden(image):raise AssertionError('bad frame reached OCR')
        self.assertFalse(recognize_bag(np.zeros((20,20,3),np.uint8),forbidden)['complete'])

class BagCaptureTests(unittest.TestCase):
    def test_uncertain_frame_retries_are_bounded_and_identity_checked(self):
        from custom.action import star_full_local_ocr_shadow as session
        with tempfile.TemporaryDirectory() as name:
            events=[]
            owner=SimpleNamespace(out=Path(name),last_capture={'file':'original.png'},progress=lambda stage:None,ocr=object(),check=lambda:None,
                capture=lambda label:events.append(label) or np.zeros((1280,720,3),np.uint8),check_backpack=lambda image,section:events.append(section))
            with patch.object(session,'recognize_bag',side_effect=lambda *args:session.empty_bag('unconfirmed')) as reco, patch.object(session.time,'sleep'):
                session.FullShadowSession.read_bag(owner,np.zeros((1280,720,3),np.uint8))
            self.assertEqual(reco.call_count,3);self.assertEqual(events,['bag-stable-retry','main','bag-stable-retry','main'])
            self.assertIsNone(owner.bag['capacity']);self.assertFalse(owner.bag['complete'])

def variants(values):
    return [dict(variant='test', all_results=[dict(text=text, score=score)
                                            for text, score in values])]

class NativePostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.output = Path(directory.name)
        cls.backends = {section: NativeStarOcr(PACKAGED_RUNTIME, lambda s=section: s)
                        for section in ('main', 'support')}

    def compare(self, kind, values, section='main'):
        inputs = variants(values)
        actual = self.backends[section].resolve(kind, inputs)
        self.assertEqual(actual, field_resolution(kind, inputs, section))
        return actual

    def test_name_order_unknown_unicode_and_alias_agreement_match_golden(self):
        cases = [
            [], [(' _！級 ', .99)], [('天機', .99)], [('天鉞', .99)],
            [('天幾', .99), ('天機', .98)],
            [('天幾', .99), ('天機', .99)], [('天機', .99), ('天幾', .99)],
            [('天機', .9), ('天机', .9), ('天機', .8), ('天机', .7)],
            [(' _天\u0301機級！\u200b', .99)],
            [('天機１', .99)], [('天機Ⅳ', .99)], [('𐐀天機', .99)],
            [('天機', .5)],
        ]
        for section in ('main', 'support'):
            for values in cases:
                with self.subTest(section=section, values=values):
                    self.compare('name', values, section)

    def test_level_ascii_bounds_negative_and_multiple_numbers_match_golden(self):
        texts = ['1', '60級', '0', '61', 'O1级', '6o級', '负1级', '負1級',
                 '-1级', '1級60', '未知', '', '６０級', '٦٠級', '²級',
                 '1.5', 'O', '00060級', '1a', '0' * 4500 + '1級', '9' * 4500]
        for text in texts:
            with self.subTest(text=text[:40]):
                self.compare('level', [(text, .99)])
        self.compare('level', [])

    def test_level_weight_conflict_suffix_threshold_and_ties_match_golden(self):
        cases = [
            [('60級', .90), ('50', .99)],
            [('60級', .80), ('50', .99)],
            [('60級', .85), ('50', .99)],
            [('60級', .8499999999999999), ('50', .99)],
            [('60', .99), ('50', .99)],
            [('50', .99), ('60', .99)],
            [('60', .7), ('50', .55)],
            [('60', .7000000000000001), ('50', .55)],
            [('60', .5), ('60級', .3), ('50', .99)],
            [('60級', .5), ('60级', .5), ('60', .5), ('50', .99)],
            [('bad', .99), ('60', .5)],
        ]
        for values in cases:
            with self.subTest(values=values):
                self.compare('level', values)

    def test_confidence_rounding_matches_js_at_half_cent_boundaries(self):
        for score in (.5, .3, .1, .999):
            with self.subTest(score=score):
                self.compare('name', [('天機', score)])
                self.compare('name', [('未知', score)])
                self.compare('level', [('60級', score)])
        self.assertEqual(self.backends['main'].resolve('name', variants([('天機', .5)]))['confidence'], .48)


    def recognize_field(self, kind, *readings):
        image = np.full((12, 12, 3), 100, np.uint8)
        rect = dict(x=1, y=1, width=3, height=3)
        card = dict(rowIndex=0, columnIndex=0)
        source = iter(readings)

        def provider(prepared):
            return dict(ocr_ms=1., all_results=[dict(text=text, score=score, box=[0, 0, 9, 9])
                                                for text, score in next(source)])

        recognizer = Recognizer(provider, self.backends['main'], self.output, save_debug_crops=False)
        result = recognizer.field(image, rect, kind, 'test', card)
        self.assertEqual(recognizer.calls, len(result['variants']))
        self.assertEqual(result['fallback_calls'], recognizer.calls - 1)
        self.assertEqual(result['fallback_ocr_ms'], recognizer.calls - 1)
        self.assertEqual(result['fallback'], recognizer.calls > 1)
        return result

    def test_color_requires_reliable_catalog_name_or_suffixed_level(self):
        for kind, text, expected in [('name', '天機', '天机'), ('level', '60级', 60), ('level', '60級', 60)]:
            with self.subTest(kind=kind, text=text):
                direct = self.recognize_field(kind, [(text, .90)])
                self.assertEqual(direct['resolved'], expected)
                self.assertEqual(direct['max_raw_confidence'], .90)
                self.assertGreaterEqual(direct['confidence'], .80)
                fallback = self.recognize_field(kind, [(text, .89)], [(text, .95)])
                self.assertEqual(fallback['resolved'], expected)
                self.assertEqual(fallback['fallback_calls'], 1)
        for kind, text in [('name', '未知'), ('level', '60'), ('level', '61級')]:
            with self.subTest(kind=kind, text=text):
                result = self.recognize_field(kind, [(text, .99)], [], [])
                self.assertEqual(result['fallback_calls'], 2)

    def test_contrast_must_be_strong_and_agree_with_existing_candidates(self):
        for kind, earlier, correct, expected in [('name', '天府', '天機', '天机'),
                                                 ('level', '50', '60級', 60)]:
            with self.subTest(kind=kind):
                clean = self.recognize_field(kind, [(correct, .89)], [(correct, .95)])
                self.assertEqual(clean['resolved'], expected)
                self.assertEqual([v['variant'] for v in clean['variants']], ['color_3x', 'contrast_3x'])
                for first, score in [(earlier, .95), (correct, .9499)]:
                    result = self.recognize_field(kind, [(first, .89)], [(correct, score)], [(correct, .99)])
                    self.assertEqual(result['resolved'], expected)
                    self.assertEqual(result['variants'][-1]['variant'], 'otsu_3x')
        ambiguous = self.recognize_field('name', [('天機', .99), ('天府', .87)],
                                         [('天機', .99)], [('天機', .99)])
        self.assertEqual(ambiguous['fallback_calls'], 2)

    def test_empty_otsu_preserves_unconfirmed_resolution(self):
        result = self.recognize_field('level', [('60', .95)], [('50', .96)], [])
        self.assertEqual(result['variants'][-1]['variant'], 'otsu_3x')
        self.assertIsNone(result['resolved'])
        self.assertEqual(result['confidence'], 0)
        self.assertIn('level_strategy_conflict', result['resolution']['reasons'])

    def test_recognizer_fallback_and_returned_fields_match_golden(self):
        for kind, readings in [('name', [('未知', .5), ('天機', .99), ('天机', .99)]),
                               ('level', [('O', .5), ('60級', .9), ('50', .99)])]:
            with self.subTest(kind=kind):
                original = [dict(variant=label, all_results=[dict(text=text, score=score)])
                            for label, (text, score) in zip(('color_3x', 'contrast_3x', 'otsu_3x'), readings)]
                golden = field_resolution(kind, original)
                native = self.recognize_field(kind, *([(text, score)] for text, score in readings))
                key = 'normalized' if kind == 'name' else 'level'
                self.assertEqual(native['resolved'], golden[key])
                self.assertEqual(native['resolution']['reasons'], golden['reasons'])
                if kind == 'name':
                    self.assertEqual(native['fallback_calls'], 1)
                    self.assertEqual(native['confidence'], .89)
                else:
                    self.assertEqual(native['fallback_calls'], 2)
                    self.assertEqual(native['resolution'], golden)

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
        cls.backends = {section: NativeStarOcr(PACKAGED_RUNTIME, lambda s=section: s)
                        for section in ('main', 'support')}
        cls.stars = json.loads((PACKAGED_RUNTIME / 'star_catalog.json').read_text(encoding='utf-8'))['stars']
        cls.sections = {s['name']: 'main' if s['kind'] == '主星' else 'support'
                        for s in cls.stars if s['kind'] in ('主星', '辅星')}

    def resolve(self, kind, candidates, section='main'):
        variants = [{'variant': 'test', 'all_results': [dict(text=t, score=c) for t, c in candidates]}]
        return self.backends[section].resolve(kind, variants)


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


    def test_traditional_and_simplified_fields_produce_identical_inventory_json(self):
        from custom.action.star_inventory_sync import compact_inventory
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


if __name__ == "__main__":
    unittest.main()
