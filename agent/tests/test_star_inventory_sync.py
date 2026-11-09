from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError
import json
import numpy as np
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from custom.action import star_full_local_ocr_shadow as session
from custom.action import star_inventory_sync as sync
from custom.action.inventory_reporting import BoundAccount
from custom.action.inventory_reporting import UploadSettings
from star_fixture_inputs import FakeGeometry


SCAN = '11111111-1111-4111-8111-111111111111'

def local_inventory():
    main = dict(section='main', name='天府', quality='橙', level=60, sourceImageId='private.png')
    support = dict(section='support', name='文曲', quality='紫', level=20)
    return dict(schemaVersion=2, complete=True, sections={
        'main': dict(complete=True, count=2, items=[main.copy(), main.copy()]),
        'support': dict(complete=True, count=1, items=[support])},
        experience=dict(complete=True, structured=True, orangeCount=0, purpleCount=525, whiteCount=91))

class InventorySyncTests(unittest.TestCase):
    def test_task_resumes_pending_batch_without_rescanning_and_reports_failure(self):
        class Context:
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); batch = root / 'saved'; batch.mkdir()
            (root / 'latest.json').write_text(json.dumps(dict(directory='saved', outcome='cloud_sync_failed')), encoding='utf-8')
            with patch.object(sync, 'REPO', root), patch.object(sync, 'Context', Context), patch.object(sync.Library, 'is_agent_server', return_value=True), patch.object(sync, 'FullShadowSession') as scanner, patch.object(sync, 'sync_saved_inventory', return_value=dict(success=False, message='permission required')) as send:
                result = sync.StarInventorySync().run(Context(), SimpleNamespace(custom_action_param=json.dumps(dict(output='.', shadow=True))))
                self.assertFalse(result)
                scanner.assert_not_called()
                self.assertEqual(send.call_args.args[1], batch)
                self.assertEqual(json.loads((root / 'latest.json').read_text(encoding='utf-8'))['outcome'], 'cloud_sync_failed')

    def test_all_instances_counts_and_no_diagnostics(self):
        payload = sync.compact_inventory(local_inventory(), SCAN)
        self.assertEqual(payload['counts'], dict(main=2, support=1))
        self.assertEqual(len(payload['inventory']), 3)
        self.assertEqual(payload['inventory'][0], payload['inventory'][1])
        self.assertEqual(payload['experience'], dict(orange=0, purple=525, white=91))
        self.assertEqual(set(payload), {'format', 'version', 'scan_id', 'complete', 'counts', 'inventory', 'experience'})
        self.assertEqual(set(payload['inventory'][0]), {'kind', 'name', 'quality', 'level'})

    def test_bag_uses_observed_capacity_and_total_but_keeps_diagnostics_local(self):
        local = local_inventory(); local['bag'] = dict(complete=True, currentCount=3, capacity=500, file='private.png', roi=[1,2,3,4])
        payload = sync.compact_inventory(local, SCAN)
        self.assertEqual(payload['bag'], dict(current_count=3, capacity=500))
        self.assertEqual(set(payload['bag']), {'current_count','capacity'})
        for current, capacity in ((2,500),(3,2),(3,0),(3,True),(3,'500'),(True,500)):
            local['bag'].update(currentCount=current, capacity=capacity)
            with self.subTest(current=current,capacity=capacity), self.assertRaises(ValueError):sync.compact_inventory(local, SCAN)
    def test_unknown_bag_does_not_invent_capacity_and_historical_batch_stays_unchanged(self):
        local = local_inventory(); self.assertNotIn('bag', sync.compact_inventory(local, SCAN))
        local['bag'] = dict(complete=False, currentCount=None, capacity=None, warnings=['bag_numbers_unconfirmed'])
        self.assertEqual(sync.compact_inventory(local, SCAN)['bag'], dict(current_count=3, capacity=None))

    def test_incomplete_or_invalid_never_sends(self):
        def variants():
            value = local_inventory(); value['complete'] = False; yield value
            value = local_inventory(); value['sections']['support']['count'] = 2; yield value
            for incomplete in (('main',), ('support',), ('main', 'support')):
                value = local_inventory()
                for section in incomplete:
                    value['sections'][section]['complete'] = False
                yield value
            for invalid in (-1, True, '0', 0.5, 2**31):
                value = local_inventory()
                value['experience']['orangeCount'] = invalid
                yield value
            value = local_inventory(); value['sections']['main']['items'][0]['level'] = 61; yield value
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in variants():
                (root / 'inventory.json').write_text(json.dumps(value), encoding='utf-8')
                with self.subTest(value=value), patch.object(sync, 'post_inventory') as post:
                    with self.assertRaises(ValueError):
                        sync.sync_saved_inventory(object(), root)
                    post.assert_not_called()
                self.assertFalse((root / 'inventory-sync.json').exists())

    def test_response_loss_then_retry_reuses_original_body_and_id(self):
        received = []

        def open_request(request, *, timeout):
            received.append(request.data)
            if len(received) == 1:
                raise URLError('response lost')
            payload = json.loads(request.data)
            return BytesIO(json.dumps(dict(data=dict(
                status='synced', scan_id=payload['scan_id'], account_id='test', generation=1,
            ))).encode('utf-8'))

        settings = UploadSettings('仅保存到本地', 'secret-test-token', 'https://sync.invalid')
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(sync, 'read_upload_settings', return_value=settings),
              patch.object(sync, 'get_bound_account', return_value=BoundAccount('test', 'test')),
              patch.object(sync.urllib_request, 'build_opener',
                           return_value=SimpleNamespace(open=open_request))):
            root = Path(directory)
            local = local_inventory()
            local['experience'].update(complete=False, structured=False, purpleCount=None)
            del local['experience']['whiteCount']
            (root / 'inventory.json').write_text(json.dumps(local), encoding='utf-8')
            first = sync.sync_saved_inventory(object(), root)
            self.assertFalse(first['success'])
            second = sync.sync_saved_inventory(object(), root)
            self.assertTrue(second['success'])
            self.assertEqual(len(received), 2)
            self.assertEqual(received[0], received[1])
            self.assertEqual(json.loads(received[0])['experience'], dict(orange=0, purple=None, white=None))
            self.assertNotIn(settings.token, ''.join(p.read_text(encoding='utf-8') for p in root.iterdir()))
            # A new token bound to another account cannot redirect an old batch.
            with patch.object(sync, 'get_bound_account', return_value=BoundAccount('other', 'other')), patch.object(sync, 'post_inventory') as post:
                with self.assertRaisesRegex(ValueError, '账号'):
                    sync.sync_saved_inventory(object(), root)
                post.assert_not_called()

    def test_saved_batch_cannot_be_mutated_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); local = local_inventory()
            (root / 'inventory.json').write_text(json.dumps(local), encoding='utf-8')
            (root / 'inventory-sync.json').write_text(json.dumps(sync.compact_inventory(local, SCAN)), encoding='utf-8')
            local['experience']['orangeCount'] = 2
            (root / 'inventory.json').write_text(json.dumps(local), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '不一致'):
                sync.sync_saved_inventory(object(), root)

class ExperienceRequestTests(unittest.TestCase):
 def test_experience_results_reach_formal_request_without_losing_confirmed_counts(self):
  class Context(SimpleNamespace):
   pass

  cases = (
      ('confirmed', [91, 0, 525], (0, 525, 91), None),
      ('one unknown', [91, 0, None, None], (0, None, 91), 'experience_count_unparsed:purple'),
      ('two unknown', [None, None, 0, None, None], (0, None, None), 'experience_count_unparsed:white'),
      ('all unknown', [None] * 6, (None, None, None), 'experience_counts_incomplete'),
      ('cropped', [], (None, None, None), 'experience_viewport_cropped'),
      ('unavailable', [], (None, None, None), 'experience_scan_failed'),
  )
  settings = UploadSettings('仅保存到本地', 'test-token', 'https://sync.invalid')
  image = np.zeros((1280, 720, 3), np.uint8)
  for label, values, counts, warning in cases:
   with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
    root = Path(name)
    values = iter(values)
    geometry = FakeGeometry().experience_geometry(image)
    if label == 'cropped':
     geometry['coverage'].update(complete=False, warnings=['experience_viewport_cropped'])

    def run_recognition(*args):
     value = next(values)
     if value is None:
      return None
     row = SimpleNamespace(text=str(value), score=.99, box=[0, 0, 3, 3])
     return SimpleNamespace(all_results=[row], raw_detail={}, reco_id=1)

    context = Context(tasker=SimpleNamespace(stopping=False, resource=SimpleNamespace(hash='simulated')),
                      run_recognition=mock.Mock(side_effect=run_recognition))

    def scan(section, frame):
     a.sections[section] = {'complete': True}
     a.inventories[section].items = [dict(section=section, name='test', quality='orange', level=1)]

    def read_bag(frame):
     a.bag = dict(complete=True, currentCount=2, capacity=270, warnings=[])
     return frame

    def switch(section):
     a.progress('switch_' + section)
     a.last_capture = {'file': section + '.png'}
     if section == 'experience' and label == 'unavailable':
      raise RuntimeError('backpack identity or selected section changed')
     return image

    def open_request(request, *, timeout):
     payload = json.loads(request.data)
     return BytesIO(json.dumps(dict(data=dict(
         status='synced', scan_id=payload['scan_id'], account_id='test', generation=1,
     ))).encode('utf-8'))

    with mock.patch.object(session, 'REPO', root):
     a = session.FullShadowSession(context, {'shadow': True, 'output': '.'})
     other = session.FullShadowSession(context, {'shadow': True, 'output': '.'})
    a.prepare = lambda: image
    a.stabilize = lambda frame, *args: frame
    a.scan = scan
    a.read_bag = read_bag
    a.switch = switch
    opener = SimpleNamespace(open=mock.Mock(side_effect=open_request))
    with (mock.patch.object(sync, 'REPO', root),
          mock.patch.object(sync, 'Context', Context),
          mock.patch.object(sync.Library, 'is_agent_server', return_value=True),
          mock.patch.object(sync, 'FullShadowSession', return_value=a),
          mock.patch.object(sync, 'read_upload_settings', return_value=settings),
          mock.patch.object(sync, 'get_bound_account', return_value=BoundAccount('test', 'test')),
          mock.patch.object(sync.urllib_request, 'build_opener', return_value=opener),
          mock.patch('utils.star_experience_ocr.recognize_experience_geometry', return_value=geometry)):
     argv = SimpleNamespace(custom_action_param='{"shadow":true,"output":"."}')
     self.assertTrue(sync.StarInventorySync().run(context, argv))

    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    self.assertEqual((request.method, request.selector), ('POST', sync.SYNC_PATH))
    payload = json.loads(request.data)
    expected = dict(zip(('orange', 'purple', 'white'), counts))
    self.assertEqual(payload['experience'], expected)
    self.assertEqual(payload['counts'], dict(main=1, support=1))
    self.assertEqual(payload, json.loads((a.out/'inventory-sync.json').read_text(encoding='utf-8')))
    inventory = json.loads((a.out/'inventory.json').read_text(encoding='utf-8'))
    self.assertTrue(inventory['complete'])
    details = json.loads((a.out/'experience-results.json').read_text(encoding='utf-8'))
    for record in (inventory['experience'], details):
     self.assertEqual({k: record[k+'Count'] for k in expected}, expected)
     self.assertEqual(record['complete'], warning is None)
     self.assertEqual(record['structured'], warning is None)
     self.assertEqual(record['file'], 'experience.png')
     if warning:
      self.assertIn(warning, record['warnings'])
    self.assertEqual(inventory['experience']['detailsFile'], 'experience-results.json')
    self.assertNotIn('items', inventory['experience'])
    if label == 'cropped':
     self.assertFalse(details['geometry']['coverage']['complete'])
     context.run_recognition.assert_not_called()
    if label == 'unavailable':
     self.assertEqual(details['failure']['stage'], 'switch_experience')
    if label == 'one unknown':
     purple = next(item for item in details['items'] if item['kind'] == 'purple')
     self.assertEqual([variant['reco_id'] for variant in purple['variants']], [None, None])
    self.assertEqual(set(p.name for p in other.out.iterdir()), {'pages'})

class ReceiptSafetyTests(unittest.TestCase):
    def test_wrong_scan_receipt_is_not_a_success(self):
        payload = sync.compact_inventory(local_inventory(), SCAN)
        settings = UploadSettings('仅保存到本地', 'test-token', 'https://sync.invalid')
        body = {'data': {'status': 'synced', 'scan_id': 'other-scan',
                         'account_id': 'test', 'generation': 1}}
        opener = SimpleNamespace(open=lambda *args, **kwargs: BytesIO(json.dumps(body).encode()))
        with patch.object(sync.urllib_request, 'build_opener', return_value=opener):
            result = sync.post_inventory(payload, settings)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'cloud_sync_failed')
        self.assertNotIn('receipt', result)

    def test_http_rejection_does_not_log_sensitive_response_text(self):
        settings = UploadSettings('仅保存到本地', 'test-token', 'https://sync.invalid')
        sensitive = 'private-response-marker'
        body = json.dumps({'error': {'code': sensitive, 'message': sensitive}}).encode()
        rejection = HTTPError('https://sync.invalid', 422, sensitive, {}, BytesIO(body))
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(sync, 'REPO', Path(directory)),
              patch.object(sync, 'read_upload_settings', return_value=settings),
              patch.object(sync, 'get_bound_account', return_value=BoundAccount('test', 'test')),
              patch.object(sync.urllib_request, 'build_opener') as opener,
              patch.object(sync, 'logger') as log):
            class Context:
                pass
            root = Path(directory)
            batch = root / 'saved'
            batch.mkdir()
            (batch/'inventory.json').write_text(json.dumps(local_inventory()), encoding='utf-8')
            (root/'latest.json').write_text(json.dumps({'directory': 'saved',
                'outcome': 'cloud_sync_pending'}), encoding='utf-8')
            opener.return_value.open.side_effect = rejection
            with patch.object(sync, 'Context', Context), patch.object(sync.Library, 'is_agent_server', return_value=True):
                self.assertFalse(sync.StarInventorySync().run(Context(), SimpleNamespace(custom_action_param='{"output":"."}')))
            result = (batch/'sync-result.json').read_text(encoding='utf-8')
            self.assertEqual(json.loads(result)['http_status'], 422)
            self.assertEqual(json.loads((root/'latest.json').read_text(encoding='utf-8'))['outcome'],
                             'cloud_sync_failed')
            self.assertNotIn(sensitive, result)
            self.assertNotIn(sensitive, repr(log.mock_calls))
            log.error.assert_called()


if __name__ == "__main__":
    unittest.main()
