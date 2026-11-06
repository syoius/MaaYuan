"""No-device checks for completeness, per-instance transfer and durable HTTP retries."""
import copy
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
from custom.action.inventory_reporting import BoundAccount, UploadSettings

spec = importlib.util.spec_from_file_location('star_inventory_sync_test_target', ROOT / 'agent/custom/action/star_inventory_sync.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)
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

    def test_incomplete_or_unknown_or_invalid_never_sends(self):
        def variants():
            value = local_inventory(); value['complete'] = False; yield value
            value = local_inventory(); value['sections']['main']['complete'] = False; yield value
            value = local_inventory(); value['sections']['support']['count'] = 2; yield value
            value = local_inventory(); value['experience']['complete'] = False; yield value
            for unknown in (None, -1, True, '0', 0.5, 2**31):
                value = local_inventory(); value['experience']['orangeCount'] = unknown; yield value
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
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                if len(received) == 1:
                    self.connection.close()
                    return
                body = json.dumps(dict(data=dict(status='synced', scan_id=received[-1]['scan_id'], account_id='test', generation=1))).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        settings = UploadSettings('仅保存到本地', 'secret-test-token', 'http://127.0.0.1:' + str(server.server_port))
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(sync, 'read_upload_settings', return_value=settings), patch.object(sync, 'get_bound_account', return_value=BoundAccount('test', 'test')):
                root = Path(directory)
                (root / 'inventory.json').write_text(json.dumps(local_inventory()), encoding='utf-8')
                first = sync.sync_saved_inventory(object(), root)
                self.assertFalse(first['success'])
                second = sync.sync_saved_inventory(object(), root)
                self.assertTrue(second['success'])
                self.assertEqual(received[0], received[1])
                self.assertNotIn(settings.token, ''.join(p.read_text(encoding='utf-8') for p in root.iterdir()))
                # A new token bound to another account cannot redirect an old batch.
                with patch.object(sync, 'get_bound_account', return_value=BoundAccount('other', 'other')), patch.object(sync, 'post_inventory') as post:
                    with self.assertRaisesRegex(ValueError, '账号'):
                        sync.sync_saved_inventory(object(), root)
                    post.assert_not_called()
        finally:
            server.shutdown(); server.server_close(); worker.join()

    def test_saved_batch_cannot_be_mutated_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); local = local_inventory()
            (root / 'inventory.json').write_text(json.dumps(local), encoding='utf-8')
            (root / 'inventory-sync.json').write_text(json.dumps(sync.compact_inventory(local, SCAN)), encoding='utf-8')
            local['experience']['orangeCount'] = 2
            (root / 'inventory.json').write_text(json.dumps(local), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '不一致'):
                sync.sync_saved_inventory(object(), root)


if __name__ == '__main__':
    unittest.main()
