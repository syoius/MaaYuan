"""Opt-in cross-project test client; accepts only an ephemeral loopback backend."""
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
from custom.action.inventory_reporting import UploadSettings
from custom.action.star_inventory_sync import compact_inventory, post_inventory


def main():
    parsed = urlparse(sys.argv[1])
    assert parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and parsed.port and parsed.port != 8080
    fixture = Path(os.environ.get('MAAYUAN_SYNC_FIXTURE', str(ROOT / '.ux/audits/star-experience-ui-integration-20261031/gui-run/inventory.json')))
    payload = compact_inventory(json.loads(fixture.read_text(encoding='utf-8')), '11111111-1111-4111-8111-111111111111')
    assert payload['counts'] == dict(main=69, support=105)
    assert payload['experience'] == dict(orange=1, purple=525, white=91)
    settings = UploadSettings('自动上报', 'isolated-sync-token', sys.argv[1])
    first = post_inventory(payload, settings)
    assert first['success'], first
    # Model a caller that lost the first successful response: discard it and resend
    # the exact same batch through the production sender HTTP implementation.
    retry = post_inventory(payload, settings)
    assert retry == first, (first, retry)
    assert retry['receipt']['account_id'] == 'a'
    assert retry['receipt']['generation'] == 1
    assert retry['receipt']['inheritance_pending'] is True
    print('MAAYUAN_SYNC_E2E_OK instances=174 main=69 support=105 exp=1/525/91 generation=1 same_batch_retry=ok')


if __name__ == '__main__':
    main()
