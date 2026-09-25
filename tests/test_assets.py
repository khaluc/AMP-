import hashlib
from pathlib import Path
import pytest
from amp_completion.submission import asset_path, fetch_asset


def test_download_paths_cannot_escape_repository(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        asset_path(tmp_path, '../outside.bin')
    with pytest.raises(ValueError, match='escapes'):
        asset_path(tmp_path, str(tmp_path.parent / 'outside.bin'))


def test_existing_asset_is_verified_and_never_overwritten(tmp_path):
    path = tmp_path / 'model.bin'
    path.write_bytes(b'original')
    item = {'path': 'model.bin', 'url': 'https://example.invalid/model', 'bytes': 8,
            'sha256': hashlib.sha256(b'original').hexdigest()}
    fetch_asset(tmp_path, item)
    item['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Existing asset differs'):
        fetch_asset(tmp_path, item)
    assert path.read_bytes() == b'original'


def test_unexpected_download_host_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='origin'):
        fetch_asset(tmp_path, {'path': 'model.bin', 'url': 'http://example.invalid',
                              'bytes': 1, 'sha256': '0' * 64})
    assert not (tmp_path / 'model.bin').exists()
