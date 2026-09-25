"""Public repository entry point: acquire pinned assets, then run frozen inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

from .common import require, sha
from .pipeline import main as run_pipeline


def asset_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root), "Asset path escapes repository")
    require(path != root, "Asset path is a directory")
    return path


def fetch_asset(root, item):
    path = asset_path(root, item['path'])
    if path.is_file():
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                f"Existing asset differs; preserved for inspection: {item['path']}")
        return
    url = urllib.parse.urlparse(item['url'])
    require(url.scheme == 'https' and url.hostname in {
        'huggingface.co', 'raw.githubusercontent.com', 'media.githubusercontent.com',
    }, "Unexpected asset download origin")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    require(not temporary.exists(), f"Partial asset exists; inspect/remove it before retrying: {temporary}")
    print(f"Downloading {item['path']} ({item['bytes'] / 2**20:.1f} MiB)", flush=True)
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'AMP-reproducible-submission'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        require(temporary.stat().st_size == item['bytes'] and sha(temporary) == item['sha256'],
                f"Downloaded asset failed SHA-256 verification: {item['path']}")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path('generate'))
    parser.add_argument('--mode', choices=['checkpoints', 'cached'], default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    require(args.seed == 42, 'This frozen release uses default seed 42; new seeds require new scoring')
    root = args.root.resolve()
    require((root / 'phase1/pyproject.toml').is_file() and (root / 'phase2/core.py').is_file(),
            'Cannot locate repository; provide --root')
    assets = json.loads((root / 'assets/downloads.json').read_text(encoding='utf-8'))
    for item in assets['files']:
        if args.mode == 'cached' and item['path'] in {
            'models/protgpt2/pytorch_model.bin', 'starter-kits/ampdiffusion/checkpoint/model.pt',
        }:
            continue
        fetch_asset(root, item)
    run_pipeline(['--root', str(root), '--output', str(args.output), '--mode', args.mode, '--seed', str(args.seed)])


if __name__ == '__main__':
    main()
