from pathlib import Path
import urllib.request
import json
import hashlib
root = Path(__file__).resolve().parent / 'artifacts/source'
root.mkdir(parents=True, exist_ok=True)
repo = 'zswitten/Antimicrobial-Peptides'
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'AMP-research-data-audit'}), timeout=90) as r:
        return r.read()
meta = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/master'))
revision = meta['sha']
manifest = {'repository':repo,'revision':revision,'files':{}}
for name in ['data/grampa.csv','README.md','LICENSE']:
    url = f'https://raw.githubusercontent.com/{repo}/{revision}/{name}'
    try:
        content = fetch(url)
    except urllib.error.HTTPError as error:
        if name == 'LICENSE' and error.code == 404:
            manifest['license_note'] = 'No root LICENSE at this revision; upstream data terms require source-specific review.'
            continue
        raise
    (root / Path(name).name).write_bytes(content)
    manifest['files'][name] = {'url':url,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
(root / 'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'))
paths = [x['path'] for x in tree['tree'] if x['path'].startswith('src/') and x['path'].endswith('.py')]
for name in paths:
    content = fetch(f'https://raw.githubusercontent.com/{repo}/{revision}/{name}')
    if b'log10' in content or b'np.log' in content:
        target = root / 'provenance_code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print('Saved unit provenance:', name)
