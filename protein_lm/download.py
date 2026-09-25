"""Download the author's ProtGPT2 checkpoint at a pinned revision; do not execute remote code."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download, HfApi
import hashlib
import json

REPO = "nferruz/ProtGPT2"
REV = "ff981fc96771d6f6d3b6453f94772d3cb6c7b90b"
TARGET = ROOT / "models/protgpt2"
files = ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json", "README.md", "LICENSE"]
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
snapshot_download(REPO, revision=REV, local_dir=TARGET, allow_patterns=files, max_workers=2)
manifest = []
for entry in info.siblings:
    if entry.rfilename not in files:
        continue
    path = TARGET / entry.rfilename
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = entry.lfs.sha256 if entry.lfs else None
    if expected and actual != expected:
        raise RuntimeError(f"Checksum mismatch: {path.name}")
    manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": actual, "hub_lfs_sha256": expected})
report = {"repo": REPO, "revision": REV, "files": manifest}
(ROOT / "protein_lm/artifacts/base_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
