"""Create a private local reproducibility archive, including frozen model assets.

This does not publish a repository or certify redistribution rights.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

from .common import dump, read_json, require, sha


def build(root, completion, results, destination):
    root, completion, results = root.resolve(), completion.resolve(), results.resolve()
    require(not destination.exists(), "Archive already exists; preserve it and choose a new path")
    assets = {}

    def add(path, relative=None):
        path = path.resolve()
        require(path.is_file(), f"Missing archive input: {path}")
        name = relative or path.relative_to(root).as_posix()
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid archive member")
        require(name not in assets or assets[name] == path, f"Duplicate archive member: {name}")
        assets[name] = path

    def tree(relative, suffixes=None):
        base = root / relative
        for path in sorted(base.rglob("*")):
            if any(part in {".git", "__pycache__", ".venv"} for part in path.relative_to(base).parts):
                continue
            if path.is_file() and (suffixes is None or path.suffix in suffixes):
                add(path)

    # Original code and lock files remain byte-identical for manifest verification.
    for folder in ["phase1", "phase2", "phase3", "protein_lm"]:
        for path in sorted((root / folder).iterdir()):
            if path.is_file() and (path.suffix in {".py", ".md", ".toml", ".lock"} or path.name == ".python-version"):
                add(path)
    for relative in read_json(root / "phase2/artifacts/run_manifest.json"):
        add(root / relative)
    directories = ["models/protgpt2", "models/torch/hub/checkpoints", "phase1/artifacts/data",
        "phase1/artifacts/checkpoints", "phase1/artifacts/final", "protein_lm/artifacts/adapter",
        "protein_lm/artifacts/data", "phase2/artifacts/library_features", "phase2/artifacts/source", "phase3/artifacts"]
    for relative in directories:
        tree(relative)
    tree("starter-kits/ampdiffusion/src", {".py"})
    tree("competition-template/scripts", {".py"})
    for path in (root / "phase2/artifacts").iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".npy", ".joblib"}:
            add(path)
    for relative in ["data/reference/antibacterial.fasta", "docs/sources.json",
        "starter-kits/ampdiffusion/checkpoint/model.pt", "starter-kits/ampdiffusion/LICENSE",
        "starter-kits/ampdiffusion/README.md", "starter-kits/hydramp/LICENSE",
        "competition-template/LICENSE", "competition-template/README.md",
        "phase1/artifacts/pools/diffusion_ddim100.fasta", "phase1/artifacts/pools/diffusion_ddim100.progress.json",
        "phase1/artifacts/multiobjective/gflownet.pt", "phase1/artifacts/multiobjective/metadata.json",
        "phase1/artifacts/multiobjective/pool.fasta", "phase1/artifacts/multiobjective/pool.json",
        "protein_lm/artifacts/base_manifest.json", "protein_lm/artifacts/hub_metadata.json",
        "protein_lm/artifacts/pool/protgpt2_lora.fasta", "protein_lm/artifacts/pool/protgpt2_lora.progress.json"]:
        add(root / relative)
    for path in sorted(completion.rglob("*")):
        relative = path.relative_to(completion)
        if any(part in {".venv", ".eval-deps", "__pycache__", "artifacts", "releases"} for part in relative.parts):
            continue
        if path.is_file():
            add(path, "completion/" + relative.as_posix())
            if relative.parts[0] == "src":
                add(path, relative.as_posix())
    for name in ["pyproject.toml", "uv.lock", ".python-version"]:
        add(completion / name, name)
    add(completion / "README.md", "README.md")
    for path in sorted(results.rglob("*")):
        if path.is_file():
            add(path, "completion/artifacts/" + path.relative_to(results).as_posix())
    for name in ["METHODS.md", "DATA_AND_LICENSES.md", "MD_READINESS.md"]:
        add(completion / name, name)
    total = sum(path.stat().st_size for path in assets.values())
    require(shutil.disk_usage(destination.parent).free > total + 1024**3, "Not enough free disk for archive")
    print(f"Hashing {len(assets)} inputs ({total/2**30:.2f} GiB)", flush=True)
    manifest = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in sorted(assets.items())}
    metadata = {"scope": "Private local checkpoint-reproduction archive; not a published or license-cleared submission",
        "members": manifest, "asset_bytes": total, "contains_virtual_environments": False,
        "fresh_machine_replay_completed": False}
    manifest_path = destination.with_suffix(".manifest.json")
    dump(manifest_path, metadata)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(assets.items()):
            archive.write(path, name)
        archive.write(manifest_path, "archive-manifest.json")
    with zipfile.ZipFile(destination) as archive:
        require(archive.testzip() is None, "ZIP CRC validation failed")
        for name, expected in manifest.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            require(digest.hexdigest() == expected["sha256"], f"Archived bytes differ from manifest: {name}")
    dump(destination.with_suffix(".verification.json"), {"archive_sha256": sha(destination),
         "member_count": len(assets)+1, "zip_crc_passed": True, "all_member_sha256_verified": True, "bytes": destination.stat().st_size,
         "public_distribution_cleared": False})
    print(f"Archive verified: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.completion, args.results, args.output)


if __name__ == "__main__":
    main()
