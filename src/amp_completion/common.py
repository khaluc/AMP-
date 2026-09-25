"""Small, dependency-free IO and validation helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fasta(path):
    records, header, parts = [], None, []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line[1:], []
            require(bool(header.strip()), f"Empty FASTA header: {path}")
        else:
            require(header is not None, f"Sequence before FASTA header: {path}")
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    require(all(sequence for _, sequence in records), f"Empty sequence: {path}")
    return records


def sequences(path):
    return [sequence for _, sequence in fasta(path)]


def valid(sequence):
    return 8 <= len(sequence) <= 50 and set(sequence) <= AA


def write_fasta(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for identifier, sequence in records:
            stream.write(f">{identifier}\n{sequence}\n")


def assemble(pools, known, quotas=(16667, 16667, 16666)):
    require(len(pools) == len(quotas), "Pool/quota mismatch")
    selected, seen = [], set()
    for pool, quota in zip(pools, quotas):
        require(quota > 0, "Quota must be positive")
        kept = []
        for sequence in pool:
            if valid(sequence) and sequence not in known and sequence not in seen:
                seen.add(sequence)
                kept.append(sequence)
                if len(kept) == quota:
                    break
        require(len(kept) == quota, "Insufficient pool after exclusions; thresholds and quotas unchanged")
        selected.extend(kept)
    return selected
