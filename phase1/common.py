from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent
AA = "ACDEFGHIKLMNPQRSTVWY"


def fasta(path):
    result, parts = [], []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(">"):
            if parts:
                result.append("".join(parts).upper())
                parts = []
        elif line:
            parts.append(line)
    if parts:
        result.append("".join(parts).upper())
    return result


def valid(seq):
    return 8 <= len(seq) <= 50 and set(seq) <= set(AA)


def write_fasta(path, sequences, prefix="seq"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for i, seq in enumerate(sequences, 1):
            stream.write(f">{prefix}_{i:06d}\n{seq}\n")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_sequences(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            seq = row.get("Sequence", row.get("sequence", "")).strip().upper()
            if seq:
                yield seq


def split(sequences):
    # Sequence-level deterministic split; not a homology-independent evaluation.
    train, val = [], []
    for seq in sorted(set(sequences)):
        bucket = int(hashlib.sha256(("phase1-v1:" + seq).encode()).hexdigest()[:8], 16) % 10
        (val if bucket == 0 else train).append(seq)
    return train, val


def prepare():
    directory = WORK / "artifacts/data"
    hyd = ROOT / "starter-kits/hydramp/data/training"
    diffusion = ROOT / "starter-kits/ampdiffusion/data/training/training.fasta"
    # Reserve all available MIC sequences before fitting our NEW models.
    # Published diffusion weights may already have seen them; disclose that separately.
    reserved = set(csv_sequences(hyd / "mic_data.csv"))
    reserved.update(fasta(hyd / "dbaasp/dbaasp_clean.fasta"))
    amp_sources = [diffusion, hyd / "unlabelled_positive.csv", hyd / "veltri_positive.csv"]
    provenance = {}
    for path in amp_sources:
        sequences = fasta(path) if path.suffix == ".fasta" else list(csv_sequences(path))
        for seq in sequences:
            if valid(seq) and seq not in reserved:
                provenance.setdefault(seq, []).append(path.relative_to(ROOT).as_posix())
    train, val = split(provenance)
    # Remove even exact AMP validation overlap from LM pretraining.
    uniprot_path = hyd / "Uniprot_0_25_train.csv"
    val_set = set(val)
    uniprot = [s for s in csv_sequences(uniprot_path) if valid(s) and s not in reserved and s not in val_set]
    pretrain, preval = split(uniprot)
    for name, sequences in [("amp_train", train), ("amp_val", val), ("uniprot_train", pretrain), ("uniprot_val", preval), ("reserved_mic", sorted(reserved))]:
        write_fasta(directory / f"{name}.fasta", sequences, name)
    with (directory / "amp_provenance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "split", "sources"])
        train_set = set(train)
        for seq in sorted(provenance):
            writer.writerow([seq, "train" if seq in train_set else "validation", "|".join(provenance[seq])])
    sources = amp_sources + [uniprot_path, hyd / "mic_data.csv", hyd / "dbaasp/dbaasp_clean.fasta"]
    report = {
        "amp_train": len(train), "amp_validation": len(val), "uniprot_train": len(pretrain),
        "uniprot_validation": len(preval), "reserved_mic_unique": len(reserved),
        "split": "SHA256 sequence split, 90/10; NOT a homology-independent test",
        "reservation": "Exact MIC sequence exclusion for newly trained models only; no calibration guarantee",
        "published_diffusion_training_overlap": "Not removed from pre-existing checkpoint; audit before Phase 2",
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sources],
    }
    dump(directory / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def accept(sequences, references, seen, limit):
    accepted = []
    for seq in sequences:
        if valid(seq) and seq not in references and seq not in seen:
            seen.add(seq)
            accepted.append(seq)
            if len(accepted) == limit:
                break
    return accepted
