"""Local seqme metrics and the upstream sequence validator; no official aggregate is invented."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import seqme as sm

from .audit import module
from .common import dump, sequences, sha


def evaluate(root, output):
    reference = sequences(root / "data/reference/antibacterial.fasta")
    provenance = pd.read_csv(root / "phase1/artifacts/final/provenance.csv")
    groups = {"library": sequences(output / "library.fasta"), "top100": sequences(output / "top.fasta")}
    groups.update({name: rows.sequence.tolist() for name, rows in provenance.groupby("generator", sort=True)})
    metrics = [sm.metrics.Count(), sm.metrics.Uniqueness(), sm.metrics.Novelty(reference), sm.metrics.Length()]
    values = sm.evaluate(groups, metrics, verbose=False)
    values.to_csv(output / "seqme_metrics.csv")
    records = {}
    for name, seqs in groups.items():
        rng = np.random.default_rng(42)
        subset = [seqs[i] for i in sorted(rng.choice(len(seqs), min(2000, len(seqs)), replace=False))]
        score = sm.metrics.Diversity(k=10 if len(subset)>100 else None, seed=42)(subset).value
        records[name] = {metric.name: {"value": float(values.loc[name, (metric.name, "value")]),
                       "deviation": None if pd.isna(values.loc[name, (metric.name, "deviation")])
                       else float(values.loc[name, (metric.name, "deviation")])} for metric in metrics}
        records[name]["Diversity"] = {"value": float(score), "sample_n": len(subset),
                                      "neighbors_k": 10 if len(subset)>100 else None, "seed": 42}
        print(f"seqme: {name} evaluated ({len(seqs)} sequences)", flush=True)
    dump(output / "seqme_metrics.json", {"seqme_version": importlib.metadata.version("seqme"),
         "groups": records, "official_aggregation_score": None,
         "scope": "Local diagnostics. Count/uniqueness/exact novelty/length use full groups; diversity uses a documented fixed subset.",
         "official_protocol_status": "No finalized scoring configuration in the inspected starter kits; do not infer it from this subset.",
         "diversity_distance": "seqme normalized Levenshtein edit distance; differs from organizer Indel novelty ratio"})
    upstream = module("amp_organizer_validator", root / "competition-template/scripts/verify_submission.py")
    library = upstream._verify_sequences(output / "library.fasta")
    upstream._verify_top(output / "top.fasta", library, 100)
    upstream._verify_no_overlap(library, set(reference))
    upstream._veritfy_max_simularity(set(groups["top100"]), set(reference), .8)
    dump(output / "organizer_validation.json", {"passed": True,
         "validator_sha256": sha(root / "competition-template/scripts/verify_submission.py"),
         "library_sha256": sha(output / "library.fasta"), "top_sha256": sha(output / "top.fasta"),
         "scope": "Upstream local sequence checks. No remote clone, public submission, or license approval claimed."})
    print("Upstream local sequence validator: passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
