"""Independent FASTA/provenance audit and descriptive diagnostics, not official seqme scores."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from common import ROOT, WORK, AA, fasta, valid, dump, sha

directory = WORK / "artifacts/library"
library = fasta(directory / "library.fasta")
references = set(fasta(ROOT / "data/reference/antibacterial.fasta"))
with (directory / "provenance.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(library) == len(set(library)) == 50000
assert all(valid(s) for s in library)
assert not set(library) & references
assert [r["sequence"] for r in rows] == library
assert [r["id"] for r in rows] == [f"seq_{i:06d}" for i in range(1, 50001)]
quotas = {"diffusion": 16667, "lm": 16667, "gflownet": 16666}
assert Counter(r["generator"] for r in rows) == quotas
report = {"count": len(library), "library_sha256": sha(directory / "library.fasta"), "structural_checks_passed": True, "generators": {}}
merge_report = json.loads((directory / "validation.json").read_text())
pool_sets = {name: set(fasta(merge_report[name]["pool_path"])) for name in quotas}
report["raw_pool_pairwise_exact_overlap"] = {
    f"{a}__{b}": len(pool_sets[a] & pool_sets[b])
    for a, b in [("diffusion", "lm"), ("diffusion", "gflownet"), ("lm", "gflownet")]
}
for name, quota in quotas.items():
    sequences = [r["sequence"] for r in rows if r["generator"] == name]
    residues = Counter("".join(sequences))
    total = sum(residues.values())
    entropy = -sum((n / total) * math.log2(n / total) for n in residues.values())
    trigrams = {s[i:i+3] for s in sequences for i in range(len(s) - 2)}
    pool_path = Path(merge_report[name]["pool_path"])
    progress = json.loads(pool_path.with_suffix(".progress.json").read_text())
    report["generators"][name] = {
        "count": len(sequences), "mean_length": sum(map(len, sequences)) / quota,
        "residue_entropy_bits": entropy, "observed_distinct_trigrams": len(trigrams),
        "amino_acid_fractions": {a: residues[a] / total for a in AA},
        "pool_acceptance_fraction": progress["accepted"] / progress["attempts"],
        "sampler": progress["settings"]["sampler"],
        "generation_seed": progress["settings"]["seed"],
        "exact_training_overlap": len(set(sequences) & set(fasta(WORK / "artifacts/data/amp_train.fasta"))),
    }
report["limitations"] = "Descriptive sequence diagnostics only. No official aggregation score, activity evidence, MoA labels or conformal guarantee."
dump(directory / "independent_audit.json", report)
print(json.dumps(report, indent=2), flush=True)
